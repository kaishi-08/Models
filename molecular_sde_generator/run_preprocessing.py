# run_preprocessing.py - Script tự động chạy preprocessing

import os
import sys
import pickle
import shutil
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')
sys.path.append('scripts')

def analyze_and_preprocess():
    """Tự động analyze format và chạy preprocessing"""
    print("🔄 Auto-analyzing and preprocessing CrossDock data...")
    
    # Check index format
    index_path = "data/crossdocked_pocket10/index.pkl"
    
    try:
        with open(index_path, 'rb') as f:
            index_data = pickle.load(f)
        
        print(f"📊 Index loaded: {len(index_data)} entries")
        
        # Analyze first entry to determine format
        if len(index_data) > 0:
            first_entry = index_data[0]
            print(f"   First entry type: {type(first_entry)}")
            
            if isinstance(first_entry, (list, tuple)):
                print(f"   Entry length: {len(first_entry)}")
                for i, item in enumerate(first_entry[:3]):
                    print(f"      [{i}]: {type(item)} - {str(item)[:50]}")
                
                # Run tuple format preprocessing
                return run_tuple_preprocessing(index_data)
            
            elif isinstance(first_entry, str):
                print(f"   String entry: {first_entry[:50]}")
                # Run string format preprocessing
                return run_string_preprocessing(index_data)
            
            else:
                print(f"❌ Unknown format: {type(first_entry)}")
                return False
        
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        return False

def run_tuple_preprocessing(index_data):
    """Chạy preprocessing cho tuple format"""
    print("\n🔄 Running tuple format preprocessing...")
    
    # Create processed directory
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    
    # Import processor components
    try:
        from preprocess_crossdock_data import CrossDockDataProcessor
    except:
        print("❌ Could not import CrossDockDataProcessor")
        return False
    
    # Create custom processor
    processor = CustomTupleProcessor(
        data_path="data/crossdocked_pocket10",
        pocket_radius=10.0,
        min_atoms=5,
        max_atoms=50,
        max_pocket_atoms=500
    )
    
    # Process data
    try:
        processed_data = processor.process_tuple_dataset(index_data, str(output_dir))
        print(f"✅ Preprocessing completed: {len(processed_data)} complexes")
        return True
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_string_preprocessing(index_data):
    """Chạy preprocessing cho string format"""
    print("\n🔄 Running string format preprocessing...")
    # Implementation for string format
    return False

class CustomTupleProcessor:
    """Custom processor cho tuple format"""
    
    def __init__(self, data_path: str, pocket_radius: float = 10.0,
                 min_atoms: int = 5, max_atoms: int = 50, max_pocket_atoms: int = 500):
        self.data_path = data_path
        self.pocket_radius = pocket_radius
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.max_pocket_atoms = max_pocket_atoms
        
        # Initialize feature mappings
        self.setup_feature_mappings()
    
    def setup_feature_mappings(self):
        """Setup atom and residue type mappings"""
        self.atom_types = {
            'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5,
            'Cl': 6, 'Br': 7, 'I': 8, 'H': 9, 'Unknown': 10
        }
        
        self.residue_types = {
            'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
            'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
            'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
            'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
            'Unknown': 20
        }
        
        self.bond_types = {
            1: 0,  # Single
            2: 1,  # Double  
            3: 2,  # Triple
            12: 3  # Aromatic
        }
    
    def process_tuple_dataset(self, index_data, output_dir: str):
        """Process tuple format dataset"""
        processed_data = []
        failed_count = 0
        
        print(f"Processing {len(index_data)} entries...")
        
        for i, entry in enumerate(tqdm(index_data)):
            try:
                complex_data = self.process_tuple_entry(entry)
                if complex_data is not None:
                    processed_data.append(complex_data)
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                if i < 5:  # Show first few errors for debugging
                    print(f"Entry {i} failed: {e}")
                continue
            
            # Save intermediate results
            if (i + 1) % 5000 == 0:
                temp_file = os.path.join(output_dir, f"temp_data_{i+1}.pkl")
                with open(temp_file, 'wb') as f:
                    pickle.dump(processed_data, f)
                print(f"Saved intermediate results: {len(processed_data)} complexes")
        
        print(f"Successfully processed: {len(processed_data)}")
        print(f"Failed: {failed_count}")
        print(f"Success rate: {len(processed_data)/(len(processed_data)+failed_count)*100:.1f}%")
        
        # Split and save data
        self.split_and_save_data(processed_data, output_dir)
        
        return processed_data
    
    def process_tuple_entry(self, entry):
        """Process một tuple entry"""
        try:
            # Extract file paths from tuple
            protein_path = None
            ligand_path = None
            
            if isinstance(entry, (list, tuple)):
                for item in entry:
                    if isinstance(item, str):
                        full_path = os.path.join(self.data_path, item)
                        if '.pdb' in item and os.path.exists(full_path):
                            protein_path = full_path
                        elif ('.sdf' in item or '.mol2' in item) and os.path.exists(full_path):
                            ligand_path = full_path
            
            if not protein_path or not ligand_path:
                return None
            
            # Process ligand
            ligand_data = self.process_ligand(ligand_path)
            if ligand_data is None:
                return None
            
            # Process pocket
            pocket_data = self.process_pocket(protein_path, ligand_data['positions'])
            if pocket_data is None:
                return None
            
            return {
                'ligand': ligand_data,
                'pocket': pocket_data,
                'protein_file': protein_path,
                'ligand_file': ligand_path
            }
            
        except Exception as e:
            return None
    
    def process_ligand(self, ligand_file: str):
        """Process ligand molecule"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            # Load molecule
            if ligand_file.endswith('.sdf'):
                mol = Chem.SDMolSupplier(ligand_file)[0]
            elif ligand_file.endswith('.mol2'):
                mol = Chem.MolFromMol2File(ligand_file)
            else:
                return None
            
            if mol is None:
                return None
            
            # Check atom count
            num_atoms = mol.GetNumAtoms()
            if num_atoms < self.min_atoms or num_atoms > self.max_atoms:
                return None
            
            # Extract features
            atom_features = self.get_atom_features(mol)
            positions = self.get_positions(mol)
            edge_index, edge_features = self.get_bonds(mol)
            
            return {
                'atom_features': atom_features,
                'positions': positions,
                'edge_index': edge_index,
                'edge_features': edge_features,
                'smiles': Chem.MolToSmiles(mol),
                'num_atoms': num_atoms
            }
            
        except Exception as e:
            return None
    
    def process_pocket(self, protein_file: str, ligand_positions):
        """Process protein pocket"""
        try:
            from Bio.PDB import PDBParser
            import numpy as np
            
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', protein_file)
            
            # Calculate ligand center
            ligand_center = np.mean(ligand_positions, axis=0)
            
            # Extract pocket atoms
            pocket_atoms = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            atom_coord = np.array(atom.coord)
                            distance = np.linalg.norm(atom_coord - ligand_center)
                            
                            if distance <= self.pocket_radius:
                                pocket_atoms.append({
                                    'coord': atom_coord,
                                    'element': atom.element,
                                    'residue': residue.resname,
                                    'residue_id': residue.id[1],
                                    'is_backbone': atom.name in ['N', 'CA', 'C', 'O']
                                })
            
            if len(pocket_atoms) == 0 or len(pocket_atoms) > self.max_pocket_atoms:
                return None
            
            # Extract features
            atom_features = self.get_pocket_features(pocket_atoms)
            positions = np.array([atom['coord'] for atom in pocket_atoms])
            edge_index, edge_features = self.get_pocket_bonds(positions)
            
            return {
                'atom_features': atom_features,
                'positions': positions,
                'edge_index': edge_index,
                'edge_features': edge_features,
                'num_atoms': len(pocket_atoms)
            }
            
        except Exception as e:
            return None
    
    def get_atom_features(self, mol):
        """Extract atom features"""
        import numpy as np
        
        features = []
        for atom in mol.GetAtoms():
            feature_vector = [
                self.atom_types.get(atom.GetSymbol(), self.atom_types['Unknown']),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetTotalNumHs(),
                int(atom.IsInRing()),
                atom.GetAtomicNum()
            ]
            features.append(feature_vector)
        
        return np.array(features, dtype=np.float32)
    
    def get_positions(self, mol):
        """Extract 3D positions"""
        import numpy as np
        
        conf = mol.GetConformer()
        positions = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            positions.append([pos.x, pos.y, pos.z])
        
        return np.array(positions, dtype=np.float32)
    
    def get_bonds(self, mol):
        """Extract bond information"""
        import numpy as np
        
        edge_index = []
        edge_features = []
        
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = self.bond_types.get(int(bond.GetBondType()), 0)
            
            # Add both directions
            edge_index.extend([[start, end], [end, start]])
            
            bond_feature = [
                bond_type,
                int(bond.GetIsConjugated()),
                int(bond.IsInRing()),
                int(bond.GetStereo())
            ]
            edge_features.extend([bond_feature, bond_feature])
        
        edge_index = np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64)
        edge_features = np.array(edge_features, dtype=np.float32) if edge_features else np.zeros((0, 4), dtype=np.float32)
        
        return edge_index, edge_features
    
    def get_pocket_features(self, pocket_atoms):
        """Extract pocket atom features"""
        import numpy as np
        
        features = []
        for atom in pocket_atoms:
            feature_vector = [
                self.atom_types.get(atom['element'], self.atom_types['Unknown']),
                self.residue_types.get(atom['residue'], self.residue_types['Unknown']),
                float(atom['is_backbone']),
                atom['residue_id'] % 100,
                0.0, 0.0, 0.0, 0.0  # Placeholders
            ]
            features.append(feature_vector)
        
        return np.array(features, dtype=np.float32)
    
    def get_pocket_bonds(self, positions, cutoff=5.0):
        """Create pocket connectivity"""
        import numpy as np
        from scipy.spatial.distance import cdist
        
        if len(positions) == 0:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 1), dtype=np.float32)
        
        distances = cdist(positions, positions)
        
        edge_index = []
        edge_features = []
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if distances[i, j] <= cutoff:
                    edge_index.extend([[i, j], [j, i]])
                    edge_features.extend([[distances[i, j]], [distances[i, j]]])
        
        edge_index = np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64)
        edge_features = np.array(edge_features, dtype=np.float32) if edge_features else np.zeros((0, 1), dtype=np.float32)
        
        return edge_index, edge_features
    
    def split_and_save_data(self, data, output_dir):
        """Split and save processed data"""
        import numpy as np
        
        np.random.seed(42)
        np.random.shuffle(data)
        
        total_size = len(data)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        
        splits = {
            'train': data[:train_size],
            'val': data[train_size:train_size + val_size],
            'test': data[train_size + val_size:]
        }
        
        for split_name, split_data in splits.items():
            output_file = os.path.join(output_dir, f"{split_name}.pkl")
            with open(output_file, 'wb') as f:
                pickle.dump(split_data, f)
            print(f"✅ Saved {len(split_data)} complexes to {output_file}")

def main():
    print("🔬 CrossDock Preprocessing Runner")
    print("=" * 50)
    
    # Check if already processed
    if all(Path(f"data/processed/{split}.pkl").exists() for split in ['train', 'val', 'test']):
        print("✅ Data already processed!")
        print("   Files found:")
        for split in ['train', 'val', 'test']:
            file_path = f"data/processed/{split}.pkl"
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            print(f"      {file_path}: {len(data)} samples")
        
        print("\n🚀 Ready for training!")
        print("   Run: python scripts/train_enhanced.py --quick_test --wandb")
        return
    
    # Run preprocessing
    success = analyze_and_preprocess()
    
    if success:
        print("\n🎉 Preprocessing completed successfully!")
        print("   Next steps:")
        print("   1. python setup_and_validate.py")
        print("   2. python scripts/train_enhanced.py --quick_test --wandb")
    else:
        print("\n❌ Preprocessing failed")
        print("   Check the errors above and debug")

if __name__ == "__main__":
    main()