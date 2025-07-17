import os
import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings('ignore')

class CrossDockPreprocessorTrainTestOnly:
    """Preprocessor cho CrossDock dataset - chỉ train và test như gốc"""
    
    def __init__(self, data_dir="data/crossdocked_pocket10", output_dir="data/processed", 
                 max_samples=None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        
        # Atom type mapping
        self.atom_types = {
            'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5, 
            'Cl': 6, 'Br': 7, 'I': 8, 'H': 9, 'UNK': 10
        }
        
        # Bond type mapping
        self.bond_types = {
            Chem.BondType.SINGLE: 0,
            Chem.BondType.DOUBLE: 1, 
            Chem.BondType.TRIPLE: 2,
            Chem.BondType.AROMATIC: 3
        }
        
        # Amino acid mapping
        self.amino_acids = {
            'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
            'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
            'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
            'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
            'UNK': 20
        }
    
    def load_splits(self):
        """Load train/test splits - CHỈ 2 splits như CrossDock gốc"""
        split_file = Path("data/split_by_name.pt")
        
        if split_file.exists():
            print("📂 Loading existing splits...")
            try:
                splits = torch.load(split_file)
                
                # Kiểm tra xem có đúng format train/test không
                if 'train' in splits and 'test' in splits:
                    print(f"✅ Found train/test splits: train={len(splits['train'])}, test={len(splits['test'])}")
                    
                    # Nếu có val thì bỏ qua, chỉ dùng train/test
                    if 'val' in splits:
                        print("⚠️  Found validation set but ignoring (using train/test only)")
                        splits = {'train': splits['train'], 'test': splits['test']}
                    
                    return splits
                else:
                    print("⚠️  Invalid split format, creating new train/test splits...")
            except Exception as e:
                print(f"⚠️  Error loading splits: {e}, creating new splits...")
        
        print("🔧 Creating train/test splits from index.pkl...")
        return self.create_train_test_splits()
    
    def create_train_test_splits(self):
        """Tạo splits train/test từ index.pkl (80/20 split)"""
        # Load index
        index_file = self.data_dir / "index.pkl"
        if not index_file.exists():
            print(f"❌ index.pkl not found at {index_file}")
            return None
        
        try:
            with open(index_file, 'rb') as f:
                index_data = pickle.load(f)
            
            print(f"📊 Found {len(index_data)} entries in index")
            
            # Tạo train/test splits (80/20)
            np.random.seed(42)
            indices = list(range(len(index_data)))
            np.random.shuffle(indices)
            
            n_total = len(indices)
            n_train = int(0.8 * n_total)  # 80% cho train
            
            splits = {
                'train': [index_data[i] for i in indices[:n_train]],
                'test': [index_data[i] for i in indices[n_train:]]
            }
            
            print(f"✅ Created train/test splits: train={len(splits['train'])}, test={len(splits['test'])}")
            
            # Lưu splits
            split_file = Path("data/split_by_name.pt")
            torch.save(splits, split_file)
            print(f"💾 Train/test splits saved to {split_file}")
            
            return splits
            
        except Exception as e:
            print(f"❌ Error creating splits: {e}")
            return None
    
    def process_complex(self, entry):
        """Process a single protein-ligand complex - handles both 2 and 4 element formats"""
        
        # Handle different entry formats
        if len(entry) == 2:
            # Format: (pocket_file, ligand_file)
            pocket_file, ligand_file = entry
            receptor_file = None
            score = 0.0  # Default score
        elif len(entry) >= 4:
            # Format: (pocket_file, ligand_file, receptor_file, score)
            pocket_file, ligand_file, receptor_file, score = entry[:4]
        else:
            return None
        
        # Full paths - files are in subdirectories
        ligand_path = self.data_dir / ligand_file
        pocket_path = self.data_dir / pocket_file
        
        # Check if files exist
        if not ligand_path.exists():
            # Try to find the file in case path is wrong
            parent_dir = ligand_path.parent
            if parent_dir.exists():
                ligand_files = list(parent_dir.glob("*.sdf"))
                if ligand_files:
                    ligand_path = ligand_files[0]  # Use first .sdf file found
                else:
                    return None
            else:
                return None
        
        if not pocket_path.exists():
            # Try to find the pocket file
            parent_dir = pocket_path.parent
            if parent_dir.exists():
                pocket_files = list(parent_dir.glob("*pocket*.pdb"))
                if pocket_files:
                    pocket_path = pocket_files[0]  # Use first pocket file found
                else:
                    pocket_path = None
            else:
                pocket_path = None
        
        # Process ligand
        try:
            # Try different ways to load ligand
            if ligand_path.suffix.lower() == '.sdf':
                supplier = Chem.SDMolSupplier(str(ligand_path))
                mol = None
                for m in supplier:
                    if m is not None:
                        mol = m
                        break
                if mol is None:
                    return None
            elif ligand_path.suffix.lower() in ['.mol', '.mol2']:
                mol = Chem.MolFromMolFile(str(ligand_path))
            else:
                return None
                
            ligand_data = self.mol_to_features(mol)
            if ligand_data is None:
                return None
        except Exception as e:
            return None
        
        # Process pocket (optional)
        pocket_data = None
        if pocket_path and pocket_path.exists():
            pocket_data = self.pdb_to_features(pocket_path)
        
        complex_data = {
            'pocket_file': str(pocket_file),
            'ligand_file': str(ligand_file),
            'receptor_file': str(receptor_file) if receptor_file else None,
            'score': float(score),
            'ligand': ligand_data
        }
        
        if pocket_data:
            complex_data['pocket'] = pocket_data
        
        return complex_data
    
    def mol_to_features(self, mol):
        """Convert RDKit molecule to graph features"""
        if mol is None:
            return None
        
        # Atom features
        atom_features = []
        positions = []
        
        # Get conformer for 3D coordinates
        if mol.GetNumConformers() == 0:
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                return None
        
        conf = mol.GetConformer()
        
        for atom in mol.GetAtoms():
            # Atom type
            atom_type = self.atom_types.get(atom.GetSymbol(), self.atom_types['UNK'])
            
            # Additional features
            features = [
                atom_type,
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetTotalNumHs(),
                int(atom.IsInRing()),
                atom.GetMass() / 100.0  # Normalize
            ]
            atom_features.append(features)
            
            # 3D position
            pos = conf.GetAtomPosition(atom.GetIdx())
            positions.append([pos.x, pos.y, pos.z])
        
        # Bond features
        edge_index = []
        edge_features = []
        
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = self.bond_types.get(bond.GetBondType(), 0)
            
            # Add both directions
            edge_index.extend([[start, end], [end, start]])
            edge_features.extend([[bond_type], [bond_type]])
        
        return {
            'atom_features': np.array(atom_features, dtype=np.float32),
            'positions': np.array(positions, dtype=np.float32),
            'edge_index': np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64),
            'edge_features': np.array(edge_features, dtype=np.float32) if edge_features else np.zeros((0, 1), dtype=np.float32),
            'smiles': Chem.MolToSmiles(mol)
        }
    
    def pdb_to_features(self, pdb_file):
        """Convert PDB file to protein pocket features"""
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('pocket', pdb_file)
            
            atom_features = []
            positions = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        res_name = residue.get_resname()
                        res_id = residue.get_id()[1]
                        
                        for atom in residue:
                            # Basic atom features
                            element = atom.element.strip() if atom.element else 'C'
                            atom_type = self.atom_types.get(element, self.atom_types['UNK'])
                            res_type = self.amino_acids.get(res_name, self.amino_acids['UNK'])
                            
                            features = [
                                atom_type,
                                res_type, 
                                res_id % 1000,  # Normalize residue ID
                                int(atom.name.startswith('C')),  # Is carbon backbone
                                int(atom.name.startswith('N')),  # Is nitrogen backbone
                                int(atom.name.startswith('O')),  # Is oxygen backbone
                                0.0,  # Placeholder for surface accessibility
                                0.0   # Placeholder for charge
                            ]
                            
                            atom_features.append(features)
                            
                            # 3D coordinates
                            coord = atom.coord
                            positions.append([coord[0], coord[1], coord[2]])
            
            if len(atom_features) == 0:
                return None
            
            # Simple distance-based connectivity
            positions_array = np.array(positions)
            
            # Limit connectivity computation to avoid memory issues
            max_atoms = min(len(positions), 1000)  # Limit to 1000 atoms
            distances = np.linalg.norm(positions_array[:max_atoms, None] - positions_array[None, :max_atoms], axis=2)
            
            edge_index = []
            edge_features = []
            
            # Connect atoms within 5Å
            for i in range(max_atoms):
                for j in range(i + 1, max_atoms):
                    if distances[i, j] < 5.0:
                        edge_index.extend([[i, j], [j, i]])
                        edge_features.extend([[distances[i, j]], [distances[i, j]]])
            
            return {
                'atom_features': np.array(atom_features, dtype=np.float32),
                'positions': np.array(positions, dtype=np.float32),
                'edge_index': np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64),
                'edge_features': np.array(edge_features, dtype=np.float32) if edge_features else np.zeros((0, 1), dtype=np.float32)
            }
            
        except Exception as e:
            return None
    
    def process_dataset(self):
        """Process entire dataset - CHỈ train và test"""
        print("🔄 Processing CrossDock dataset (train/test only)...")
        
        # Load splits
        splits = self.load_splits()
        if splits is None:
            print("❌ Failed to load or create splits!")
            return
        
        # CHỈ xử lý train và test
        for split_name in ['train', 'test']:
            if split_name not in splits:
                print(f"❌ Missing {split_name} split!")
                continue
                
            entries = splits[split_name]
            print(f"\n📊 Processing {split_name} split ({len(entries)} entries)...")
            
            processed_data = []
            failed_count = 0
            
            # Determine max samples for this split
            if self.max_samples:
                if split_name == 'train':
                    max_for_split = min(self.max_samples, len(entries))
                else:  # test
                    max_for_split = min(self.max_samples // 4, len(entries))  # Test = 1/4 của train
                print(f"⚠️  Limited to {max_for_split} samples for testing")
            else:
                max_for_split = len(entries)
            
            for i, entry in enumerate(tqdm(entries, desc=f"Processing {split_name}")):
                if i >= max_for_split:
                    break
                    
                complex_data = self.process_complex(entry)
                if complex_data:
                    processed_data.append(complex_data)
                else:
                    failed_count += 1
            
            print(f"✅ Processed {len(processed_data)} complexes, failed: {failed_count}")
            
            # Save processed data
            if processed_data:
                output_file = self.output_dir / f"{split_name}.pkl"
                with open(output_file, 'wb') as f:
                    pickle.dump(processed_data, f)
                print(f"💾 Saved to {output_file}")
            else:
                print(f"❌ No valid data for {split_name}")
        
        print("\n🎉 Dataset preprocessing completed! (train/test only)")

def main():
    parser = argparse.ArgumentParser(description='Preprocess CrossDock dataset (Train/Test only)')
    parser.add_argument('--data_dir', type=str, default='data/crossdocked_pocket10',
                       help='Path to crossdocked_pocket10 data')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples per split (for testing)')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not Path(args.data_dir).exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        print("Please make sure crossdocked_pocket10 is in the correct location")
        return
    
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🎯 Using train/test splits only (no validation)")
    if args.max_samples:
        print(f"⚠️  Max samples: {args.max_samples}")
    
    preprocessor = CrossDockPreprocessorTrainTestOnly(
        data_dir=args.data_dir, 
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )
    
    preprocessor.process_dataset()

if __name__ == "__main__":
    main()