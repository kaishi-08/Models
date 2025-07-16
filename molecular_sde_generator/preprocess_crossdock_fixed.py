# preprocess_crossdock_fixed.py - Fixed preprocessor cho CrossDock tuple format

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CrossDockTupleProcessor:
    """Processor cho CrossDock tuple format"""
    
    def __init__(self, data_path: str, 
                 pocket_radius: float = 10.0,
                 min_atoms: int = 5,
                 max_atoms: int = 50,
                 max_pocket_atoms: int = 500):
        self.data_path = data_path
        self.pocket_radius = pocket_radius
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.max_pocket_atoms = max_pocket_atoms
        
        # Atom type mappings
        self.atom_types = {
            'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5, 
            'Cl': 6, 'Br': 7, 'I': 8, 'H': 9, 'Unknown': 10
        }
        
        # Residue type mappings
        self.residue_types = {
            'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
            'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
            'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
            'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
            'Unknown': 20
        }
        
        # Bond type mappings
        self.bond_types = {
            Chem.BondType.SINGLE: 0,
            Chem.BondType.DOUBLE: 1,
            Chem.BondType.TRIPLE: 2,
            Chem.BondType.AROMATIC: 3
        }
    
    def analyze_index_format(self):
        """Analyze format cua index file"""
        print("Analyzing index format...")
        
        index_file = os.path.join(self.data_path, "index.pkl")
        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)
        
        print(f"Total entries: {len(index_data)}")
        
        # Check first few entries
        for i in range(min(3, len(index_data))):
            entry = index_data[i]
            print(f"Entry {i}: {type(entry)}")
            if isinstance(entry, (list, tuple)):
                print(f"  Length: {len(entry)}")
                for j, item in enumerate(entry):
                    print(f"    [{j}]: {type(item)} - {str(item)[:50]}")
        
        return index_data
    
    def process_dataset(self, output_dir: str = "data/processed/"):
        """Process the complete dataset"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and analyze index
        index_data = self.analyze_index_format()
        
        processed_data = []
        failed_count = 0
        
        print(f"Processing {len(index_data)} protein-ligand complexes...")
        
        # Process in batches
        batch_size = 1000
        for batch_start in range(0, len(index_data), batch_size):
            batch_end = min(batch_start + batch_size, len(index_data))
            batch_data = index_data[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1}: entries {batch_start}-{batch_end}")
            
            for i, entry in enumerate(tqdm(batch_data, desc=f"Batch {batch_start//batch_size + 1}")):
                try:
                    complex_data = self.process_tuple_entry(entry)
                    if complex_data is not None:
                        processed_data.append(complex_data)
                    else:
                        failed_count += 1
                except Exception as e:
                    failed_count += 1
                    if i < 5:  # Show first few errors for debugging
                        print(f"Entry {batch_start + i} failed: {e}")
                    continue
            
            # Save intermediate results
            if len(processed_data) > 0:
                temp_file = os.path.join(output_dir, f"temp_batch_{batch_start//batch_size + 1}.pkl")
                with open(temp_file, 'wb') as f:
                    pickle.dump(processed_data[-len(batch_data):], f)
        
        print(f"Successfully processed: {len(processed_data)}")
        print(f"Failed: {failed_count}")
        print(f"Success rate: {len(processed_data)/(len(processed_data)+failed_count)*100:.1f}%")
        
        # Split and save data
        self.split_and_save_data(processed_data, output_dir)
        
        return processed_data
    
    def process_tuple_entry(self, entry) -> Optional[Dict]:
        """Process a single tuple entry"""
        try:
            protein_path = None
            ligand_path = None
            
            # Extract paths from tuple/list
            if isinstance(entry, (list, tuple)):
                for item in entry:
                    if isinstance(item, str):
                        # Check if it's a file path
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
            
            # Process protein pocket
            pocket_data = self.process_pocket(protein_path, ligand_data['positions'])
            if pocket_data is None:
                return None
            
            # Combine data
            complex_data = {
                'ligand': ligand_data,
                'pocket': pocket_data,
                'complex_id': f"{os.path.basename(protein_path)}_{os.path.basename(ligand_path)}",
                'protein_file': protein_path,
                'ligand_file': ligand_path
            }
            
            return complex_data
            
        except Exception as e:
            return None
    
    def process_ligand(self, ligand_file: str) -> Optional[Dict]:
        """Process ligand molecule"""
        try:
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
            atom_features = self.get_ligand_atom_features(mol)
            positions = self.get_ligand_positions(mol)
            edge_index, edge_features = self.get_ligand_bonds(mol)
            
            # Get SMILES
            smiles = Chem.MolToSmiles(mol)
            
            return {
                'atom_features': atom_features,
                'positions': positions,
                'edge_index': edge_index,
                'edge_features': edge_features,
                'smiles': smiles,
                'num_atoms': num_atoms
            }
            
        except Exception as e:
            return None
    
    def process_pocket(self, protein_file: str, ligand_positions: np.ndarray) -> Optional[Dict]:
        """Process protein pocket around ligand"""
        try:
            # Parse protein structure
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', protein_file)
            
            # Extract pocket atoms within radius
            pocket_atoms = []
            
            # Calculate ligand center
            ligand_center = np.mean(ligand_positions, axis=0)
            
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
            atom_features = self.get_pocket_atom_features(pocket_atoms)
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
    
    def get_ligand_atom_features(self, mol: Chem.Mol) -> np.ndarray:
        """Extract ligand atom features"""
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
    
    def get_ligand_positions(self, mol: Chem.Mol) -> np.ndarray:
        """Extract ligand 3D coordinates"""
        conf = mol.GetConformer()
        positions = []
        
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            positions.append([pos.x, pos.y, pos.z])
        
        return np.array(positions, dtype=np.float32)
    
    def get_ligand_bonds(self, mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray]:
        """Extract ligand bond information"""
        edge_index = []
        edge_features = []
        
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = self.bond_types.get(bond.GetBondType(), 0)
            
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
    
    def get_pocket_atom_features(self, pocket_atoms: List[Dict]) -> np.ndarray:
        """Extract pocket atom features"""
        features = []
        
        for atom in pocket_atoms:
            feature_vector = [
                self.atom_types.get(atom['element'], self.atom_types['Unknown']),
                self.residue_types.get(atom['residue'], self.residue_types['Unknown']),
                float(atom['is_backbone']),
                atom['residue_id'] % 100,  # Normalize residue ID
                0.0,  # Placeholder for surface accessibility
                0.0,  # Placeholder for phi angle
                0.0,  # Placeholder for psi angle
                0.0   # Placeholder for charge
            ]
            features.append(feature_vector)
        
        return np.array(features, dtype=np.float32)
    
    def get_pocket_bonds(self, positions: np.ndarray, cutoff: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        """Create pocket connectivity based on distance"""
        if len(positions) == 0:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 1), dtype=np.float32)
        
        # Compute pairwise distances
        from scipy.spatial.distance import cdist
        distances = cdist(positions, positions)
        
        # Create edges based on distance cutoff
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
    
    def split_and_save_data(self, data: List[Dict], output_dir: str):
        """Split data into train/val/test and save"""
        np.random.seed(42)
        np.random.shuffle(data)
        
        total_size = len(data)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        # Save splits
        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        for split_name, split_data in splits.items():
            output_file = os.path.join(output_dir, f"{split_name}.pkl")
            with open(output_file, 'wb') as f:
                pickle.dump(split_data, f)
            print(f"Saved {len(split_data)} complexes to {output_file}")
        
        # Clean up temp files
        import glob
        temp_files = glob.glob(os.path.join(output_dir, "temp_*.pkl"))
        for temp_file in temp_files:
            os.remove(temp_file)
        print(f"Cleaned up {len(temp_files)} temporary files")

def test_sample_processing():
    """Test processing on a small sample"""
    print("Testing sample processing...")
    
    processor = CrossDockTupleProcessor(
        data_path="data/crossdocked_pocket10",
        pocket_radius=10.0,
        min_atoms=5,
        max_atoms=50,
        max_pocket_atoms=500
    )
    
    # Load a few samples
    index_file = os.path.join(processor.data_path, "index.pkl")
    with open(index_file, 'rb') as f:
        index_data = pickle.load(f)
    
    print(f"Testing on first 10 entries out of {len(index_data)}...")
    
    success_count = 0
    for i, entry in enumerate(index_data[:10]):
        try:
            result = processor.process_tuple_entry(entry)
            if result is not None:
                success_count += 1
                print(f"Entry {i}: SUCCESS")
                print(f"  Ligand atoms: {result['ligand']['num_atoms']}")
                print(f"  Pocket atoms: {result['pocket']['num_atoms']}")
            else:
                print(f"Entry {i}: FAILED (returned None)")
        except Exception as e:
            print(f"Entry {i}: ERROR - {e}")
    
    print(f"Test results: {success_count}/10 successful")
    return success_count > 0

def main():
    """Main processing function"""
    print("CrossDock Tuple Format Preprocessor")
    print("=" * 50)
    
    # Test on small sample first
    if not test_sample_processing():
        print("❌ Sample testing failed. Check data format and paths.")
        return
    
    print("✅ Sample testing passed. Proceeding with full processing...")
    
    # Initialize processor
    processor = CrossDockTupleProcessor(
        data_path="data/crossdocked_pocket10",
        pocket_radius=10.0,
        min_atoms=5,
        max_atoms=50,
        max_pocket_atoms=500
    )
    
    # Process dataset
    try:
        processed_data = processor.process_dataset("data/processed/")
        print(f"Data processing completed! Total complexes: {len(processed_data)}")
        
        print("\nNext steps:")
        print("1. python setup_and_validate.py")
        print("2. python scripts/train_enhanced.py --quick_test --wandb")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()