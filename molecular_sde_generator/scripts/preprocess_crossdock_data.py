# scripts/preprocess_crossdock_data.py
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser, DSSP, NeighborSearch
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CrossDockDataProcessor:
    """Data processor for CrossDock dataset similar to Pocket2Mol"""
    
    def __init__(self, 
                 data_path: str,
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
        
        # Residue type mappings (20 standard amino acids)
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
        
    def process_dataset(self, output_dir: str = "data/processed/"):
        """Process the complete CrossDock dataset"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load CrossDock index file
        index_file = os.path.join(self.data_path, "crossdock2020-v1.1-other-pl/index.pkl")
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"CrossDock index file not found: {index_file}")
        
        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)
        
        processed_data = []
        failed_count = 0
        
        print(f"Processing {len(index_data)} protein-ligand complexes...")
        
        for i, item in enumerate(tqdm(index_data)):
            try:
                complex_data = self.process_complex(item)
                if complex_data is not None:
                    processed_data.append(complex_data)
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                continue
                
            # Save intermediate results every 1000 complexes
            if (i + 1) % 1000 == 0:
                temp_file = os.path.join(output_dir, f"temp_data_{i+1}.pkl")
                with open(temp_file, 'wb') as f:
                    pickle.dump(processed_data, f)
        
        print(f"Successfully processed: {len(processed_data)}")
        print(f"Failed: {failed_count}")
        
        # Split data
        self.split_and_save_data(processed_data, output_dir)
        
        return processed_data
    
    def process_complex(self, item: Dict) -> Optional[Dict]:
        """Process a single protein-ligand complex"""
        try:
            # Extract file paths
            protein_file = os.path.join(self.data_path, item['protein'])
            ligand_file = os.path.join(self.data_path, item['ligand'])
            
            if not os.path.exists(protein_file) or not os.path.exists(ligand_file):
                return None
            
            # Process ligand
            ligand_data = self.process_ligand(ligand_file)
            if ligand_data is None:
                return None
            
            # Process protein pocket
            pocket_data = self.process_pocket(protein_file, ligand_data['positions'])
            if pocket_data is None:
                return None
            
            # Combine data
            complex_data = {
                'ligand': ligand_data,
                'pocket': pocket_data,
                'complex_id': item.get('complex_id', ''),
                'protein_file': protein_file,
                'ligand_file': ligand_file
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
            pocket_residues = []
            
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

def main():
    """Main processing function"""
    # Set data paths
    data_path = "data/raw/crossdock2020"  # Adjust path as needed
    output_dir = "data/processed/"
    
    # Initialize processor
    processor = CrossDockDataProcessor(
        data_path=data_path,
        pocket_radius=10.0,
        min_atoms=5,
        max_atoms=50,
        max_pocket_atoms=500
    )
    
    # Process dataset
    processed_data = processor.process_dataset(output_dir)
    
    print(f"Data processing completed! Total complexes: {len(processed_data)}")

if __name__ == "__main__":
    main()