# src/data/pocket_dataset.py
import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from Bio.PDB import PDBParser, DSSP
from typing import List, Dict, Any, Optional
import os

class ProteinPocketDataset(Dataset):
    """Dataset for protein pockets with binding site information"""
    
    def __init__(self, data_path: str, transform=None, pre_transform=None,
                 pocket_radius: float = 10.0, include_surface: bool = True):
        self.data_path = data_path
        self.pocket_radius = pocket_radius
        self.include_surface = include_surface
        
        super().__init__(None, transform, pre_transform)
        
        # Load pocket data
        self.pocket_data = self._load_pocket_data()
    
    def _load_pocket_data(self):
        """Load and process protein pocket data"""
        # Implementation for loading PDB files and extracting pocket information
        pocket_data = []
        
        for file in os.listdir(self.data_path):
            if file.endswith('.pdb'):
                pocket_info = self._process_pocket_file(os.path.join(self.data_path, file))
                if pocket_info is not None:
                    pocket_data.append(pocket_info)
        
        return pocket_data
    
    def _process_pocket_file(self, file_path: str) -> Optional[Dict]:
        """Process a single pocket PDB file"""
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('pocket', file_path)
            
            # Extract atomic features
            atoms = []
            positions = []
            residue_info = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            atoms.append(atom.element)
                            positions.append(atom.coord)
                            residue_info.append({
                                'residue_type': residue.resname,
                                'residue_id': residue.id[1],
                                'chain_id': chain.id
                            })
            
            if len(atoms) == 0:
                return None
            
            # Convert to features
            atom_features = self._atoms_to_features(atoms, residue_info)
            positions = np.array(positions)
            
            # Build connectivity graph
            edge_index, edge_features = self._build_pocket_graph(positions)
            
            return {
                'atom_features': atom_features,
                'positions': positions,
                'edge_index': edge_index,
                'edge_features': edge_features,
                'file_path': file_path
            }
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def _atoms_to_features(self, atoms: List[str], residue_info: List[Dict]) -> np.ndarray:
        """Convert atom and residue information to feature vectors"""
        features = []
        
        # Atom type mapping
        atom_types = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'H': 5}
        
        # Residue type mapping (20 standard amino acids)
        residue_types = {
            'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
            'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
            'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
            'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
        }
        
        for atom, res_info in zip(atoms, residue_info):
            # Atom features
            atom_type = atom_types.get(atom, 6)  # Unknown atom type
            
            # Residue features
            residue_type = residue_types.get(res_info['residue_type'], 20)  # Unknown residue
            
            # Combine features
            feature_vector = [
                atom_type,
                residue_type,
                res_info['residue_id'],
                ord(res_info['chain_id']) - ord('A')  # Chain ID as number
            ]
            
            features.append(feature_vector)
        
        return np.array(features, dtype=np.float32)
    
    def _build_pocket_graph(self, positions: np.ndarray) -> tuple:
        """Build graph connectivity for pocket atoms"""
        from scipy.spatial.distance import cdist
        
        # Compute pairwise distances
        distances = cdist(positions, positions)
        
        # Create edges based on distance threshold
        edge_index = []
        edge_features = []
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = distances[i, j]
                if dist < self.pocket_radius:
                    edge_index.extend([[i, j], [j, i]])
                    edge_features.extend([dist, dist])
        
        edge_index = np.array(edge_index).T if edge_index else np.zeros((2, 0))
        edge_features = np.array(edge_features).reshape(-1, 1) if edge_features else np.zeros((0, 1))
        
        return edge_index, edge_features
    
    def len(self):
        return len(self.pocket_data)
    
    def get(self, idx):
        pocket_info = self.pocket_data[idx]
        
        return Data(
            x=torch.tensor(pocket_info['atom_features'], dtype=torch.float),
            pos=torch.tensor(pocket_info['positions'], dtype=torch.float),
            edge_index=torch.tensor(pocket_info['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(pocket_info['edge_features'], dtype=torch.float),
            file_path=pocket_info['file_path']
        )