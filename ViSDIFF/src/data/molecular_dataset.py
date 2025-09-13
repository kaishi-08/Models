# src/data/molecular_dataset.py
import torch
import pickle
import numpy as np
from torch_geometric.data import Dataset, Data
from typing import List, Dict, Any, Optional
import os

class CrossDockMolecularDataset(Dataset):    
    def __init__(self, data_path: str, transform=None, pre_transform=None, 
                 include_pocket: bool = True, max_atoms: int = 50,
                 augment: bool = False):
        self.data_path = data_path
        self.include_pocket = include_pocket
        self.max_atoms = max_atoms
        self.augment = augment
        
        super().__init__(None, transform, pre_transform)
        
        # Load processed data
        with open(data_path, 'rb') as f:
            self.data_list = pickle.load(f)
        
        print(f"Loaded {len(self.data_list)} complexes from {data_path}")
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        complex_data = self.data_list[idx]
        
        # Extract ligand data
        ligand = complex_data['ligand']
        
        # Create molecular graph data
        mol_data = Data(
            x=torch.tensor(ligand['atom_features'], dtype=torch.float),
            pos=torch.tensor(ligand['positions'], dtype=torch.float),
            edge_index=torch.tensor(ligand['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(ligand['edge_features'], dtype=torch.float),
            smiles=ligand['smiles'],
            mol_id=idx
        )
        
        # Add pocket information if available and requested
        if self.include_pocket and 'pocket' in complex_data:
            pocket = complex_data['pocket']
            mol_data.pocket_x = torch.tensor(pocket['atom_features'], dtype=torch.float)
            mol_data.pocket_pos = torch.tensor(pocket['positions'], dtype=torch.float)
            mol_data.pocket_edge_index = torch.tensor(pocket['edge_index'], dtype=torch.long)
            mol_data.pocket_edge_attr = torch.tensor(pocket['edge_features'], dtype=torch.float)
        
        # Apply augmentation if requested
        if self.augment:
            mol_data = self._apply_augmentation(mol_data)
        
        return mol_data
    
    def _apply_augmentation(self, data: Data) -> Data:
        """Apply data augmentation"""
        # Random rotation for ligand
        if hasattr(data, 'pos'):
            rotation_matrix = self._random_rotation_matrix()
            data.pos = torch.matmul(data.pos, rotation_matrix.T)
        
        # Random rotation for pocket
        if hasattr(data, 'pocket_pos'):
            data.pocket_pos = torch.matmul(data.pocket_pos, rotation_matrix.T)
        
        # Add small noise to positions
        noise_scale = 0.1
        if hasattr(data, 'pos'):
            noise = torch.randn_like(data.pos) * noise_scale
            data.pos = data.pos + noise
        
        if hasattr(data, 'pocket_pos'):
            pocket_noise = torch.randn_like(data.pocket_pos) * (noise_scale * 0.5)
            data.pocket_pos = data.pocket_pos + pocket_noise
        
        return data
    
    def _random_rotation_matrix(self) -> torch.Tensor:
        """Generate random 3D rotation matrix"""
        # Generate random quaternion
        q = torch.randn(4)
        q = q / torch.norm(q)
        
        # Convert to rotation matrix
        w, x, y, z = q
        rotation_matrix = torch.tensor([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ], dtype=torch.float)
        
        return rotation_matrix

def collate_crossdock_data(batch):
    """Custom collate function for CrossDock molecular data with pockets"""
    from torch_geometric.data import Batch
    
    # Handle empty batch
    if not batch:
        return None
    
    # Separate molecular and pocket data
    mol_data_list = []
    pocket_data_list = []
    
    for i, data in enumerate(batch):
        # Add batch index to molecular data
        mol_data_list.append(data)
        
        # Handle pocket data if present
        if hasattr(data, 'pocket_x'):
            pocket_data = Data(
                x=data.pocket_x,
                pos=data.pocket_pos,
                edge_index=data.pocket_edge_index,
                edge_attr=data.pocket_edge_attr,
                batch=torch.full((data.pocket_x.size(0),), i, dtype=torch.long)
            )
            pocket_data_list.append(pocket_data)
    
    # Batch molecular data
    mol_batch = Batch.from_data_list(mol_data_list)
    
    # Batch pocket data if available
    if pocket_data_list:
        pocket_batch = Batch.from_data_list(pocket_data_list)
        
        # Add pocket data to molecular batch
        mol_batch.pocket_x = pocket_batch.x
        mol_batch.pocket_pos = pocket_batch.pos
        mol_batch.pocket_edge_index = pocket_batch.edge_index
        mol_batch.pocket_edge_attr = pocket_batch.edge_attr
        mol_batch.pocket_batch = pocket_batch.batch
    
    return mol_batch
