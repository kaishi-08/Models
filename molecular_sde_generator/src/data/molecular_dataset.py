# src/data/molecular_dataset.py
import torch
import pickle
import numpy as np
from torch_geometric.data import Dataset, Data
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Dict, Any, Optional
import os

class MolecularDataset(Dataset):
    """Dataset for molecular structures with optional protein pocket conditioning"""
    
    def __init__(self, data_path: str, transform=None, pre_transform=None, 
                 include_pocket: bool = True, max_atoms: int = 50):
        self.data_path = data_path
        self.include_pocket = include_pocket
        self.max_atoms = max_atoms
        
        super().__init__(None, transform, pre_transform)
        
        # Load processed data
        with open(data_path, 'rb') as f:
            self.data_list = pickle.load(f)
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        data_dict = self.data_list[idx]
        
        # Create molecular graph
        mol_data = Data(
            x=torch.tensor(data_dict['atom_features'], dtype=torch.float),
            pos=torch.tensor(data_dict['positions'], dtype=torch.float),
            edge_index=torch.tensor(data_dict['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(data_dict['edge_features'], dtype=torch.float),
            y=torch.tensor(data_dict.get('target', 0), dtype=torch.float)
        )
        
        # Add pocket information if available
        if self.include_pocket and 'pocket' in data_dict:
            pocket_data = data_dict['pocket']
            mol_data.pocket_x = torch.tensor(pocket_data['atom_features'], dtype=torch.float)
            mol_data.pocket_pos = torch.tensor(pocket_data['positions'], dtype=torch.float)
            mol_data.pocket_edge_index = torch.tensor(pocket_data['edge_index'], dtype=torch.long)
            mol_data.pocket_edge_attr = torch.tensor(pocket_data.get('edge_features', []), dtype=torch.float)
        
        # Add metadata
        mol_data.smiles = data_dict.get('smiles', '')
        mol_data.mol_id = data_dict.get('mol_id', idx)
        
        return mol_data

def collate_molecular_data(batch):
    """Custom collate function for molecular data with pockets"""
    from torch_geometric.data import Batch
    
    # Separate molecular and pocket data
    mol_batch = Batch.from_data_list([data for data in batch])
    
    # Handle pocket data separately if present
    if hasattr(batch[0], 'pocket_x'):
        pocket_data_list = []
        for i, data in enumerate(batch):
            pocket_data = Data(
                x=data.pocket_x,
                pos=data.pocket_pos,
                edge_index=data.pocket_edge_index,
                edge_attr=data.pocket_edge_attr if hasattr(data, 'pocket_edge_attr') else None,
                batch=torch.full((data.pocket_x.size(0),), i, dtype=torch.long)
            )
            pocket_data_list.append(pocket_data)
        
        pocket_batch = Batch.from_data_list(pocket_data_list)
        
        # Add pocket data to molecular batch
        mol_batch.pocket_x = pocket_batch.x
        mol_batch.pocket_pos = pocket_batch.pos
        mol_batch.pocket_edge_index = pocket_batch.edge_index
        mol_batch.pocket_edge_attr = pocket_batch.edge_attr
        mol_batch.pocket_batch = pocket_batch.batch
    
    return mol_batch