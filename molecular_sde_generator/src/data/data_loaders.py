# src/data/data_loaders.py
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader
from .molecular_dataset import MolecularDataset, collate_molecular_data
from .pocket_dataset import ProteinPocketDataset
from typing import Optional, Dict, Any

class MolecularDataLoader:
    """Factory class for creating molecular data loaders"""
    
    @staticmethod
    def create_train_loader(config: Dict[str, Any]) -> DataLoader:
        """Create training data loader"""
        dataset = MolecularDataset(
            data_path=config['data']['train_path'],
            include_pocket=config.get('include_pocket', True),
            max_atoms=config.get('max_atoms', 50)
        )
        
        return GeometricDataLoader(
            dataset,
            batch_size=config['data']['batch_size'],
            shuffle=config['data'].get('shuffle', True),
            num_workers=config['data'].get('num_workers', 4),
            pin_memory=config['data'].get('pin_memory', True),
            collate_fn=collate_molecular_data
        )
    
    @staticmethod
    def create_val_loader(config: Dict[str, Any]) -> DataLoader:
        """Create validation data loader"""
        dataset = MolecularDataset(
            data_path=config['data']['val_path'],
            include_pocket=config.get('include_pocket', True),
            max_atoms=config.get('max_atoms', 50)
        )
        
        return GeometricDataLoader(
            dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data'].get('num_workers', 4),
            pin_memory=config['data'].get('pin_memory', True),
            collate_fn=collate_molecular_data
        )
    
    @staticmethod
    def create_test_loader(config: Dict[str, Any]) -> DataLoader:
        """Create test data loader"""
        dataset = MolecularDataset(
            data_path=config['data']['test_path'],
            include_pocket=config.get('include_pocket', True),
            max_atoms=config.get('max_atoms', 50)
        )
        
        return GeometricDataLoader(
            dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data'].get('num_workers', 4),
            pin_memory=config['data'].get('pin_memory', True),
            collate_fn=collate_molecular_data
        )

class PocketDataLoader:
    """Factory class for creating protein pocket data loaders"""
    
    @staticmethod
    def create_loader(data_path: str, config: Dict[str, Any]) -> DataLoader:
        """Create pocket data loader"""
        dataset = ProteinPocketDataset(
            data_path=data_path,
            pocket_radius=config.get('pocket_radius', 10.0),
            include_surface=config.get('include_surface', True)
        )
        
        return GeometricDataLoader(
            dataset,
            batch_size=config.get('batch_size', 16),
            shuffle=config.get('shuffle', False),
            num_workers=config.get('num_workers', 2)
        )