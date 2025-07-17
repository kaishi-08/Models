# src/data/data_loaders.py - Fixed for train/test only
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader
from .molecular_dataset import CrossDockMolecularDataset, collate_crossdock_data
from .pocket_dataset import ProteinPocketDataset
from typing import Optional, Dict, Any

class CrossDockDataLoader:
    """Factory class for creating CrossDock data loaders - Train/Test only"""
    
    @staticmethod
    def create_train_loader(config: Dict[str, Any]) -> DataLoader:
        """Create training data loader for CrossDock"""
        dataset = CrossDockMolecularDataset(
            data_path=config['data']['train_path'],
            include_pocket=config.get('include_pocket', True),
            max_atoms=config.get('max_atoms', 50),
            augment=config.get('augment', True)
        )
        
        return GeometricDataLoader(
            dataset,
            batch_size=config['data']['batch_size'],
            shuffle=config['data'].get('shuffle', True),
            num_workers=config['data'].get('num_workers', 4),
            pin_memory=config['data'].get('pin_memory', True),
            collate_fn=collate_crossdock_data,
            drop_last=True  # Important for consistent batch sizes
        )
    
    @staticmethod
    def create_val_loader(config: Dict[str, Any]) -> DataLoader:
        """Create validation data loader - uses val_path from config"""
        # Check if val_path exists, otherwise use test_path
        val_path = config['data'].get('val_path', config['data']['test_path'])
        
        dataset = CrossDockMolecularDataset(
            data_path=val_path,
            include_pocket=config.get('include_pocket', True),
            max_atoms=config.get('max_atoms', 50),
            augment=False  # No augmentation for validation
        )
        
        return GeometricDataLoader(
            dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data'].get('num_workers', 4),
            pin_memory=config['data'].get('pin_memory', True),
            collate_fn=collate_crossdock_data,
            drop_last=False
        )
    
    @staticmethod
    def create_test_loader(config: Dict[str, Any]) -> DataLoader:
        """Create test data loader for CrossDock"""
        dataset = CrossDockMolecularDataset(
            data_path=config['data']['test_path'],
            include_pocket=config.get('include_pocket', True),
            max_atoms=config.get('max_atoms', 50),
            augment=False  # No augmentation for testing
        )
        
        return GeometricDataLoader(
            dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data'].get('num_workers', 4),
            pin_memory=config['data'].get('pin_memory', True),
            collate_fn=collate_crossdock_data,
            drop_last=False
        )

    @staticmethod
    def create_train_val_split_loader(config: Dict[str, Any], val_ratio: float = 0.1):
        """Create train/val loaders by splitting train data"""
        from torch.utils.data import random_split
        
        full_dataset = CrossDockMolecularDataset(
            data_path=config['data']['train_path'],
            include_pocket=config.get('include_pocket', True),
            max_atoms=config.get('max_atoms', 50),
            augment=False  # No augmentation for splitting
        )
        
        # Split dataset
        total_size = len(full_dataset)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create train loader with augmentation
        train_loader = GeometricDataLoader(
            train_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=config['data'].get('num_workers', 4),
            pin_memory=config['data'].get('pin_memory', True),
            collate_fn=collate_crossdock_data,
            drop_last=True
        )
        
        # Create val loader without augmentation
        val_loader = GeometricDataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data'].get('num_workers', 4),
            pin_memory=config['data'].get('pin_memory', True),
            collate_fn=collate_crossdock_data,
            drop_last=False
        )
        
        return train_loader, val_loader

# Legacy class for backward compatibility
class MolecularDataLoader(CrossDockDataLoader):
    """Factory class for creating molecular data loaders (backward compatibility)"""
    pass

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