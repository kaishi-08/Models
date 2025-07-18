# src/data/data_loaders.py - Safe v# src/data/data_loaders.py - Safe version with memory limits
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.data import Data, Batch
from .molecular_dataset import CrossDockMolecularDataset
from .pocket_dataset import ProteinPocketDataset
from typing import Optional, Dict, Any

def safe_collate_crossdock_data(batch, max_pocket_atoms_per_mol: int = 500):
    """Safe collate function with memory limits"""
    
    if not batch or len(batch) == 0:
        return None
    
    # Filter out None entries
    batch = [data for data in batch if data is not None]
    if not batch:
        return None
    
    # Separate molecular and pocket data
    mol_data_list = []
    pocket_data_list = []
    
    for i, data in enumerate(batch):
        # Add molecular data
        mol_data_list.append(data)
        
        # Handle pocket data with size limits
        if hasattr(data, 'pocket_x') and data.pocket_x is not None:
            pocket_x = data.pocket_x
            pocket_pos = data.pocket_pos
            pocket_edge_index = getattr(data, 'pocket_edge_index', None)
            pocket_edge_attr = getattr(data, 'pocket_edge_attr', None)
            
            # LIMIT POCKET SIZE to prevent memory issues
            if pocket_x.size(0) > max_pocket_atoms_per_mol:
                print(f"Limiting pocket {i} from {pocket_x.size(0)} to {max_pocket_atoms_per_mol} atoms")
                indices = torch.randperm(pocket_x.size(0))[:max_pocket_atoms_per_mol]
                pocket_x = pocket_x[indices]
                pocket_pos = pocket_pos[indices]
                
                # Update edge indices if present
                if pocket_edge_index is not None:
                    # Filter edges that reference removed atoms
                    mask = (pocket_edge_index[0] < max_pocket_atoms_per_mol) & \
                           (pocket_edge_index[1] < max_pocket_atoms_per_mol)
                    pocket_edge_index = pocket_edge_index[:, mask]
                    if pocket_edge_attr is not None:
                        pocket_edge_attr = pocket_edge_attr[mask]
            
            # Create pocket data with correct batch index
            pocket_data = Data(
                x=pocket_x,
                pos=pocket_pos,
                edge_index=pocket_edge_index if pocket_edge_index is not None else torch.zeros((2, 0), dtype=torch.long),
                edge_attr=pocket_edge_attr if pocket_edge_attr is not None else torch.zeros((0, 1), dtype=torch.float),
                batch=torch.full((pocket_x.size(0),), i, dtype=torch.long)
            )
            pocket_data_list.append(pocket_data)
    
    try:
        # Batch molecular data
        mol_batch = Batch.from_data_list(mol_data_list)
        
        # Batch pocket data if available
        if pocket_data_list:
            pocket_batch = Batch.from_data_list(pocket_data_list)
            
            # Safely add pocket data to molecular batch
            mol_batch.pocket_x = pocket_batch.x
            mol_batch.pocket_pos = pocket_batch.pos
            mol_batch.pocket_edge_index = pocket_batch.edge_index
            mol_batch.pocket_edge_attr = pocket_batch.edge_attr
            mol_batch.pocket_batch = pocket_batch.batch
            
            # VALIDATE: Ensure pocket batch indices are valid
            max_mol_batch = mol_batch.batch.max().item()
            max_pocket_batch = mol_batch.pocket_batch.max().item()
            
            if max_pocket_batch > max_mol_batch:
                print(f"Fixing pocket batch indices: {max_pocket_batch} -> {max_mol_batch}")
                mol_batch.pocket_batch = torch.clamp(mol_batch.pocket_batch, 0, max_mol_batch)
        
        return mol_batch
        
    except Exception as e:
        print(f"Collation error: {e}")
        # Return simplified batch without pocket
        try:
            mol_batch = Batch.from_data_list(mol_data_list)
            return mol_batch
        except:
            return None

class CrossDockDataLoader:
    """Safe factory class for creating CrossDock data loaders"""
    
    @staticmethod
    def create_train_loader(config: Dict[str, Any]) -> DataLoader:
        """Create safe training data loader"""
        dataset = CrossDockMolecularDataset(
            data_path=config['data']['train_path'],
            include_pocket=config.get('include_pocket', True),
            max_atoms=config.get('max_atoms', 50),
            augment=config.get('augment', True)
        )
        
        return GeometricDataLoader(
            dataset,
            batch_size=min(config['data']['batch_size'], 16),  # Limit batch size
            shuffle=config['data'].get('shuffle', True),
            num_workers=min(config['data'].get('num_workers', 4), 2),  # Limit workers
            pin_memory=config['data'].get('pin_memory', True),
            collate_fn=safe_collate_crossdock_data,
            drop_last=True
        )
    
    @staticmethod
    def create_val_loader(config: Dict[str, Any]) -> DataLoader:
        """Create safe validation data loader"""
        val_path = config['data'].get('val_path', config['data']['test_path'])
        
        dataset = CrossDockMolecularDataset(
            data_path=val_path,
            include_pocket=config.get('include_pocket', True),
            max_atoms=config.get('max_atoms', 50),
            augment=False
        )
        
        return GeometricDataLoader(
            dataset,
            batch_size=min(config['data']['batch_size'], 8),  # Smaller batch for validation
            shuffle=False,
            num_workers=min(config['data'].get('num_workers', 4), 2),
            pin_memory=config['data'].get('pin_memory', True),
            collate_fn=safe_collate_crossdock_data,
            drop_last=False
        )
    
    @staticmethod
    def create_test_loader(config: Dict[str, Any]) -> DataLoader:
        """Create safe test data loader"""
        dataset = CrossDockMolecularDataset(
            data_path=config['data']['test_path'],
            include_pocket=config.get('include_pocket', True),
            max_atoms=config.get('max_atoms', 50),
            augment=False
        )
        
        return GeometricDataLoader(
            dataset,
            batch_size=min(config['data']['batch_size'], 8),
            shuffle=False,
            num_workers=min(config['data'].get('num_workers', 4), 2),
            pin_memory=config['data'].get('pin_memory', True),
            collate_fn=safe_collate_crossdock_data,
            drop_last=False
        )
    
    @staticmethod
    def create_train_val_split_loader(config: Dict[str, Any], val_ratio: float = 0.1):
        """Create safe train/val loaders by splitting train data"""
        from torch.utils.data import random_split
        
        full_dataset = CrossDockMolecularDataset(
            data_path=config['data']['train_path'],
            include_pocket=config.get('include_pocket', True),
            max_atoms=config.get('max_atoms', 50),
            augment=False
        )
        
        # Split dataset
        total_size = len(full_dataset)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create safe loaders
        train_loader = GeometricDataLoader(
            train_dataset,
            batch_size=min(config['data']['batch_size'], 16),
            shuffle=True,
            num_workers=min(config['data'].get('num_workers', 4), 2),
            pin_memory=config['data'].get('pin_memory', True),
            collate_fn=safe_collate_crossdock_data,
            drop_last=True
        )
        
        val_loader = GeometricDataLoader(
            val_dataset,
            batch_size=min(config['data']['batch_size'], 8),
            shuffle=False,
            num_workers=min(config['data'].get('num_workers', 4), 2),
            pin_memory=config['data'].get('pin_memory', True),
            collate_fn=safe_collate_crossdock_data,
            drop_last=False
        )
        
        return train_loader, val_loader

# Legacy classes for backward compatibility
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