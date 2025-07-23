# src/data/data_loaders.py - Complete fix with pocket_batch
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.data import Data, Batch
from .molecular_dataset import CrossDockMolecularDataset
from .pocket_dataset import ProteinPocketDataset
from typing import Optional, Dict, Any

def safe_collate_crossdock_data(batch, max_pocket_atoms_per_mol: int = 500):
    """
    Comprehensive collate function with proper pocket_batch creation
    
    This function fixes the missing pocket_batch issue that was causing training errors
    """
    
    if not batch or len(batch) == 0:
        return None
    
    # Filter out None entries
    valid_batch = [data for data in batch if data is not None]
    if not valid_batch:
        return None
    
    # Separate molecular and pocket data
    mol_data_list = []
    pocket_data_list = []
    
    for mol_idx, data in enumerate(valid_batch):
        # Add molecular data to list
        mol_data_list.append(data)
        
        # Handle pocket data with size limits and proper batch indexing
        if hasattr(data, 'pocket_x') and data.pocket_x is not None:
            pocket_x = data.pocket_x
            pocket_pos = data.pocket_pos
            pocket_edge_index = getattr(data, 'pocket_edge_index', None)
            pocket_edge_attr = getattr(data, 'pocket_edge_attr', None)
            
            # CRITICAL: Limit pocket size to prevent memory issues
            if pocket_x.size(0) > max_pocket_atoms_per_mol:
                # Randomly sample atoms to stay within limit
                indices = torch.randperm(pocket_x.size(0))[:max_pocket_atoms_per_mol]
                pocket_x = pocket_x[indices]
                pocket_pos = pocket_pos[indices]
                
                # Update edge indices if present
                if pocket_edge_index is not None:
                    # Create mapping from old to new indices
                    old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(indices)}
                    
                    # Filter edges that reference kept atoms
                    mask = torch.zeros(pocket_edge_index.size(1), dtype=torch.bool)
                    new_edge_index = []
                    
                    for i in range(pocket_edge_index.size(1)):
                        src, dst = pocket_edge_index[0, i].item(), pocket_edge_index[1, i].item()
                        if src in old_to_new and dst in old_to_new:
                            new_edge_index.append([old_to_new[src], old_to_new[dst]])
                            mask[i] = True
                    
                    if new_edge_index:
                        pocket_edge_index = torch.tensor(new_edge_index, dtype=torch.long).t()
                    else:
                        pocket_edge_index = torch.zeros((2, 0), dtype=torch.long)
                    
                    if pocket_edge_attr is not None:
                        pocket_edge_attr = pocket_edge_attr[mask]
            
            # CRITICAL FIX: Create proper pocket batch indices
            # Each pocket atom gets the molecule index it belongs to
            pocket_batch_indices = torch.full((pocket_x.size(0),), mol_idx, dtype=torch.long)
            
            # Create pocket data with all required attributes
            pocket_data = Data(
                x=pocket_x,
                pos=pocket_pos,
                edge_index=pocket_edge_index if pocket_edge_index is not None else torch.zeros((2, 0), dtype=torch.long),
                edge_attr=pocket_edge_attr if pocket_edge_attr is not None else torch.zeros((0, 1), dtype=torch.float),
                batch=pocket_batch_indices  # THIS WAS MISSING - CRITICAL FIX!
            )
            pocket_data_list.append(pocket_data)
    
    try:
        # Batch molecular data
        mol_batch = Batch.from_data_list(mol_data_list)
        
        # Batch pocket data if available
        if pocket_data_list:
            pocket_batch = Batch.from_data_list(pocket_data_list)
            
            # Add pocket data to molecular batch with proper attributes
            mol_batch.pocket_x = pocket_batch.x
            mol_batch.pocket_pos = pocket_batch.pos
            mol_batch.pocket_edge_index = pocket_batch.edge_index
            mol_batch.pocket_edge_attr = pocket_batch.edge_attr
            mol_batch.pocket_batch = pocket_batch.batch  # CRITICAL: This ensures pocket_batch is present!
            
            # Validation: ensure pocket batch indices are valid
            max_mol_batch = mol_batch.batch.max().item()
            max_pocket_batch = mol_batch.pocket_batch.max().item()
            
            if max_pocket_batch > max_mol_batch:
                print(f"Warning: Fixing pocket batch indices: {max_pocket_batch} -> {max_mol_batch}")
                mol_batch.pocket_batch = torch.clamp(mol_batch.pocket_batch, 0, max_mol_batch)
        else:
            # No pocket data - add None attributes to avoid AttributeError
            mol_batch.pocket_x = None
            mol_batch.pocket_pos = None
            mol_batch.pocket_edge_index = None
            mol_batch.pocket_edge_attr = None
            mol_batch.pocket_batch = None
        
        return mol_batch
        
    except Exception as e:
        print(f"Collation error: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: return simple molecular batch without pocket
        try:
            mol_batch = Batch.from_data_list(mol_data_list)
            # Add None pocket attributes to prevent AttributeError
            mol_batch.pocket_x = None
            mol_batch.pocket_pos = None
            mol_batch.pocket_edge_index = None
            mol_batch.pocket_edge_attr = None
            mol_batch.pocket_batch = None
            return mol_batch
        except Exception as e2:
            print(f"Even fallback collation failed: {e2}")
            return None

class CrossDockDataLoader:
    """Factory class for creating CrossDock data loaders with proper pocket handling"""
    
    @staticmethod
    def create_train_loader(config: Dict[str, Any]) -> DataLoader:
        """Create training data loader with fixed collation"""
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
            pin_memory=config['data'].get('pin_memory', False),  # Disabled for stability
            collate_fn=safe_collate_crossdock_data,
            drop_last=True,
            persistent_workers=False  # Disabled for stability
        )
    
    @staticmethod
    def create_val_loader(config: Dict[str, Any]) -> DataLoader:
        """Create validation data loader"""
        val_path = config['data'].get('val_path', config['data']['test_path'])
        
        dataset = CrossDockMolecularDataset(
            data_path=val_path,
            include_pocket=config.get('include_pocket', True),
            max_atoms=config.get('max_atoms', 50),
            augment=False
        )
        
        return GeometricDataLoader(
            dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data'].get('num_workers', 4),
            pin_memory=config['data'].get('pin_memory', False),
            collate_fn=safe_collate_crossdock_data,
            drop_last=False,
            persistent_workers=False
        )
    
    @staticmethod
    def create_test_loader(config: Dict[str, Any]) -> DataLoader:
        """Create test data loader"""
        dataset = CrossDockMolecularDataset(
            data_path=config['data']['test_path'],
            include_pocket=config.get('include_pocket', True),
            max_atoms=config.get('max_atoms', 50),
            augment=False
        )
        
        return GeometricDataLoader(
            dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data'].get('num_workers', 4),
            pin_memory=config['data'].get('pin_memory', False),
            collate_fn=safe_collate_crossdock_data,
            drop_last=False,
            persistent_workers=False
        )
    
    @staticmethod
    def create_train_val_split_loader(config: Dict[str, Any], val_ratio: float = 0.1):
        """Create train/val loaders by splitting train data"""
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
        
        # Create loaders
        train_loader = GeometricDataLoader(
            train_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=config['data'].get('num_workers', 4),
            pin_memory=config['data'].get('pin_memory', False),
            collate_fn=safe_collate_crossdock_data,
            drop_last=True,
            persistent_workers=False
        )
        
        val_loader = GeometricDataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data'].get('num_workers', 4),
            pin_memory=config['data'].get('pin_memory', False),
            collate_fn=safe_collate_crossdock_data,
            drop_last=False,
            persistent_workers=False
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
            num_workers=config.get('num_workers', 2),
            pin_memory=False,
            persistent_workers=False
        )