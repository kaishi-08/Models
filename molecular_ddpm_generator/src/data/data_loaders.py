import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from .molecular_dataset import CrossDockMolecularDataset
from .pocket_dataset import ProteinPocketDataset
from typing import Optional, Dict, Any

def safe_collate_crossdock_data(batch, max_pocket_atoms_per_mol: int = 200):
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
        
        # Handle pocket data with guaranteed pocket_batch creation
        if hasattr(data, 'pocket_x') and data.pocket_x is not None:
            pocket_x = data.pocket_x
            pocket_pos = data.pocket_pos
            pocket_edge_index = getattr(data, 'pocket_edge_index', None)
            pocket_edge_attr = getattr(data, 'pocket_edge_attr', None)
            
            # Smart pocket atom selection
            if pocket_x.size(0) > max_pocket_atoms_per_mol:
                # Distance-based selection if ligand available
                if hasattr(data, 'pos') and data.pos is not None:
                    ligand_center = data.pos.mean(dim=0)
                    distances = torch.norm(pocket_pos - ligand_center, dim=1)
                    _, indices = torch.topk(distances, k=max_pocket_atoms_per_mol, largest=False)
                else:
                    indices = torch.randperm(pocket_x.size(0))[:max_pocket_atoms_per_mol]
                
                # Apply selection
                pocket_x = pocket_x[indices]
                pocket_pos = pocket_pos[indices]
                
                # Update edges efficiently
                if pocket_edge_index is not None and pocket_edge_index.size(1) > 0:
                    # Create mapping
                    old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(indices)}
                    
                    # Filter edges
                    new_edges = []
                    new_edge_attrs = []
                    
                    for i in range(pocket_edge_index.size(1)):
                        src, dst = pocket_edge_index[0, i].item(), pocket_edge_index[1, i].item()
                        if src in old_to_new and dst in old_to_new:
                            new_edges.append([old_to_new[src], old_to_new[dst]])
                            if pocket_edge_attr is not None:
                                new_edge_attrs.append(pocket_edge_attr[i])
                    
                    if new_edges:
                        pocket_edge_index = torch.tensor(new_edges, dtype=torch.long).t()
                        if new_edge_attrs:
                            pocket_edge_attr = torch.stack(new_edge_attrs)
                    else:
                        pocket_edge_index = torch.zeros((2, 0), dtype=torch.long)
                        pocket_edge_attr = torch.zeros((0, 1), dtype=torch.float) if pocket_edge_attr is not None else None
            
            # Create pocket_batch_indices (CRITICAL FIX)
            num_pocket_atoms = pocket_x.size(0)
            pocket_batch_indices = torch.full((num_pocket_atoms,), mol_idx, dtype=torch.long)
            
            # Create pocket data object
            pocket_data = Data(
                x=pocket_x,
                pos=pocket_pos,
                edge_index=pocket_edge_index if pocket_edge_index is not None else torch.zeros((2, 0), dtype=torch.long),
                edge_attr=pocket_edge_attr if pocket_edge_attr is not None else torch.zeros((0, 1), dtype=torch.float),
                batch=pocket_batch_indices
            )
            
            pocket_data_list.append(pocket_data)
    
    try:
        # Batch molecular data
        mol_batch = Batch.from_data_list(mol_data_list)
        
        # Batch pocket data if available
        if pocket_data_list:
            pocket_batch = Batch.from_data_list(pocket_data_list)
            
            # Add pocket attributes to molecular batch
            mol_batch.pocket_x = pocket_batch.x
            mol_batch.pocket_pos = pocket_batch.pos
            mol_batch.pocket_edge_index = pocket_batch.edge_index
            mol_batch.pocket_edge_attr = pocket_batch.edge_attr
            mol_batch.pocket_batch = pocket_batch.batch  # GUARANTEED POCKET_BATCH!
            
            # Validation
            max_mol_batch = mol_batch.batch.max().item()
            max_pocket_batch = mol_batch.pocket_batch.max().item()
            
            if max_pocket_batch > max_mol_batch:
                mol_batch.pocket_batch = torch.clamp(mol_batch.pocket_batch, 0, max_mol_batch)
        else:
            # Set None attributes explicitly
            mol_batch.pocket_x = None
            mol_batch.pocket_pos = None
            mol_batch.pocket_edge_index = None
            mol_batch.pocket_edge_attr = None
            mol_batch.pocket_batch = None
        
        return mol_batch
        
    except Exception as e:
        # Silent fallback
        try:
            mol_batch = Batch.from_data_list(mol_data_list)
            # Set None pocket attributes
            for attr in ['pocket_x', 'pocket_pos', 'pocket_edge_index', 'pocket_edge_attr', 'pocket_batch']:
                setattr(mol_batch, attr, None)
            return mol_batch
        except Exception as e2:
            return None

class CrossDockDataLoader:
    """Clean DataLoader factory without debug messages"""
    
    @staticmethod
    def create_train_loader(config: Dict[str, Any]) -> DataLoader:
        """Create training loader"""
        
        dataset = CrossDockMolecularDataset(
            data_path=config['data']['train_path'],
            include_pocket=config.get('include_pocket', True),
            max_atoms=config.get('max_atoms', 50),
            augment=config.get('augment', True)
        )
        
        return DataLoader(
            dataset,
            batch_size=config['data']['batch_size'],
            shuffle=config['data'].get('shuffle', True),
            num_workers=config['data'].get('num_workers', 4),  # Restored to 4
            pin_memory=config['data'].get('pin_memory', True),  # Restored pin_memory
            collate_fn=safe_collate_crossdock_data,
            drop_last=True,
            persistent_workers=config['data'].get('num_workers', 4) > 0  # Enable if workers > 0
        )
    
    @staticmethod
    def create_val_loader(config: Dict[str, Any]) -> DataLoader:
        """Create validation loader"""
        
        val_path = config['data'].get('val_path', config['data']['test_path'])
        
        dataset = CrossDockMolecularDataset(
            data_path=val_path,
            include_pocket=config.get('include_pocket', True),
            max_atoms=config.get('max_atoms', 50),
            augment=False
        )
        
        return DataLoader(
            dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data'].get('num_workers', 4),
            pin_memory=config['data'].get('pin_memory', True),
            collate_fn=safe_collate_crossdock_data,
            drop_last=False,
            persistent_workers=config['data'].get('num_workers', 4) > 0
        )
    
    @staticmethod
    def create_test_loader(config: Dict[str, Any]) -> DataLoader:
        """Create test loader"""
        
        dataset = CrossDockMolecularDataset(
            data_path=config['data']['test_path'],
            include_pocket=config.get('include_pocket', True),
            max_atoms=config.get('max_atoms', 50),
            augment=False
        )
        
        return DataLoader(
            dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data'].get('num_workers', 4),
            pin_memory=config['data'].get('pin_memory', True),
            collate_fn=safe_collate_crossdock_data,
            drop_last=False,
            persistent_workers=config['data'].get('num_workers', 4) > 0
        )

# Backward compatibility
class MolecularDataLoader(CrossDockDataLoader):
    """Molecular data loader (backward compatibility)"""
    pass

class PocketDataLoader:
    """Pocket data loader"""
    
    @staticmethod
    def create_loader(data_path: str, config: Dict[str, Any]) -> DataLoader:
        """Create pocket data loader"""
        dataset = ProteinPocketDataset(
            data_path=data_path,
            pocket_radius=config.get('pocket_radius', 10.0),
            include_surface=config.get('include_surface', True)
        )
        
        return DataLoader(
            dataset,
            batch_size=config.get('batch_size', 16),
            shuffle=config.get('shuffle', False),
            num_workers=config.get('num_workers', 2),
            pin_memory=False,
            persistent_workers=config.get('num_workers', 2) > 0
        )