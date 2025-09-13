import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from .molecular_dataset import CrossDockMolecularDataset
from .pocket_dataset import ProteinPocketDataset
from typing import Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_loaders.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def safe_collate_crossdock_data(batch, max_pocket_atoms_per_mol: int = 200):
    if not batch or len(batch) == 0:
        logger.error("Received empty or None batch")
        return None
    
    # Filter out None entries
    valid_batch = [data for data in batch if data is not None]
    if not valid_batch:
        logger.error("All batch entries are None")
        return None
    
    # Initialize lists for ligand and pocket data
    ligand_data_list = []
    pocket_data_list = []
    
    for mol_idx, data in enumerate(valid_batch):
        # Validate ligand batch indices
        if hasattr(data, 'batch') and data.batch is not None:
            if data.batch.min() < 0 or data.batch.max() >= len(valid_batch):
                logger.warning(f"Invalid ligand batch indices in molecule {mol_idx}: min={data.batch.min()}, max={data.batch.max()}. Setting to {mol_idx}.")
                data.batch = torch.full((data.x.size(0),), mol_idx, dtype=torch.long)
        
        # Ligand data
        ligand_data = {
            'pos': data.pos,
            'features': data.x,
            'mask': data.batch if hasattr(data, 'batch') and data.batch is not None else torch.full((data.x.size(0),), mol_idx, dtype=torch.long),
            'size': torch.tensor([data.x.size(0)], dtype=torch.long)
        }
        ligand_data_list.append(ligand_data)
        
        # Pocket data
        if hasattr(data, 'pocket_x') and data.pocket_x is not None:
            pocket_x = data.pocket_x
            pocket_pos = data.pocket_pos
            pocket_edge_index = getattr(data, 'pocket_edge_index', None)
            pocket_edge_attr = getattr(data, 'pocket_edge_attr', None)
            
            # Smart pocket atom selection
            if pocket_x.size(0) > max_pocket_atoms_per_mol:
                if hasattr(data, 'pos') and data.pos is not None:
                    ligand_center = data.pos.mean(dim=0)
                    distances = torch.norm(pocket_pos - ligand_center, dim=1)
                    _, indices = torch.topk(distances, k=max_pocket_atoms_per_mol, largest=False)
                else:
                    indices = torch.randperm(pocket_x.size(0))[:max_pocket_atoms_per_mol]
                
                pocket_x = pocket_x[indices]
                pocket_pos = pocket_pos[indices]
                
                if pocket_edge_index is not None and pocket_edge_index.size(1) > 0:
                    old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(indices)}
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
            
            num_pocket_atoms = pocket_x.size(0)
            pocket_batch_indices = torch.full((num_pocket_atoms,), mol_idx, dtype=torch.long)
            
            pocket_data = {
                'pos': pocket_pos,
                'features': pocket_x,
                'mask': pocket_batch_indices,
                'size': torch.tensor([num_pocket_atoms], dtype=torch.long)
            }
            pocket_data_list.append(pocket_data)
    
    try:
        # Create batched ligand and pocket dictionaries
        ligand = {
            'pos': torch.cat([d['pos'] for d in ligand_data_list], dim=0),
            'features': torch.cat([d['features'] for d in ligand_data_list], dim=0),
            'mask': torch.cat([d['mask'] for d in ligand_data_list], dim=0),
            'size': torch.cat([d['size'] for d in ligand_data_list], dim=0)
        }
        pocket = {
            'pos': torch.cat([d['pos'] for d in pocket_data_list], dim=0) if pocket_data_list else torch.zeros(0, 3, dtype=torch.float),
            'features': torch.cat([d['features'] for d in pocket_data_list], dim=0) if pocket_data_list else torch.zeros(0, ligand_data_list[0]['features'].size(1), dtype=torch.float),
            'mask': torch.cat([d['mask'] for d in pocket_data_list], dim=0) if pocket_data_list else torch.zeros(0, dtype=torch.long),
            'size': torch.cat([d['size'] for d in pocket_data_list], dim=0) if pocket_data_list else torch.zeros(0, dtype=torch.long)
        }
        
        # Validate batch indices
        num_molecules = len(valid_batch)
        if ligand['mask'].min() < 0 or ligand['mask'].max() >= num_molecules:
            logger.warning(f"Invalid ligand mask indices: min={ligand['mask'].min()}, max={ligand['mask'].max()}. Resetting to valid range.")
            ligand['mask'] = torch.cat([torch.full((d['size'].item(),), i, dtype=torch.long) for i, d in enumerate(ligand_data_list)], dim=0)
        
        if pocket_data_list and (pocket['mask'].min() < 0 or pocket['mask'].max() >= num_molecules):
            logger.warning(f"Invalid pocket mask indices: min={pocket['mask'].min()}, max={pocket['mask'].max()}. Resetting to valid range.")
            pocket['mask'] = torch.cat([torch.full((d['size'].item(),), i, dtype=torch.long) for i, d in enumerate(pocket_data_list)], dim=0)
        
        logger.debug(f"Ligand mask min/max: {ligand['mask'].min()}/{ligand['mask'].max()}")
        logger.debug(f"Pocket mask min/max: {pocket['mask'].min()}/{pocket['mask'].max()}")
        
        return {'ligand': ligand, 'pocket': pocket}
    
    except Exception as e:
        logger.error(f"Error in collation: {e}")
        return None

# Rest of data_loaders.py remains unchanged
class CrossDockDataLoader:
    """Clean DataLoader factory without debug messages"""
    
    @staticmethod
    def create_train_loader(config: Dict[str, Any]) -> DataLoader:
        """Create training loader"""
        
        dataset = CrossDockMolecularDataset(
            data_path=config['train_path'],
            include_pocket=config.get('include_pocket', True),
            max_atoms=config.get('max_atoms', 50),
            augment=config.get('augment', True)
        )
        
        return DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=config.get('shuffle', True),
            num_workers=config.get('num_workers', 4),
            pin_memory=config.get('pin_memory', True),
            collate_fn=safe_collate_crossdock_data,
            drop_last=True,
            persistent_workers=config.get('num_workers', 4) > 0
        )
    
    @staticmethod
    def create_val_loader(config: Dict[str, Any]) -> DataLoader:
        """Create validation loader"""
        
        val_path = config.get('val_path') or config.get('test_path')
        
        dataset = CrossDockMolecularDataset(
            data_path=val_path,
            include_pocket=config.get('include_pocket', True),
            max_atoms=config.get('max_atoms', 50),
            augment=False
        )
        
        return DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=config.get('pin_memory', True),
            collate_fn=safe_collate_crossdock_data,
            drop_last=False,
            persistent_workers=config.get('num_workers', 4) > 0
        )
    
    @staticmethod
    def create_test_loader(config: Dict[str, Any]) -> DataLoader:
        """Create test loader"""
        
        dataset = CrossDockMolecularDataset(
            data_path=config['test_path'],
            include_pocket=config.get('include_pocket', True),
            max_atoms=config.get('max_atoms', 50),
            augment=False
        )
        
        return DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=config.get('pin_memory', True),
            collate_fn=safe_collate_crossdock_data,
            drop_last=False,
            persistent_workers=config.get('num_workers', 4) > 0
        )

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