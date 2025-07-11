# src/models/base_model.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseModel(nn.Module, ABC):
    """Abstract base class for all models"""
    
    def __init__(self):
        super().__init__()
        self.model_name = self.__class__.__name__
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass of the model"""
        pass
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_checkpoint(self, path: str, optimizer_state: Optional[Dict] = None,
                       scheduler_state: Optional[Dict] = None, 
                       epoch: int = 0, loss: float = 0.0):
        """Save model checkpoint"""
        checkpoint = {
            'model_name': self.model_name,
            'model_state_dict': self.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'num_parameters': self.get_num_parameters()
        }
        
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state
        if scheduler_state:
            checkpoint['scheduler_state_dict'] = scheduler_state
            
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, device: str = 'cpu') -> Dict[str, Any]:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
    
    def freeze_parameters(self, module_names: list = None):
        """Freeze parameters of specified modules"""
        if module_names is None:
            # Freeze all parameters
            for param in self.parameters():
                param.requires_grad = False
        else:
            # Freeze specific modules
            for name, module in self.named_modules():
                if any(mod_name in name for mod_name in module_names):
                    for param in module.parameters():
                        param.requires_grad = False
    
    def unfreeze_parameters(self, module_names: list = None):
        """Unfreeze parameters of specified modules"""
        if module_names is None:
            # Unfreeze all parameters
            for param in self.parameters():
                param.requires_grad = True
        else:
            # Unfreeze specific modules
            for name, module in self.named_modules():
                if any(mod_name in name for mod_name in module_names):
                    for param in module.parameters():
                        param.requires_grad = True

class MolecularModel(BaseModel):
    """Base class for molecular models"""
    
    def __init__(self, atom_types: int, bond_types: int, hidden_dim: int):
        super().__init__()
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.hidden_dim = hidden_dim
    
    def compute_molecular_properties(self, x: torch.Tensor, pos: torch.Tensor,
                                   edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute molecular properties from graph representation"""
        # This can be overridden by subclasses to compute specific properties
        return {}
    
    def validate_input_shapes(self, x: torch.Tensor, pos: torch.Tensor,
                            edge_index: torch.Tensor):
        """Validate input tensor shapes"""
        assert x.dim() == 2, f"Node features should be 2D, got {x.dim()}D"
        assert pos.dim() == 2 and pos.size(1) == 3, f"Positions should be [N, 3], got {pos.shape}"
        assert edge_index.dim() == 2 and edge_index.size(0) == 2, f"Edge index should be [2, E], got {edge_index.shape}"
        assert x.size(0) == pos.size(0), f"Mismatch between nodes and positions: {x.size(0)} vs {pos.size(0)}"