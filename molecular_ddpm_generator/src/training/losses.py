# src/training/losses.py - DDPM-focused losses
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

class MolecularLoss(nn.Module):
    """Combined loss function for molecular DDPM generation"""
    
    def __init__(self, position_weight: float = 1.0, atom_weight: float = 0.1,
                 bond_weight: float = 0.1, property_weight: float = 0.01):
        super().__init__()
        self.position_weight = position_weight
        self.atom_weight = atom_weight
        self.bond_weight = bond_weight
        self.property_weight = property_weight
        
        # Individual loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute combined molecular loss for DDPM"""
        losses = {}
        total_loss = 0.0
        
        # Main DDPM noise prediction loss
        if 'noise_pred' in predictions and 'noise_target' in targets:
            noise_loss = self.mse_loss(predictions['noise_pred'], targets['noise_target'])
            losses['noise_loss'] = noise_loss
            total_loss += self.position_weight * noise_loss
        
        # Atom type loss (auxiliary)
        if 'atom_logits' in predictions and 'atom_types' in targets:
            atom_loss = self.ce_loss(predictions['atom_logits'], targets['atom_types'].squeeze())
            losses['atom_loss'] = atom_loss
            total_loss += self.atom_weight * atom_loss
        
        # Bond type loss (auxiliary)
        if 'bond_logits' in predictions and 'bond_types' in targets:
            bond_loss = self.ce_loss(predictions['bond_logits'], targets['bond_types'].squeeze())
            losses['bond_loss'] = bond_loss
            total_loss += self.bond_weight * bond_loss
        
        losses['total_loss'] = total_loss
        return losses

class DDPMLoss(nn.Module):
    """Pure DDPM loss for noise prediction"""
    
    def __init__(self, loss_type: str = 'mse'):
        super().__init__()
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'huber':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, noise_pred: torch.Tensor, noise_target: torch.Tensor) -> torch.Tensor:
        """Simple DDPM noise prediction loss"""
        return self.loss_fn(noise_pred, noise_target)

class GeometryConsistencyLoss(nn.Module):
    """Loss to enforce geometric consistency in molecular structures"""
    
    def __init__(self, bond_length_weight: float = 1.0, angle_weight: float = 0.5):
        super().__init__()
        self.bond_length_weight = bond_length_weight
        self.angle_weight = angle_weight
        
    def forward(self, positions: torch.Tensor, edge_index: torch.Tensor,
                bond_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Compute geometry consistency loss"""
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=positions.device)
        
        row, col = edge_index
        
        # Bond length consistency
        bond_vectors = positions[row] - positions[col]
        bond_lengths = torch.norm(bond_vectors, dim=-1)
        
        # Expected bond lengths (chemical knowledge)
        expected_lengths = self._get_expected_bond_lengths(bond_types)
        bond_loss = F.mse_loss(bond_lengths, expected_lengths)
        
        return self.bond_length_weight * bond_loss
    
    def _get_expected_bond_lengths(self, bond_types: torch.Tensor) -> torch.Tensor:
        """Get expected bond lengths based on chemical knowledge"""
        # Chemical bond length standards (Å)
        length_map = {
            0: 1.54,  # C-C single
            1: 1.34,  # C=C double  
            2: 1.20,  # C≡C triple
            3: 1.40   # Aromatic
        }
        
        expected = torch.full_like(bond_types, 1.54, dtype=torch.float)
        for bond_type, length in length_map.items():
            mask = bond_types == bond_type
            expected[mask] = length
        
        return expected