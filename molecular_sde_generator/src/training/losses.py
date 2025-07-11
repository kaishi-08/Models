# src/training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

class MolecularLoss(nn.Module):
    """Combined loss function for molecular generation"""
    
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
        """
        Compute combined molecular loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            batch: Batch indices for batched computation
        """
        losses = {}
        total_loss = 0.0
        
        # Position loss (main SDE loss)
        if 'positions' in predictions and 'positions' in targets:
            pos_loss = self.mse_loss(predictions['positions'], targets['positions'])
            losses['position_loss'] = pos_loss
            total_loss += self.position_weight * pos_loss
        
        # Atom type loss
        if 'atom_logits' in predictions and 'atom_types' in targets:
            atom_loss = self.ce_loss(predictions['atom_logits'], targets['atom_types'].squeeze())
            losses['atom_loss'] = atom_loss
            total_loss += self.atom_weight * atom_loss
        
        # Bond type loss
        if 'bond_logits' in predictions and 'bond_types' in targets:
            bond_loss = self.ce_loss(predictions['bond_logits'], targets['bond_types'].squeeze())
            losses['bond_loss'] = bond_loss
            total_loss += self.bond_weight * bond_loss
        
        # Property prediction loss
        if 'properties' in predictions and 'properties' in targets:
            prop_loss = self.smooth_l1_loss(predictions['properties'], targets['properties'])
            losses['property_loss'] = prop_loss
            total_loss += self.property_weight * prop_loss
        
        losses['total_loss'] = total_loss
        return losses

class ScoreMatchingLoss(nn.Module):
    """Score matching loss for SDE-based diffusion"""
    
    def __init__(self, loss_type: str = 'mse', lambda_reg: float = 0.0):
        super().__init__()
        self.loss_type = loss_type
        self.lambda_reg = lambda_reg
        
        if loss_type == 'mse':
            self.base_loss = nn.MSELoss()
        elif loss_type == 'l1':
            self.base_loss = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, score_pred: torch.Tensor, score_target: torch.Tensor,
                positions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute score matching loss
        
        Args:
            score_pred: Predicted scores [N, 3]
            score_target: Target scores [N, 3]
            positions: Current positions [N, 3]
            t: Time steps [batch_size]
        """
        # Basic score matching loss
        loss = self.base_loss(score_pred, score_target)
        
        # Optional regularization
        if self.lambda_reg > 0:
            # Regularize score magnitude
            score_norm = torch.norm(score_pred, dim=-1, keepdim=True)
            reg_loss = self.lambda_reg * torch.mean(score_norm ** 2)
            loss = loss + reg_loss
        
        return loss

class GeometryConsistencyLoss(nn.Module):
    """Loss to enforce geometric consistency in molecular structures"""
    
    def __init__(self, bond_length_weight: float = 1.0, angle_weight: float = 0.5):
        super().__init__()
        self.bond_length_weight = bond_length_weight
        self.angle_weight = angle_weight
        
    def forward(self, positions: torch.Tensor, edge_index: torch.Tensor,
                bond_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute geometry consistency loss
        
        Args:
            positions: Atomic positions [N, 3]
            edge_index: Bond connectivity [2, E]
            bond_types: Bond type indices [E]
            batch: Batch indices [N]
        """
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=positions.device)
        
        row, col = edge_index
        
        # Bond length consistency
        bond_vectors = positions[row] - positions[col]
        bond_lengths = torch.norm(bond_vectors, dim=-1)
        
        # Expected bond lengths (simplified)
        expected_lengths = self._get_expected_bond_lengths(bond_types)
        bond_loss = F.mse_loss(bond_lengths, expected_lengths)
        
        # Bond angle consistency (for triplets)
        angle_loss = self._compute_angle_loss(positions, edge_index, batch)
        
        return self.bond_length_weight * bond_loss + self.angle_weight * angle_loss
    
    def _get_expected_bond_lengths(self, bond_types: torch.Tensor) -> torch.Tensor:
        """Get expected bond lengths based on bond types"""
        # Simplified bond length expectations
        length_map = {0: 1.5, 1: 1.3, 2: 1.2, 3: 1.1, 4: 1.0}  # Single, double, triple, etc.
        
        expected = torch.zeros_like(bond_types, dtype=torch.float)
        for bond_type, length in length_map.items():
            mask = bond_types == bond_type
            expected[mask] = length
        
        return expected
    
    def _compute_angle_loss(self, positions: torch.Tensor, edge_index: torch.Tensor,
                          batch: torch.Tensor) -> torch.Tensor:
        """Compute bond angle loss for molecular geometry"""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated angle calculations
        return torch.tensor(0.0, device=positions.device)