# src/training/losses.py - Updated with constraint losses
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

class MolecularLoss(nn.Module):
    """Combined loss function for molecular DDPM generation with constraints"""
    
    def __init__(self, 
                 position_weight: float = 1.0, 
                 atom_weight: float = 0.1,
                 bond_weight: float = 0.1, 
                 property_weight: float = 0.01,
                 constraint_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.position_weight = position_weight
        self.atom_weight = atom_weight
        self.bond_weight = bond_weight
        self.property_weight = property_weight
        
        # Constraint weights
        self.constraint_weights = constraint_weights or {
            'bond_length': 0.1,
            'valence': 0.05,
            'steric_clash': 0.02,
            'total_constraint': 0.2
        }
        
        # Individual loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        
        # Constraint-specific losses
        self.bond_length_loss = nn.MSELoss()
        self.valence_loss = nn.CrossEntropyLoss()
        self.steric_loss = nn.ReLU()  # Only penalize violations
        
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute combined molecular loss for DDPM with constraints"""
        losses = {}
        total_loss = 0.0
        
        # Main DDPM noise prediction loss
        if 'noise_pred' in predictions and 'noise_target' in targets:
            noise_loss = self.mse_loss(predictions['noise_pred'], targets['noise_target'])
            losses['noise_loss'] = noise_loss
            total_loss += self.position_weight * noise_loss
        
        # Position prediction loss (if available)
        if 'pos_pred' in predictions and 'pos_target' in targets:
            pos_loss = self.mse_loss(predictions['pos_pred'], targets['pos_target'])
            losses['position_loss'] = pos_loss
            total_loss += self.position_weight * pos_loss
        
        # Atom type loss (auxiliary)
        if 'atom_logits' in predictions and 'atom_types' in targets:
            atom_target = targets['atom_types']
            if atom_target.dim() > 1:
                atom_target = atom_target.squeeze()
            atom_loss = self.ce_loss(predictions['atom_logits'], atom_target.long())
            losses['atom_loss'] = atom_loss
            total_loss += self.atom_weight * atom_loss
        
        # Bond type loss (auxiliary)
        if 'bond_logits' in predictions and 'bond_types' in targets:
            bond_target = targets['bond_types']
            if bond_target.dim() > 1:
                bond_target = bond_target.squeeze()
            bond_loss = self.ce_loss(predictions['bond_logits'], bond_target.long())
            losses['bond_loss'] = bond_loss
            total_loss += self.bond_weight * bond_loss
        
        # Constraint losses
        constraint_loss = self._compute_constraint_losses(predictions, targets, losses)
        total_loss += constraint_loss
        
        # Property losses (if available)
        if 'chemical_properties' in predictions and 'target_properties' in targets:
            prop_loss = self.mse_loss(
                predictions['chemical_properties'], 
                targets['target_properties']
            )
            losses['property_loss'] = prop_loss
            total_loss += self.property_weight * prop_loss
        
        # Consistency loss
        if 'consistency_score' in predictions:
            # Encourage high consistency (score close to 1)
            consistency_target = torch.ones_like(predictions['consistency_score'])
            consistency_loss = self.mse_loss(predictions['consistency_score'], consistency_target)
            losses['consistency_loss'] = consistency_loss
            total_loss += 0.01 * consistency_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def _compute_constraint_losses(self, predictions: Dict[str, torch.Tensor],
                                 targets: Dict[str, torch.Tensor],
                                 losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute constraint violation losses"""
        total_constraint_loss = 0.0
        
        # Direct constraint losses from model
        if 'constraint_losses' in predictions:
            constraint_dict = predictions['constraint_losses']
            
            for constraint_name, constraint_loss in constraint_dict.items():
                if constraint_loss is not None and constraint_loss.requires_grad:
                    weight = self.constraint_weights.get(constraint_name, 0.1)
                    weighted_loss = weight * constraint_loss
                    losses[f'{constraint_name}_constraint'] = constraint_loss
                    total_constraint_loss += weighted_loss
        
        # Total constraint loss from model
        if 'total_constraint_loss' in predictions:
            model_constraint_loss = predictions['total_constraint_loss']
            if model_constraint_loss is not None and model_constraint_loss.requires_grad:
                weight = self.constraint_weights.get('total_constraint', 0.2)
                total_constraint_loss += weight * model_constraint_loss
                losses['model_constraint_loss'] = model_constraint_loss
        
        # Additional constraint losses based on predictions vs targets
        
        # Valence consistency loss
        if ('valence_predictions' in predictions and 
            'atom_types' in targets and 
            'edge_index' in targets):
            valence_loss = self._compute_valence_consistency_loss(
                predictions['valence_predictions'],
                targets['atom_types'],
                targets['edge_index']
            )
            losses['valence_consistency_loss'] = valence_loss
            total_constraint_loss += self.constraint_weights.get('valence', 0.05) * valence_loss
        
        # Bond type consistency loss
        if ('bond_type_predictions' in predictions and 
            'bond_types' in targets):
            bond_consistency_loss = self.ce_loss(
                predictions['bond_type_predictions'],
                targets['bond_types'].long().squeeze()
            ) if predictions['bond_type_predictions'] is not None else torch.tensor(0.0)
            losses['bond_consistency_loss'] = bond_consistency_loss
            total_constraint_loss += self.constraint_weights.get('bond_length', 0.1) * bond_consistency_loss
        
        return total_constraint_loss
    
    def _compute_valence_consistency_loss(self, valence_pred: torch.Tensor,
                                        atom_types: torch.Tensor,
                                        edge_index: torch.Tensor) -> torch.Tensor:
        """Compute valence consistency loss"""
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=valence_pred.device, requires_grad=True)
        
        # Count actual bonds per atom
        row, col = edge_index
        bond_counts = torch.zeros(valence_pred.size(0), device=valence_pred.device)
        bond_counts.index_add_(0, row, torch.ones(row.size(0), device=valence_pred.device))
        
        # Convert bond counts to target valence distribution
        max_valence = valence_pred.size(1)
        valence_targets = torch.zeros_like(valence_pred)
        
        for i, count in enumerate(bond_counts):
            target_valence = min(int(count.item()), max_valence - 1)
            valence_targets[i, target_valence] = 1.0
        
        # Cross-entropy loss between predicted and target valence
        valence_loss = -torch.sum(valence_targets * torch.log(valence_pred + 1e-8)) / valence_pred.size(0)
        
        return valence_loss

class DDPMLoss(nn.Module):
    """Pure DDPM loss for noise prediction with optional constraints"""
    
    def __init__(self, loss_type: str = 'mse', include_constraints: bool = True,
                 constraint_weight: float = 0.1):
        super().__init__()
        self.include_constraints = include_constraints
        self.constraint_weight = constraint_weight
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'huber':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, noise_pred: torch.Tensor, noise_target: torch.Tensor,
                model_outputs: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Simple DDPM noise prediction loss with optional constraints"""
        
        # Main DDPM loss
        ddpm_loss = self.loss_fn(noise_pred, noise_target)
        
        # Add constraint losses if available and enabled
        if (self.include_constraints and 
            model_outputs is not None and 
            'total_constraint_loss' in model_outputs):
            
            constraint_loss = model_outputs['total_constraint_loss']
            if constraint_loss is not None and constraint_loss.requires_grad:
                total_loss = ddpm_loss + self.constraint_weight * constraint_loss
                return total_loss
        
        return ddpm_loss

class GeometryConsistencyLoss(nn.Module):
    """Loss to enforce geometric consistency in molecular structures"""
    
    def __init__(self, bond_length_weight: float = 1.0, angle_weight: float = 0.5,
                 dihedral_weight: float = 0.2):
        super().__init__()
        self.bond_length_weight = bond_length_weight
        self.angle_weight = angle_weight
        self.dihedral_weight = dihedral_weight
        
    def forward(self, positions: torch.Tensor, edge_index: torch.Tensor,
                bond_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Compute geometry consistency loss"""
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=positions.device, requires_grad=True)
        
        total_loss = 0.0
        
        # Bond length consistency
        bond_loss = self._compute_bond_length_loss(positions, edge_index, bond_types)
        total_loss += self.bond_length_weight * bond_loss
        
        # Bond angle consistency (if we have enough bonds)
        if edge_index.size(1) > 2:
            angle_loss = self._compute_bond_angle_loss(positions, edge_index)
            total_loss += self.angle_weight * angle_loss
        
        return total_loss
    
    def _compute_bond_length_loss(self, positions: torch.Tensor, 
                                 edge_index: torch.Tensor,
                                 bond_types: torch.Tensor) -> torch.Tensor:
        """Compute bond length consistency loss"""
        row, col = edge_index
        
        # Current bond lengths
        bond_vectors = positions[row] - positions[col]
        bond_lengths = torch.norm(bond_vectors, dim=-1)
        
        # Expected bond lengths based on types
        expected_lengths = self._get_expected_bond_lengths(bond_types)
        
        # MSE loss between current and expected lengths
        bond_loss = F.mse_loss(bond_lengths, expected_lengths)
        
        return bond_loss
    
    def _compute_bond_angle_loss(self, positions: torch.Tensor,
                                edge_index: torch.Tensor) -> torch.Tensor:
        """Compute bond angle consistency loss"""
        # This is a simplified version - a full implementation would
        # require more sophisticated angle detection
        
        # For now, just return zero
        return torch.tensor(0.0, device=positions.device, requires_grad=True)
    
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
            if bond_types.dim() > 1:
                bond_type_values = bond_types[:, 0] if bond_types.size(1) > 0 else bond_types
            else:
                bond_type_values = bond_types
            
            mask = bond_type_values == bond_type
            expected[mask] = length
        
        return expected

class PropertyConstraintLoss(nn.Module):
    """Loss to enforce molecular property constraints"""
    
    def __init__(self, target_properties: Dict[str, tuple] = None):
        super().__init__()
        # target_properties: {'property_name': (min_val, max_val)}
        self.target_properties = target_properties or {
            'molecular_weight': (100, 500),
            'logp': (-2, 5),
            'num_atoms': (5, 50)
        }
    
    def forward(self, predicted_properties: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute property constraint loss"""
        total_loss = 0.0
        
        for prop_name, (min_val, max_val) in self.target_properties.items():
            if prop_name in predicted_properties:
                prop_values = predicted_properties[prop_name]
                
                # Penalty for values outside range
                below_min = F.relu(min_val - prop_values)
                above_max = F.relu(prop_values - max_val)
                
                prop_loss = torch.mean(below_min + above_max)
                total_loss += prop_loss
        
        return total_loss

class CombinedConstraintLoss(nn.Module):
    """Combined constraint loss wrapper"""
    
    def __init__(self, constraint_weights: Dict[str, float] = None):
        super().__init__()
        
        self.constraint_weights = constraint_weights or {
            'geometry': 0.1,
            'property': 0.05,
            'chemical': 0.1
        }
        
        self.geometry_loss = GeometryConsistencyLoss()
        self.property_loss = PropertyConstraintLoss()
        
    def forward(self, model_outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute all constraint losses"""
        
        constraint_losses = {}
        total_constraint_loss = 0.0
        
        # Geometry constraints
        if all(key in targets for key in ['positions', 'edge_index', 'bond_types', 'batch']):
            geom_loss = self.geometry_loss(
                targets['positions'], targets['edge_index'], 
                targets['bond_types'], targets['batch']
            )
            constraint_losses['geometry'] = geom_loss
            total_constraint_loss += self.constraint_weights['geometry'] * geom_loss
        
        # Property constraints
        if 'chemical_properties' in model_outputs:
            # Convert model outputs to property dict
            prop_dict = {'molecular_weight': model_outputs['chemical_properties'][:, 0]}
            prop_loss = self.property_loss(prop_dict)
            constraint_losses['property'] = prop_loss
            total_constraint_loss += self.constraint_weights['property'] * prop_loss
        
        # Model-provided constraint losses
        if 'constraint_losses' in model_outputs:
            model_constraints = model_outputs['constraint_losses']
            for name, loss in model_constraints.items():
                if loss is not None and loss.requires_grad:
                    constraint_losses[f'model_{name}'] = loss
                    weight = self.constraint_weights.get('chemical', 0.1)
                    total_constraint_loss += weight * loss
        
        constraint_losses['total_constraint'] = total_constraint_loss
        return constraint_losses