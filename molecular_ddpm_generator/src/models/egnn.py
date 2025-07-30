# src/models/egnn.py - Enhanced EGNN with Chemical Constraints
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph
from typing import Dict, Optional, List, Tuple

class BondLengthConstraint(nn.Module):
    """Chemical bond length constraint for molecular structures"""
    
    def __init__(self, constraint_weight: float = 0.1, soft_constraint: bool = True):
        super().__init__()
        self.constraint_weight = constraint_weight
        self.soft_constraint = soft_constraint
        
        # Standard bond lengths (Å) - chemical knowledge
        self.register_buffer('target_lengths', torch.tensor({
            0: 1.54,  # C-C single
            1: 1.34,  # C=C double  
            2: 1.20,  # C≡C triple
            3: 1.40,  # Aromatic
            4: 1.47   # Default
        }))
        
        # Learnable adjustment factors
        self.length_adjustment = nn.Parameter(torch.ones(5) * 0.01)
        
    def forward(self, pos: torch.Tensor, edge_index: torch.Tensor, 
                bond_types: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply bond length constraints to positions
        
        Returns:
            constrained_pos: Updated positions
            constraint_loss: Constraint violation loss
        """
        if edge_index.size(1) == 0:
            return pos, torch.tensor(0.0, device=pos.device, requires_grad=True)
        
        row, col = edge_index
        bond_vectors = pos[row] - pos[col]
        current_lengths = torch.norm(bond_vectors, dim=-1, keepdim=True)
        current_lengths = torch.clamp(current_lengths, min=1e-6)  # Avoid division by zero
        
        # Get target lengths based on bond types
        if bond_types is not None:
            bond_types_clamped = torch.clamp(bond_types.long(), 0, 4)
            target_lengths = self.target_lengths[bond_types_clamped].unsqueeze(-1)
            # Apply learnable adjustments
            adjustments = self.length_adjustment[bond_types_clamped].unsqueeze(-1)
            target_lengths = target_lengths + adjustments
        else:
            target_lengths = torch.full_like(current_lengths, 1.54)  # Default C-C length
        
        # Compute constraint loss
        length_errors = torch.abs(current_lengths - target_lengths)
        constraint_loss = torch.mean(length_errors) * self.constraint_weight
        
        if self.soft_constraint:
            # Soft constraint: gradual position adjustment
            length_ratio = target_lengths / current_lengths
            length_ratio = torch.clamp(length_ratio, 0.8, 1.2)  # Limit adjustment magnitude
            
            # Apply gentle correction
            correction_factor = 0.1  # Gentle adjustment
            target_vectors = bond_vectors * length_ratio
            adjustment = (target_vectors - bond_vectors) * correction_factor
            
            # Distribute adjustment between both atoms
            pos_new = pos.clone()
            pos_new.index_add_(0, row, adjustment * 0.5)
            pos_new.index_add_(0, col, -adjustment * 0.5)
            
            return pos_new, constraint_loss
        else:
            # Hard constraint: direct length enforcement
            unit_vectors = bond_vectors / current_lengths
            target_vectors = unit_vectors * target_lengths
            
            pos_new = pos.clone()
            pos_new[col] = pos[row] - target_vectors
            
            return pos_new, constraint_loss

class ValenceConstraint(nn.Module):
    """Chemical valence constraint for atoms"""
    
    def __init__(self, constraint_weight: float = 0.05):
        super().__init__()
        self.constraint_weight = constraint_weight
        
        # Max valence for common atoms
        self.max_valence = {
            1: 1,   # Hydrogen
            6: 4,   # Carbon
            7: 3,   # Nitrogen (can be 5 in some cases)
            8: 2,   # Oxygen (can be 6 in some cases)
            9: 1,   # Fluorine
            15: 5,  # Phosphorus
            16: 6,  # Sulfur
            17: 7,  # Chlorine
        }
        
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, 
                atom_types: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply valence constraints to node features
        
        Returns:
            constrained_h: Updated node features
            constraint_loss: Valence violation loss
        """
        if edge_index.size(1) == 0 or atom_types is None:
            return h, torch.tensor(0.0, device=h.device, requires_grad=True)
        
        # Count bonds per atom
        row, col = edge_index
        bond_counts = torch.zeros(h.size(0), device=h.device)
        bond_counts.index_add_(0, row, torch.ones(row.size(0), device=h.device))
        
        # Compute valence violations
        valence_violations = torch.zeros(h.size(0), device=h.device)
        
        for i, atom_type in enumerate(atom_types):
            if len(atom_type.shape) > 0:
                atom_num = atom_type[0].item() if atom_type.numel() > 0 else 6
            else:
                atom_num = atom_type.item()
            
            max_val = self.max_valence.get(int(atom_num), 4)
            violation = max(0, bond_counts[i] - max_val)
            valence_violations[i] = violation
        
        # Constraint loss
        constraint_loss = torch.mean(valence_violations) * self.constraint_weight
        
        # Apply penalty to over-valent atoms
        h_new = h.clone()
        for i, violation in enumerate(valence_violations):
            if violation > 0:
                # Reduce features for over-valent atoms
                penalty_factor = 1.0 - (violation * 0.1)
                h_new[i] = h_new[i] * penalty_factor
        
        return h_new, constraint_loss

class StericClashConstraint(nn.Module):
    """Steric clash prevention constraint"""
    
    def __init__(self, min_distance: float = 1.0, constraint_weight: float = 0.02):
        super().__init__()
        self.min_distance = min_distance
        self.constraint_weight = constraint_weight
        
    def forward(self, pos: torch.Tensor, batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prevent steric clashes between atoms
        
        Returns:
            constrained_pos: Updated positions
            constraint_loss: Steric clash loss
        """
        if pos.size(0) < 2:
            return pos, torch.tensor(0.0, device=pos.device, requires_grad=True)
        
        # Compute pairwise distances
        distances = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0)).squeeze(0)
        
        # Mask out self-distances
        mask = torch.eye(pos.size(0), device=pos.device, dtype=torch.bool)
        distances = distances.masked_fill(mask, float('inf'))
        
        # Find violations
        violations = torch.clamp(self.min_distance - distances, min=0)
        constraint_loss = torch.mean(violations) * self.constraint_weight
        
        return pos, constraint_loss  # For now, just return loss without position adjustment

class ConstrainedEGNNLayer(nn.Module):
    """EGNN Layer with integrated chemical constraints"""
    
    def __init__(self, hidden_dim: int, edge_dim: int = 1, 
                 constraints: Optional[Dict] = None, residual: bool = True, 
                 attention: bool = False):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.residual = residual
        self.attention = attention
        
        # Standard EGNN components
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + 3, hidden_dim),  # +1 for distance
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.1)
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.1)
        )
        
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        if attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        
        # Chemical constraints
        self.constraints = constraints or {}
        self.constraint_modules = nn.ModuleDict()
        
        if self.constraints.get('bond_length', False):
            self.constraint_modules['bond_length'] = BondLengthConstraint(
                constraint_weight=self.constraints.get('bond_length_weight', 0.1),
                soft_constraint=self.constraints.get('soft_constraints', True)
            )
            
        if self.constraints.get('valence', False):
            self.constraint_modules['valence'] = ValenceConstraint(
                constraint_weight=self.constraints.get('valence_weight', 0.05)
            )
            
        if self.constraints.get('steric_clash', False):
            self.constraint_modules['steric_clash'] = StericClashConstraint(
                min_distance=self.constraints.get('min_distance', 1.0),
                constraint_weight=self.constraints.get('steric_weight', 0.02)
            )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None, 
                atom_types: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Enhanced EGNN forward with constraints
        
        Returns dict with:
            - h: Updated node features
            - pos: Updated positions  
            - constraint_losses: Dict of constraint violation losses
        """
        if edge_index.size(1) == 0:
            return {
                'h': h,
                'pos': pos, 
                'constraint_losses': {}
            }
        
        row, col = edge_index
        
        # Compute edge features with distance
        radial = pos[row] - pos[col]
        radial_norm = torch.norm(radial, dim=-1, keepdim=True)
        radial_norm = torch.clamp(radial_norm, min=1e-8)
        
        # Prepare edge input
        edge_input = [h[row], h[col], radial_norm]
        if edge_attr is not None:
            edge_input.append(edge_attr)
        else:
            # Add dummy edge attributes
            dummy_edge_attr = torch.zeros(edge_index.size(1), 1, device=h.device)
            edge_input.append(dummy_edge_attr)
        
        edge_feat = torch.cat(edge_input, dim=-1)
        
        # Edge model
        m_ij = self.edge_mlp(edge_feat)
        
        # Attention (if enabled)
        if self.attention:
            att = self.att_mlp(m_ij)
            m_ij = m_ij * att
        
        # Coordinate update
        coord_diff = self.coord_mlp(m_ij)
        coord_diff = torch.tanh(coord_diff) * 0.1  # Limit coordinate changes
        radial_normalized = radial / radial_norm
        coord_update = coord_diff * radial_normalized
        
        # Apply coordinate updates
        pos_new = pos.clone()
        pos_new.index_add_(0, row, coord_update)
        
        # Node feature update
        agg = torch.zeros(h.size(0), self.hidden_dim, device=h.device)
        agg.index_add_(0, row, m_ij)
        
        h_input = torch.cat([h, agg], dim=-1)
        h_new = self.node_mlp(h_input)
        
        # Residual connection and normalization
        if self.residual:
            h_new = h + h_new
        h_new = self.layer_norm(h_new)
        
        # Apply constraints
        constraint_losses = {}
        
        # Bond length constraints
        if 'bond_length' in self.constraint_modules:
            bond_types = edge_attr[:, 0] if edge_attr is not None and edge_attr.size(1) > 0 else None
            pos_new, bond_loss = self.constraint_modules['bond_length'](
                pos_new, edge_index, bond_types
            )
            constraint_losses['bond_length'] = bond_loss
        
        # Valence constraints
        if 'valence' in self.constraint_modules:
            h_new, valence_loss = self.constraint_modules['valence'](
                h_new, edge_index, atom_types
            )
            constraint_losses['valence'] = valence_loss
        
        # Steric clash constraints
        if 'steric_clash' in self.constraint_modules:
            pos_new, steric_loss = self.constraint_modules['steric_clash'](
                pos_new, batch
            )
            constraint_losses['steric_clash'] = steric_loss
        
        return {
            'h': h_new,
            'pos': pos_new,
            'constraint_losses': constraint_losses
        }

class ConstrainedEGNNBackbone(nn.Module):
    """EGNN backbone with chemical constraints"""
    
    def __init__(self, hidden_dim: int = 256, num_layers: int = 3, 
                 cutoff: float = 10.0, constraints: Optional[Dict] = None,
                 use_attention: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cutoff = cutoff
        self.constraints = constraints or {}
        
        # Create constrained EGNN layers
        self.egnn_layers = nn.ModuleList([
            ConstrainedEGNNLayer(
                hidden_dim=hidden_dim,
                constraints=self.constraints,
                residual=True,
                attention=use_attention and (i > 0)  # Attention for deeper layers
            ) for i in range(num_layers)
        ])
        
        # Global constraint loss aggregation
        self.constraint_loss_weights = nn.ParameterDict({
            'bond_length': nn.Parameter(torch.tensor(self.constraints.get('bond_length_weight', 0.1))),
            'valence': nn.Parameter(torch.tensor(self.constraints.get('valence_weight', 0.05))),
            'steric_clash': nn.Parameter(torch.tensor(self.constraints.get('steric_weight', 0.02)))
        })
        
    def forward(self, h: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None,
                edge_attr: Optional[torch.Tensor] = None,
                atom_types: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through constrained EGNN layers"""
        
        # Build edge index if not provided
        if edge_index is None:
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=32)
        
        if edge_index.size(1) == 0:
            return {
                'h': h,
                'pos': pos,
                'constraint_losses': {},
                'total_constraint_loss': torch.tensor(0.0, device=h.device, requires_grad=True)
            }
        
        h_current = h
        pos_current = pos
        total_constraint_losses = {}
        
        # Pass through EGNN layers
        for layer_idx, egnn_layer in enumerate(self.egnn_layers):
            outputs = egnn_layer(
                h=h_current, 
                pos=pos_current, 
                edge_index=edge_index,
                edge_attr=edge_attr, 
                atom_types=atom_types,
                batch=batch
            )
            
            h_current = outputs['h']
            pos_current = outputs['pos']
            
            # Accumulate constraint losses
            for constraint_name, loss in outputs['constraint_losses'].items():
                if constraint_name not in total_constraint_losses:
                    total_constraint_losses[constraint_name] = []
                total_constraint_losses[constraint_name].append(loss)
        
        # Aggregate constraint losses
        aggregated_losses = {}
        total_constraint_loss = torch.tensor(0.0, device=h.device, requires_grad=True)
        
        for constraint_name, losses in total_constraint_losses.items():
            if losses:
                avg_loss = torch.stack(losses).mean()
                aggregated_losses[constraint_name] = avg_loss
                
                # Weight the loss
                if constraint_name in self.constraint_loss_weights:
                    weighted_loss = avg_loss * self.constraint_loss_weights[constraint_name]
                    total_constraint_loss = total_constraint_loss + weighted_loss
        
        return {
            'h': h_current,
            'pos': pos_current,
            'constraint_losses': aggregated_losses,
            'total_constraint_loss': total_constraint_loss
        }

def create_constrained_egnn_backbone(hidden_dim: int = 256, num_layers: int = 3, 
                                   cutoff: float = 10.0, constraints: Optional[Dict] = None):
    """Factory function to create constrained EGNN backbone"""
    return ConstrainedEGNNBackbone(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        cutoff=cutoff,
        constraints=constraints
    )

# Legacy compatibility
def create_egnn_backbone(hidden_dim=256, num_layers=3, cutoff=10.0):
    """Factory function to create standard EGNN backbone (backward compatibility)"""
    return ConstrainedEGNNBackbone(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        cutoff=cutoff,
        constraints={}  # No constraints
    )