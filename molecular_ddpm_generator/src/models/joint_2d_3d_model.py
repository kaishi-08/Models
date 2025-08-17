# src/models/optimized_joint_2d_3d_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from .base_model import MolecularModel
from .egnn import EGNNBackbone

try:
    from .pocket_encoder import create_improved_pocket_encoder, SimplePocketEncoder
    IMPROVED_POCKET_AVAILABLE = True
except ImportError:
    IMPROVED_POCKET_AVAILABLE = False

class Chemical2DBranch(nn.Module):    
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Chemical feature extractors
        self.atom_type_embedding = nn.Embedding(11, 64)
        self.bond_type_embedding = nn.Embedding(5, 32)
        
        # 2D Graph convolutions for chemical patterns
        self.chemical_convs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(3)
        ])
        
        # Chemical pattern recognition
        self.pattern_detector = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64)
        )
        
        self.property_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32)
        )
        
    def forward(self, x, edge_index, edge_attr, batch):
        
        # Extract atom types for embedding
        atom_types = x[:, 0].long().clamp(0, 10)
        atom_emb = self.atom_type_embedding(atom_types)
        
        # Initial chemical features
        h_chem = torch.cat([
            atom_emb,
            torch.zeros(atom_emb.size(0), self.hidden_dim - 64, device=atom_emb.device)
        ], dim=-1)
        
        # Chemical convolutions (focus on connectivity patterns)
        for conv in self.chemical_convs:
            h_prev = h_chem
            h_chem = conv(h_chem)
            h_chem = h_chem + h_prev  # Residual
        
        # Chemical analysis
        chemical_patterns = self.pattern_detector(h_chem)
        molecular_properties = self.property_predictor(h_chem)
        
        return {
            'chemical_features': h_chem,
            'chemical_patterns': chemical_patterns,
            'molecular_properties': molecular_properties,
            'atom_types': atom_types
        }

class Physical3DBranch(nn.Module):    
    def __init__(self, hidden_dim=256, cutoff=10.0, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        
        # EGNN backbone với ALL constraints
        self.egnn_backbone = EGNNBackbone(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            cutoff=cutoff,
            sin_embedding=True,
            reflection_equiv=True,
            enable_chemical_constraints=True  # 🎯 ALL constraints here
        )
        
        # Physics-based analyzers
        self.geometric_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64)
        )
        
        # SE(3) invariant features
        self.invariant_extractor = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),  # +3 for position info
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, h, pos, batch, edge_index=None, edge_attr=None, atom_types=None):
        """SE(3) equivariant processing with all constraints"""
        
        # EGNN processing with constraints
        egnn_outputs = self.egnn_backbone(
            h=h, pos=pos, batch=batch,
            edge_index=edge_index, edge_attr=edge_attr,
            atom_types=atom_types
        )
        
        h_spatial = egnn_outputs['h']
        pos_updated = egnn_outputs['pos']
        
        # Physical analysis
        geometric_features = self.geometric_analyzer(h_spatial)
        
        # SE(3) invariant combination
        pos_info = torch.cat([
            torch.norm(pos_updated, dim=-1, keepdim=True),  # Distance from origin
            pos_updated.mean(dim=0, keepdim=True).expand_as(pos_updated)  # Center of mass
        ], dim=-1)
        
        invariant_features = self.invariant_extractor(
            torch.cat([h_spatial, pos_info], dim=-1)
        )
        
        return {
            'spatial_features': h_spatial,
            'updated_positions': pos_updated,
            'geometric_features': geometric_features,
            'invariant_features': invariant_features,
            'constraint_losses': egnn_outputs['constraint_losses'],
            'total_constraint_loss': egnn_outputs['total_constraint_loss']
        }

class SmartFusion(nn.Module):
    """🔗 Smart 2D-3D Fusion without duplicate constraints"""
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Cross-modal attention
        self.chem_to_spatial = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )
        self.spatial_to_chem = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64 + 64, hidden_dim),  # chem + spatial + patterns + geometric
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Complementarity scorer
        self.complementarity = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, chemical_output, physical_output):
        """Fuse 2D chemistry with 3D physics"""
        
        h_chem = chemical_output['chemical_features']
        h_spatial = physical_output['spatial_features']
        
        # Cross-attention (batch dimension for attention)
        h_chem_3d = h_chem.unsqueeze(0)
        h_spatial_3d = h_spatial.unsqueeze(0)
        
        # Chemistry informed by spatial structure
        h_chem_attended, _ = self.chem_to_spatial(h_chem_3d, h_spatial_3d, h_spatial_3d)
        h_chem_attended = h_chem_attended.squeeze(0)
        
        # Spatial structure informed by chemistry
        h_spatial_attended, _ = self.spatial_to_chem(h_spatial_3d, h_chem_3d, h_chem_3d)
        h_spatial_attended = h_spatial_attended.squeeze(0)
        
        # Feature combination
        combined_features = torch.cat([
            h_chem_attended,
            h_spatial_attended,
            chemical_output['chemical_patterns'],
            physical_output['geometric_features']
        ], dim=-1)
        
        fused_features = self.feature_fusion(combined_features)
        
        # Measure complementarity
        complementarity_score = self.complementarity(
            torch.cat([h_chem, h_spatial], dim=-1)
        )
        
        return {
            'fused_features': fused_features,
            'complementarity_score': complementarity_score,
            'constraint_losses': physical_output['constraint_losses'],
            'total_constraint_loss': physical_output['total_constraint_loss']
        }

class OptimizedJoint2D3DModel(MolecularModel):
    """🎯 Optimized Joint 2D-3D Model: Clean separation + Smart fusion"""
    
    def __init__(self, atom_types=11, bond_types=4, hidden_dim=256,
                 pocket_dim=256, num_layers=4, max_radius=10.0,
                 conditioning_type="add"):
        super().__init__(atom_types, bond_types, hidden_dim)
        
        self.hidden_dim = hidden_dim
        self.conditioning_type = conditioning_type
        
        print(f"🎯 Creating Optimized Joint2D3D Model:")
        print(f"   - Clean 2D Chemistry Branch (no constraints)")
        print(f"   - 3D Physics Branch with ALL constraints")
        print(f"   - Smart fusion without duplication")
        
        # Input embedding
        self.atom_embedding = nn.Linear(6, hidden_dim)
        
        # 🧪 2D Chemistry Branch (pure chemical patterns)
        self.chemistry_2d = Chemical2DBranch(hidden_dim)
        
        # ⚛️ 3D Physics Branch (SE(3) + ALL constraints)
        self.physics_3d = Physical3DBranch(
            hidden_dim=hidden_dim,
            cutoff=max_radius,
            num_layers=num_layers
        )
        
        # 🔗 Smart Fusion
        self.fusion = SmartFusion(hidden_dim)
        
        # Pocket conditioning
        if IMPROVED_POCKET_AVAILABLE:
            self.pocket_encoder = create_improved_pocket_encoder(
                hidden_dim=hidden_dim,
                output_dim=pocket_dim
            )
        else:
            self.pocket_encoder = SimplePocketEncoder(
                input_dim=7,
                hidden_dim=hidden_dim,
                output_dim=pocket_dim
            )
        
        # Conditioning transform
        if conditioning_type == "add":
            assert pocket_dim == hidden_dim
            self.condition_transform = nn.Identity()
        else:
            self.condition_transform = nn.Linear(pocket_dim, hidden_dim)
        
        # Output heads
        self.position_head = nn.Linear(hidden_dim, 3)
        self.atom_head = nn.Linear(hidden_dim, atom_types)
        self.bond_head = nn.Linear(hidden_dim * 2, bond_types)
        
    def forward(self, x, pos, edge_index, edge_attr, batch,
                pocket_x=None, pocket_pos=None, pocket_edge_index=None,
                pocket_batch=None, **kwargs):
        
        # Initial embedding
        h_init = self._embed_atoms_flexible(x)
        
        # 🧪 2D Chemistry Processing (pure chemical patterns)
        chemistry_output = self.chemistry_2d(x, edge_index, edge_attr, batch)
        
        # ⚛️ 3D Physics Processing (SE(3) + constraints)
        physics_output = self.physics_3d(
            h_init, pos, batch, edge_index, edge_attr,
            atom_types=chemistry_output['atom_types']
        )
        
        # 🔗 Smart Fusion
        fusion_output = self.fusion(chemistry_output, physics_output)
        
        # Pocket conditioning
        pocket_condition = self._encode_pocket(
            pocket_x, pocket_pos, pocket_edge_index, pocket_batch, batch
        )
        
        h_final = fusion_output['fused_features']
        if pocket_condition is not None:
            h_final = self._apply_conditioning(h_final, pocket_condition, batch)
        
        # Predictions
        pos_pred = physics_output['updated_positions'] + self.position_head(h_final)
        atom_pred = self.atom_head(h_final)
        
        # Bond predictions (if edges exist)
        if edge_index.size(1) > 0:
            row, col = edge_index
            bond_features = torch.cat([h_final[row], h_final[col]], dim=-1)
            bond_pred = self.bond_head(bond_features)
        else:
            bond_pred = torch.zeros(0, self.bond_types, device=h_final.device)
        
        return {
            'pos_pred': pos_pred,
            'atom_pred': atom_pred,
            'bond_pred': bond_pred,
            'node_features': h_final,
            
            # Chemistry branch outputs
            'chemical_patterns': chemistry_output['chemical_patterns'],
            'molecular_properties': chemistry_output['molecular_properties'],
            
            # Physics branch outputs (with constraints)
            'geometric_features': physics_output['geometric_features'],
            'invariant_features': physics_output['invariant_features'],
            
            # Fusion outputs
            'complementarity_score': fusion_output['complementarity_score'],
            
            # Constraints (only from 3D branch)
            'constraint_losses': fusion_output['constraint_losses'],
            'total_constraint_loss': fusion_output['total_constraint_loss']
        }
    
    def _embed_atoms_flexible(self, x):
        """Flexible atom embedding"""
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got {x.dim()}D")
        
        input_dim = x.size(1)
        if input_dim < 6:
            padding = torch.zeros(x.size(0), 6 - input_dim, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
            return self.atom_embedding(x_padded.float())
        elif input_dim > 6:
            x_truncated = x[:, :6]
            return self.atom_embedding(x_truncated.float())
        else:
            return self.atom_embedding(x.float())
    
    def _encode_pocket(self, pocket_x, pocket_pos, pocket_edge_index, 
                      pocket_batch, ligand_batch):
        """Encode protein pocket"""
        if pocket_x is None or pocket_pos is None:
            return None
        
        try:
            pocket_repr = self.pocket_encoder(
                pocket_x, pocket_pos, pocket_edge_index, pocket_batch
            )
            return pocket_repr
        except Exception:
            batch_size = ligand_batch.max().item() + 1
            return torch.zeros(batch_size, self.hidden_dim, device=ligand_batch.device)
    
    def _apply_conditioning(self, atom_features, pocket_condition, batch):
        """Apply pocket conditioning"""
        if pocket_condition is None:
            return atom_features
        
        pocket_transformed = self.condition_transform(pocket_condition)
        
        # Handle batch size mismatch
        batch_size = pocket_transformed.size(0)
        max_batch_idx = batch.max().item()
        
        if max_batch_idx >= batch_size:
            last_condition = pocket_transformed[-1:].expand(max_batch_idx + 1 - batch_size, -1)
            pocket_transformed = torch.cat([pocket_transformed, last_condition], dim=0)
        
        batch_safe = torch.clamp(batch, 0, pocket_transformed.size(0) - 1)
        broadcasted_condition = pocket_transformed[batch_safe]
        
        return atom_features + broadcasted_condition

# Factory function
def create_optimized_joint2d3d_model(hidden_dim=256, num_layers=4, conditioning_type="add", **kwargs):
    """Create optimized joint2d3d model with clean separation"""
    return OptimizedJoint2D3DModel(
        atom_types=11,
        bond_types=4,
        hidden_dim=hidden_dim,
        pocket_dim=hidden_dim,
        num_layers=num_layers,
        conditioning_type=conditioning_type,
        **kwargs
    )