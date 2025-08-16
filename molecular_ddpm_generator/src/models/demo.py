# src/models/cleaned_joint_2d_3d_model.py - Cleaned version keeping 2D+3D
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, radius_graph
from torch_geometric.utils import to_dense_batch
from .base_model import MolecularModel
from .egnn import CorrectedEGNNBackbone, create_corrected_egnn_backbone

try:
    from .pocket_encoder import create_improved_pocket_encoder, SimplePocketEncoder
    IMPROVED_POCKET_AVAILABLE = True
except ImportError:
    IMPROVED_POCKET_AVAILABLE = False

def safe_global_pool(x, batch, pool_type='mean'):
    """Safe global pooling with fallback"""
    try:
        if pool_type == 'mean':
            return global_mean_pool(x, batch)
        else:
            return global_max_pool(x, batch)
    except Exception:
        unique_batch = torch.unique(batch)
        pooled = []
        for b in unique_batch:
            mask = batch == b
            if pool_type == 'mean':
                pooled.append(torch.mean(x[mask], dim=0))
            else:
                pooled.append(torch.max(x[mask], dim=0)[0])
        return torch.stack(pooled)

class CleanedChemicalSpecialist2D(nn.Module):
    """Cleaned 2D chemical processing - removed redundant parts"""
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Basic chemical embeddings (KEPT)
        self.bond_type_embedding = nn.Embedding(5, 64)
        self.atom_type_embedding = nn.Embedding(11, 64)
        
        # Simple chemical GNN layers (CLEANED - removed constraints)
        self.chemical_gnn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            ) for _ in range(2)  # Reduced from 3 to 2 layers
        ])
        
        # Chemical properties (SIMPLIFIED - removed complex predictors)
        self.chemical_properties = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # REMOVED: valence_predictor, bond_type_classifier, ValenceConstraintLayer, ChemicalConstraintGNN
        
    def forward(self, x, edge_index, edge_attr, batch):
        # Extract basic chemical features
        atom_types = x[:, 0].long().clamp(0, 10)
        
        # Chemical embeddings
        atom_emb = self.atom_type_embedding(atom_types)
        
        # Simple feature combination
        h_chemical = torch.cat([
            atom_emb, 
            torch.zeros(atom_emb.size(0), self.hidden_dim - 64, device=atom_emb.device)
        ], dim=-1)
        
        # Apply simplified GNN layers
        for gnn_layer in self.chemical_gnn:
            h_prev = h_chemical
            h_chemical = gnn_layer(h_chemical)
            h_chemical = h_chemical + h_prev  # Residual connection
        
        # Chemical analysis (simplified)
        chemical_props = self.chemical_properties(h_chemical)
        
        return {
            'chemical_features': h_chemical,
            'chemical_properties': chemical_props,
            'atom_types': atom_types
        }

class CleanedPhysicalSpecialist3D(nn.Module):
    """Cleaned 3D physical processing - removed redundant analyzers"""
    
    def __init__(self, hidden_dim=256, cutoff=10.0, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        
        # EGNN backbone (KEPT - essential)
        self.egnn_backbone = create_corrected_egnn_backbone(
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            cutoff=cutoff,
            sin_embedding=True,
            reflection_equiv=True
        )
        
        # REMOVED: force_predictor, interaction_classifier, conformer_analyzer
        
        print(f"✅ Cleaned PhysicalSpecialist3D with EGNN")
        
    def forward(self, h, pos, batch, edge_index=None, edge_attr=None, atom_types=None):
        # EGNN processing (core functionality)
        egnn_outputs = self.egnn_backbone(
            h=h, pos=pos, batch=batch, 
            edge_index=edge_index, edge_attr=edge_attr,
            atom_types=atom_types
        )
        
        h_spatial = egnn_outputs['h']
        pos_final = egnn_outputs['pos']
        constraint_losses = egnn_outputs['constraint_losses']
        total_constraint_loss = egnn_outputs['total_constraint_loss']
        
        return {
            'spatial_features': h_spatial,
            'updated_positions': pos_final,
            'constraint_losses': constraint_losses,
            'total_constraint_loss': total_constraint_loss
        }

class CleanedComplementaryFusion(nn.Module):
    """Cleaned fusion - simplified consistency logic"""
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Cross-attention (KEPT - essential for 2D-3D fusion)
        self.chemical_to_physical = nn.MultiheadAttention(self.hidden_dim, num_heads=8, batch_first=True)
        self.physical_to_chemical = nn.MultiheadAttention(self.hidden_dim, num_heads=8, batch_first=True)
        
        # Feature fusion (SIMPLIFIED)
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + 32, self.hidden_dim),  # +32 for chemical props
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Simple consistency check (SIMPLIFIED)
        self.consistency_checker = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # REMOVED: consistency_enforcer, complex chemical-physical consistency logic
        
    def forward(self, chemical_output, physical_output):
        h_chem = chemical_output['chemical_features']
        h_phys = physical_output['spatial_features']
        
        # Cross-attention (requires 3D input for nn.MultiheadAttention)
        h_chem_3d = h_chem.unsqueeze(0)
        h_phys_3d = h_phys.unsqueeze(0)
        
        h_chem_attended, _ = self.chemical_to_physical(h_chem_3d, h_phys_3d, h_phys_3d)
        h_phys_attended, _ = self.physical_to_chemical(h_phys_3d, h_chem_3d, h_chem_3d)
        
        h_chem_attended = h_chem_attended.squeeze(0)
        h_phys_attended = h_phys_attended.squeeze(0)
        
        # Integrate additional features
        chemical_props = chemical_output['chemical_properties']
        
        # Fusion
        combined_features = torch.cat([
            h_chem_attended,
            h_phys_attended,
            chemical_props
        ], dim=-1)
        
        h_fused = self.fusion(combined_features)
        
        # Simple consistency check
        consistency_input = torch.cat([h_chem, h_phys], dim=-1)
        consistency_score = self.consistency_checker(consistency_input)
        
        return {
            'fused_features': h_fused,
            'consistency_score': consistency_score,
            'constraint_losses': physical_output['constraint_losses'],
            'total_constraint_loss': physical_output['total_constraint_loss']
        }

class CleanedJoint2D3DModel(MolecularModel):
    """Cleaned Joint 2D-3D Model - removed redundant components but kept 2D+3D processing"""
    
    def __init__(self, atom_types=11, bond_types=4, hidden_dim=256, 
                 pocket_dim=256, num_layers=6, max_radius=10.0,
                 max_pocket_atoms=1000, conditioning_type="add"):
        super().__init__(atom_types, bond_types, hidden_dim)
        
        self.hidden_dim = hidden_dim
        self.pocket_dim = pocket_dim
        self.conditioning_type = conditioning_type
        
        # Atom embedding
        self.atom_embedding = nn.Linear(6, self.hidden_dim)
        
        # Cleaned specialized processing branches
        self.chemical_2d = CleanedChemicalSpecialist2D(self.hidden_dim)
        self.physical_3d = CleanedPhysicalSpecialist3D(
            self.hidden_dim, 
            cutoff=max_radius,
            num_layers=num_layers
        )
        
        # Cleaned fusion
        self.fusion = CleanedComplementaryFusion(self.hidden_dim)
        
        # Pocket encoder (KEPT AS REQUESTED)
        if IMPROVED_POCKET_AVAILABLE:
            self.pocket_encoder = create_improved_pocket_encoder(
                hidden_dim=self.hidden_dim,
                output_dim=pocket_dim
            )
        else:
            self.pocket_encoder = SimplePocketEncoder(
                input_dim=7,
                hidden_dim=self.hidden_dim,
                output_dim=pocket_dim,
                max_atoms=max_pocket_atoms
            )
        
        # Conditioning
        if conditioning_type == "add":
            assert pocket_dim == self.hidden_dim
            self.condition_transform = nn.Identity()
        else:
            self.condition_transform = nn.Linear(pocket_dim, self.hidden_dim)
        
        # Output head
        self.position_head = nn.Linear(self.hidden_dim, 3)
        
        print(f"✅ Cleaned Joint2D3D Model created:")
        print(f"   Hidden dim: {hidden_dim}, Layers: {num_layers}")
        print(f"   Kept: 2D chemical + 3D physical + cross-attention fusion")
        print(f"   Removed: redundant predictors, complex constraints")
            
    def forward(self, x, pos, edge_index, edge_attr, batch,
                pocket_x=None, pocket_pos=None, pocket_edge_index=None, 
                pocket_batch=None, **kwargs):
        
        # Initial embedding
        h_init = self._embed_atoms_flexible(x)
        
        # 2D chemical processing (cleaned)
        chemical_output = self.chemical_2d(x, edge_index, edge_attr, batch)
        
        # 3D physical processing (cleaned)
        physical_output = self.physical_3d(
            h_init, pos, batch, edge_index, edge_attr, 
            atom_types=chemical_output.get('atom_types')
        )
        
        # 2D-3D fusion (cleaned)
        fusion_output = self.fusion(chemical_output, physical_output)
        
        # Pocket conditioning (kept as original)
        pocket_condition = self._encode_pocket(
            pocket_x, pocket_pos, pocket_edge_index, pocket_batch, batch
        )
        
        h_final = fusion_output['fused_features']
        if pocket_condition is not None:
            h_final = self._apply_conditioning(h_final, pocket_condition, batch)
        
        # Position prediction
        pos_pred = physical_output['updated_positions'] + self.position_head(h_final)
        
        return {
            'pos_pred': pos_pred,
            'node_features': h_final,
            'chemical_properties': chemical_output['chemical_properties'],
            'consistency_score': fusion_output['consistency_score'],
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
        """Encode pocket using existing pocket_encoder.py"""
        if pocket_x is None or pocket_pos is None:
            return None
        
        try:
            pocket_repr = self.pocket_encoder(
                pocket_x, pocket_pos, pocket_edge_index, pocket_batch
            )
            return pocket_repr
        except Exception:
            batch_size = ligand_batch.max().item() + 1
            return torch.zeros(batch_size, self.pocket_dim, device=ligand_batch.device)
    
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
def create_cleaned_joint2d3d_model(hidden_dim=256, num_layers=6, conditioning_type="add", **kwargs):
    """Create cleaned joint2d3d model"""
    return CleanedJoint2D3DModel(
        atom_types=11,
        bond_types=4,
        hidden_dim=hidden_dim,
        pocket_dim=hidden_dim,
        num_layers=num_layers,
        conditioning_type=conditioning_type,
        **kwargs
    )