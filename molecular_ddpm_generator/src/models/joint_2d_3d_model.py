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
        
        # Features: [atom_type, degree, formal_charge, is_aromatic, total_Hs, is_in_ring]
        self.atom_feature_embedding = nn.Linear(6, hidden_dim)  # Use all 6 features directly
        
        # Features: [bond_type, is_conjugated, is_in_ring]
        self.bond_feature_embedding = nn.Linear(3, 32)
        
        self.register_buffer('atom_type_mapping', torch.tensor([
            6,   # 0: C  → atomic number 6
            7,   # 1: N  → atomic number 7
            8,   # 2: O  → atomic number 8
            16,  # 3: S  → atomic number 16
            9,   # 4: F  → atomic number 9
            17,  # 5: Cl → atomic number 17
            35,  # 6: Br → atomic number 35
            53,  # 7: I  → atomic number 53
            15,  # 8: P  → atomic number 15
            1,   # 9: H  → atomic number 1 (if used)
            6    # 10: UNK → default to Carbon (6)
        ]))
        
        # Chemical graph convolutions with bond awareness
        self.chemical_convs = nn.ModuleList([
            ChemicalConvWithBonds(hidden_dim, 32) for _ in range(3)
        ])
        
        # Chemical pattern analysis
        self.chemical_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64)
        )
        
    def forward(self, x, edge_index, edge_attr, batch):
        
        # x shape: [N, 6] - [atom_type, degree, formal_charge, is_aromatic, total_Hs, is_in_ring]
        h_chem = self.atom_feature_embedding(x.float())
        
        atom_type_indices = x[:, 0].long().clamp(0, 10)  # Preprocessing indices [0-10]
        atom_types_atomic = self.atom_type_mapping[atom_type_indices]  # → Atomic numbers for EGNN
        
        # Use ALL 3 bond features from preprocessing if available
        bond_features = None
        if edge_attr is not None and edge_attr.size(1) >= 3:
            # edge_attr shape: [E, 3] - [bond_type, is_conjugated, is_in_ring]
            bond_features = self.bond_feature_embedding(edge_attr.float())
        
        # Chemical graph convolutions with bond awareness
        for conv in self.chemical_convs:
            h_prev = h_chem
            h_chem = conv(h_chem, edge_index, bond_features)
            h_chem = h_chem + h_prev  # Residual connection
        
        # Chemical pattern analysis
        chemical_patterns = self.chemical_analyzer(h_chem)
        
        # Extract individual atom properties for interpretation
        degrees = x[:, 1]            # Second feature is degree
        formal_charges = x[:, 2]     # Third feature is formal_charge  
        is_aromatic = x[:, 3]        # Fourth feature is aromatic
        total_hs = x[:, 4]           # Fifth feature is total Hs
        is_in_ring = x[:, 5]         # Sixth feature is in_ring
        
        return {
            'chemical_features': h_chem,
            'chemical_patterns': chemical_patterns,
            'atom_types': atom_types_atomic,      # Atomic numbers for EGNN constraints
            'atom_type_indices': atom_type_indices,  # Original preprocessing indices
            'degrees': degrees,
            'formal_charges': formal_charges,
            'is_aromatic': is_aromatic,
            'total_hs': total_hs,
            'is_in_ring': is_in_ring
        }

class ChemicalConvWithBonds(nn.Module):
    """Chemical graph convolution that uses bond features"""
    
    def __init__(self, hidden_dim, bond_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bond_dim = bond_dim
        
        # Message function with bond information
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + bond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, h, edge_index, bond_features=None):
        if edge_index.size(1) == 0:
            return h
        
        row, col = edge_index
        
        # Messages with bond information
        if bond_features is not None:
            messages = torch.cat([h[row], h[col], bond_features], dim=-1)
        else:
            # Use zero bond features if not available
            zero_bonds = torch.zeros(edge_index.size(1), self.bond_dim, device=h.device)
            messages = torch.cat([h[row], h[col], zero_bonds], dim=-1)
        
        messages = self.message_mlp(messages)
        
        # Aggregate messages
        h_out = torch.zeros_like(h)
        h_out.index_add_(0, row, messages)
        
        return h_out

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
            enable_chemical_constraints=True  # ALL constraints here
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
    """Smart 2D-3D Fusion using full feature information"""
    
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
    
        # Feature fusion using all chemical patterns and geometric features
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64 + 64, hidden_dim),  # chem + spatial + patterns + geometric
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Complementarity scorer using rich features
        self.complementarity = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, chemical_output, physical_output):
        """Fusion using all available chemical and spatial features"""
        
        h_chem = chemical_output['chemical_features']
        h_spatial = physical_output['spatial_features']
        
        # Cross-attention
        h_chem_3d = h_chem.unsqueeze(0)
        h_spatial_3d = h_spatial.unsqueeze(0)
        
        # Chemistry informed by spatial structure
        h_chem_attended, _ = self.chem_to_spatial(h_chem_3d, h_spatial_3d, h_spatial_3d)
        h_chem_attended = h_chem_attended.squeeze(0)
        
        # Spatial structure informed by chemistry
        h_spatial_attended, _ = self.spatial_to_chem(h_spatial_3d, h_chem_3d, h_chem_3d)
        h_spatial_attended = h_spatial_attended.squeeze(0)
        
        # Feature combination using all patterns
        combined_features = torch.cat([
            h_chem_attended,
            h_spatial_attended,
            chemical_output['chemical_patterns'],
            physical_output['geometric_features']
        ], dim=-1)
        
        fused_features = self.feature_fusion(combined_features)
        
        # Complementarity scoring
        complementarity_score = self.complementarity(
            torch.cat([h_chem, h_spatial], dim=-1)
        )
        
        return {
            'fused_features': fused_features,
            'complementarity_score': complementarity_score,
            'constraint_losses': physical_output['constraint_losses'],
            'total_constraint_loss': physical_output['total_constraint_loss']
        }

class Joint2D3DModel(MolecularModel):
    """Joint 2D-3D Model utilizing ALL preprocessing features"""
    
    def __init__(self, atom_types=11, bond_types=4, hidden_dim=256,
                 pocket_dim=256, num_layers=4, max_radius=10.0,
                 conditioning_type="add"):
        super().__init__(atom_types, bond_types, hidden_dim)
        
        self.hidden_dim = hidden_dim
        self.conditioning_type = conditioning_type
        
        # 2D Chemistry Branch using full preprocessing features
        self.chemistry_2d = Chemical2DBranch(hidden_dim)
        
        # 3D Physics Branch (uses 3D positions from preprocessing)
        self.physics_3d = Physical3DBranch(
            hidden_dim=hidden_dim,
            cutoff=max_radius,
            num_layers=num_layers
        )
        
        # Smart Fusion
        self.fusion = SmartFusion(hidden_dim)
        
        # Pocket conditioning (uses 7 residue features from preprocessing)
        if IMPROVED_POCKET_AVAILABLE:
            self.pocket_encoder = create_improved_pocket_encoder(
                hidden_dim=hidden_dim,
                output_dim=pocket_dim
            )
        else:
            self.pocket_encoder = SimplePocketEncoder(
                input_dim=7,  # 7 residue features from preprocessing
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
        
        # 🧪 2D Chemistry Processing using ALL atom and bond features
        chemistry_output = self.chemistry_2d(x, edge_index, edge_attr, batch)
        
        # Convert to hidden representation for 3D processing
        h_init = chemistry_output['chemical_features']
        
        # 3D Physics Processing using 3D positions
        physics_output = self.physics_3d(
            h_init, pos, batch, edge_index, edge_attr,
            atom_types=chemistry_output['atom_types']
        )
        
        # Smart Fusion
        fusion_output = self.fusion(chemistry_output, physics_output)
        
        # Pocket conditioning using ALL 7 residue features
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
            
            # Full chemistry outputs from ALL preprocessing features
            'chemical_patterns': chemistry_output['chemical_patterns'],
            'atom_types': chemistry_output['atom_types'],
            'degrees': chemistry_output['degrees'],
            'formal_charges': chemistry_output['formal_charges'],
            'is_aromatic': chemistry_output['is_aromatic'],
            'total_hs': chemistry_output['total_hs'],
            'is_in_ring': chemistry_output['is_in_ring'],
            
            # Physics outputs
            'geometric_features': physics_output['geometric_features'],
            'invariant_features': physics_output['invariant_features'],
            
            # Fusion outputs
            'complementarity_score': fusion_output['complementarity_score'],
            
            # Constraints (only from 3D branch)
            'constraint_losses': fusion_output['constraint_losses'],
            'total_constraint_loss': fusion_output['total_constraint_loss']
        }
    
    def _encode_pocket(self, pocket_x, pocket_pos, pocket_edge_index, 
                      pocket_batch, ligand_batch):
        """Encode protein pocket using ALL 7 residue features"""
        if pocket_x is None or pocket_pos is None:
            return None
        
        try:
            # pocket_x shape should be [M, 7] from preprocessing
            # Features: [res_type, res_id, hydrophobic, charged, polar, aromatic, bfactor]
            pocket_repr = self.pocket_encoder(
                pocket_x, pocket_pos, pocket_edge_index, pocket_batch
            )
            return pocket_repr
        except Exception as e:
            print(f"Warning: Pocket encoding failed: {e}")
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
def joint2d3d_model(hidden_dim=256, num_layers=4, conditioning_type="add", **kwargs):
    """Create joint2d3d model using ALL preprocessing features"""
    return Joint2D3DModel(
        atom_types=11,
        bond_types=4,
        hidden_dim=hidden_dim,
        pocket_dim=hidden_dim,
        num_layers=num_layers,
        conditioning_type=conditioning_type,
        **kwargs
    )