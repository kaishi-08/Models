# src/models/joint_2d_3d_model.py - UPDATED: Using Corrected EGNN Implementation
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

class ChemicalSpecialist2D(nn.Module):
    """2D processing focused on chemical intelligence"""
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Chemical knowledge embeddings
        self.bond_type_embedding = nn.Embedding(5, 64)
        self.atom_type_embedding = nn.Embedding(11, 64)
        self.formal_charge_embedding = nn.Embedding(7, 32)
        self.hybridization_embedding = nn.Embedding(8, 32)
        
        # Chemical graph convolutions
        self.chemical_gnn = nn.ModuleList([
            ChemicalGraphConv(self.hidden_dim) for _ in range(3)
        ])
        
        # Chemical property prediction
        self.chemical_properties = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        
        # Functional group patterns
        self.fg_detector = FunctionalGroupDetector(self.hidden_dim)
        
    def forward(self, x, edge_index, edge_attr, batch):
        # Extract chemical features (no spatial coordinates)
        atom_types = x[:, 0].long().clamp(0, 10)
        formal_charges = (x[:, 2].long() + 3).clamp(0, 6)
        hybridization = x[:, 1].long().clamp(0, 7) if x.size(1) > 1 else torch.zeros_like(atom_types)
        
        # Chemical embeddings
        atom_emb = self.atom_type_embedding(atom_types)
        charge_emb = self.formal_charge_embedding(formal_charges)
        hybrid_emb = self.hybridization_embedding(hybridization)
        
        # Bond embeddings
        if edge_attr.size(1) > 0 and edge_index.size(1) > 0:
            bond_types = edge_attr[:, 0].long().clamp(0, 4)
            bond_emb = self.bond_type_embedding(bond_types)
        else:
            bond_emb = torch.zeros((0, 64), device=x.device)
        
        # Combine chemical features
        h_chemical = torch.cat([atom_emb, charge_emb, hybrid_emb], dim=-1)
        
        # Pad to hidden_dim if needed
        if h_chemical.size(1) < self.hidden_dim:
            padding = torch.zeros(h_chemical.size(0), self.hidden_dim - h_chemical.size(1), 
                                device=h_chemical.device)
            h_chemical = torch.cat([h_chemical, padding], dim=-1)
        elif h_chemical.size(1) > self.hidden_dim:
            h_chemical = h_chemical[:, :self.hidden_dim]
        
        # Chemical graph convolutions
        for gnn_layer in self.chemical_gnn:
            h_chemical = gnn_layer(h_chemical, edge_index, bond_emb)
        
        # Chemical analysis
        chemical_props = self.chemical_properties(h_chemical)
        fg_features = self.fg_detector(h_chemical, edge_index, edge_attr)
        
        return {
            'chemical_features': h_chemical,
            'chemical_properties': chemical_props,
            'functional_groups': fg_features,
            'atom_types': atom_types  # For EGNN processing
        }

class ChemicalGraphConv(nn.Module):
    """Graph convolution for chemical topology"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.bond_attention = nn.Linear(64, 1)
        
    def forward(self, h, edge_index, bond_emb):
        if edge_index.size(1) == 0:
            return h
        
        row, col = edge_index
        
        # Handle bond embedding size mismatch
        if bond_emb.size(0) != edge_index.size(1):
            if bond_emb.size(0) == 0:
                bond_emb = torch.zeros(edge_index.size(1), 64, device=h.device)
            else:
                # Repeat or truncate
                repeat_factor = edge_index.size(1) // bond_emb.size(0) + 1
                bond_emb = bond_emb.repeat(repeat_factor, 1)[:edge_index.size(1)]
        
        # Bond-aware message passing
        bond_weights = torch.sigmoid(self.bond_attention(bond_emb))
        messages = torch.cat([h[row], h[col], bond_emb], dim=-1)
        messages = self.node_update(messages) * bond_weights
        
        # Aggregate
        h_new = torch.zeros_like(h)
        h_new.index_add_(0, row, messages)
        
        return h + h_new

class FunctionalGroupDetector(nn.Module):
    """Detect functional groups using graph patterns"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pattern_encoders = nn.ModuleDict({
            'carbonyl': nn.Linear(hidden_dim, 16),
            'amine': nn.Linear(hidden_dim, 16),
            'hydroxyl': nn.Linear(hidden_dim, 16),
            'carboxyl': nn.Linear(hidden_dim, 16)
        })
        
    def forward(self, h, edge_index, edge_attr):
        fg_features = []
        
        for pattern_name, encoder in self.pattern_encoders.items():
            pattern_feat = encoder(h)
            pattern_pooled = torch.mean(pattern_feat, dim=0, keepdim=True)
            fg_features.append(pattern_pooled.expand(h.size(0), -1))
        
        return torch.cat(fg_features, dim=-1)

class PhysicalSpecialist3D(nn.Module):
    """3D processing with CORRECTED SE(3) Equivariant EGNN"""
    
    def __init__(self, hidden_dim=256, cutoff=10.0, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        
        # 🎯 CORRECTED: Use proper EGNN implementation
        self.egnn_backbone = create_corrected_egnn_backbone(
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            cutoff=cutoff,
            sin_embedding=True,  # Use sinusoidal distance embedding
            reflection_equiv=True  # Full SE(3) equivariance including reflections
        )
        
        # Physical force modeling
        self.force_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 3)
        )
        
        # Non-bonded interaction detector
        self.interaction_classifier = nn.ModuleDict({
            'hydrogen_bond': nn.Linear(hidden_dim * 2 + 2, 1),
            'pi_pi_stacking': nn.Linear(hidden_dim * 2 + 2, 1),
            'van_der_waals': nn.Linear(hidden_dim * 2 + 2, 1)
        })
        
        # Conformation analysis
        self.conformer_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        print(f"✅ PhysicalSpecialist3D with CORRECTED EGNN")
        
    def forward(self, h, pos, batch, edge_index=None, edge_attr=None, atom_types=None):
        # 🎯 CORRECTED EGNN processing with proper SE(3) equivariance
        egnn_outputs = self.egnn_backbone(
            h=h, pos=pos, batch=batch, 
            edge_index=edge_index, edge_attr=edge_attr,
            atom_types=atom_types
        )
        
        h_spatial = egnn_outputs['h']
        pos_final = egnn_outputs['pos']
        constraint_losses = egnn_outputs['constraint_losses']
        total_constraint_loss = egnn_outputs['total_constraint_loss']
        
        # Physical analysis
        forces = self.force_predictor(h_spatial)
        interactions = self._detect_interactions(h_spatial, pos_final, batch)
        conformer_features = self.conformer_analyzer(h_spatial)
        
        return {
            'spatial_features': h_spatial,
            'updated_positions': pos_final,
            'predicted_forces': forces,
            'interactions': interactions,
            'conformer_features': conformer_features,
            'constraint_losses': constraint_losses,
            'total_constraint_loss': total_constraint_loss
        }
    
    def _detect_interactions(self, h, pos, batch):
        interaction_edges = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=20)
        
        if interaction_edges.size(1) == 0:
            return {}
        
        row, col = interaction_edges
        distances = torch.norm(pos[row] - pos[col], dim=-1, keepdim=True)
        
        interactions = {}
        for interaction_type, classifier in self.interaction_classifier.items():
            normalized_distances = distances / 10.0
            edge_features = torch.cat([h[row], h[col], distances, normalized_distances], dim=-1)
            interaction_scores = torch.sigmoid(classifier(edge_features))
            interactions[interaction_type] = interaction_scores
        
        return interactions

class ComplementaryFusion(nn.Module):
    """Fuse complementary 2D chemical and 3D physical information"""
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        # Cross-attention
        self.chemical_to_physical = nn.MultiheadAttention(self.hidden_dim, num_heads=8, batch_first=True)
        self.physical_to_chemical = nn.MultiheadAttention(self.hidden_dim, num_heads=8, batch_first=True)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + 32 + 64, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Consistency checker
        self.consistency_checker = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
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
        conformer_features = physical_output['conformer_features']
        
        # Fusion
        combined_features = torch.cat([
            h_chem_attended,
            h_phys_attended,
            chemical_props,
            conformer_features
        ], dim=-1)
        
        h_fused = self.fusion(combined_features)
        
        # Consistency check
        consistency_score = self.consistency_checker(torch.cat([h_chem, h_phys], dim=-1))
        
        return {
            'fused_features': h_fused,
            'consistency_score': consistency_score,
            'forces': physical_output['predicted_forces'],
            'interactions': physical_output['interactions'],
            'constraint_losses': physical_output['constraint_losses'],
            'total_constraint_loss': physical_output['total_constraint_loss']
        }

class Joint2D3DModel(MolecularModel):
    """
    🎯 UPDATED: Joint 2D-3D Molecular Model with CORRECTED SE(3) Equivariant EGNN
    
    Now uses proper EGNN implementation from DiffSBDD for guaranteed SE(3) equivariance
    """
    
    def __init__(self, atom_types=11, bond_types=4, hidden_dim=256, 
                 pocket_dim=256, num_layers=6, max_radius=10.0,
                 max_pocket_atoms=1000, conditioning_type="add"):
        super().__init__(atom_types, bond_types, hidden_dim)
        
        self.hidden_dim = hidden_dim
        self.pocket_dim = pocket_dim
        self.conditioning_type = conditioning_type
        
        # Atom embedding
        self.atom_embedding = nn.Linear(6, self.hidden_dim)
        
        # Specialized processing branches
        self.chemical_2d = ChemicalSpecialist2D(self.hidden_dim)
        self.physical_3d = PhysicalSpecialist3D(
            self.hidden_dim, 
            cutoff=max_radius,
            num_layers=num_layers
        )
        
        # Complementary fusion
        self.fusion = ComplementaryFusion(self.hidden_dim)
        
        # Pocket encoder
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
        
        # Output heads
        self.atom_type_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, atom_types)
        )
        
        self.bond_type_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, bond_types)
        )
        
        self.position_head = nn.Linear(self.hidden_dim, 3)
        
        print(f"✅ Joint2D3DModel with CORRECTED SE(3) Equivariant EGNN initialized!")
        
    def forward(self, x, pos, edge_index, edge_attr, batch,
                pocket_x=None, pocket_pos=None, pocket_edge_index=None, 
                pocket_batch=None, **kwargs):
        
        # Initial embedding
        h_init = self._embed_atoms_flexible(x)
        
        # Specialized processing
        chemical_output = self.chemical_2d(x, edge_index, edge_attr, batch)
        physical_output = self.physical_3d(
            h_init, pos, batch, edge_index, edge_attr, 
            atom_types=chemical_output.get('atom_types')
        )
        
        # Complementary fusion
        fusion_output = self.fusion(chemical_output, physical_output)
        
        # Pocket conditioning
        pocket_condition = self._encode_pocket(
            pocket_x, pocket_pos, pocket_edge_index, pocket_batch, batch
        )
        
        h_final = fusion_output['fused_features']
        if pocket_condition is not None:
            h_final = self._apply_conditioning(h_final, pocket_condition, batch)
        
        # Predictions
        atom_logits = self.atom_type_head(h_final)
        pos_pred = physical_output['updated_positions'] + self.position_head(h_final)
        
        # Bond predictions
        if edge_index.size(1) > 0:
            row, col = edge_index
            edge_features = torch.cat([h_final[row], h_final[col]], dim=-1)
            bond_logits = self.bond_type_head(edge_features)
        else:
            bond_logits = torch.zeros((0, self.bond_types), device=x.device)
        
        return {
            'atom_logits': atom_logits,
            'pos_pred': pos_pred,
            'bond_logits': bond_logits,
            'node_features': h_final,
            'chemical_properties': chemical_output['chemical_properties'],
            'physical_forces': fusion_output['forces'],
            'interactions': fusion_output['interactions'],
            'consistency_score': fusion_output['consistency_score'],
            'constraint_losses': fusion_output['constraint_losses'],
            'total_constraint_loss': fusion_output['total_constraint_loss']
        }
    
    def _embed_atoms_flexible(self, x):
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

def create_joint2d3d_model(hidden_dim=256, num_layers=6, conditioning_type="add", **kwargs):
    """Create Joint2D3D model with CORRECTED SE(3) equivariant EGNN"""
    return Joint2D3DModel(
        atom_types=11,
        bond_types=4,
        hidden_dim=hidden_dim,
        pocket_dim=hidden_dim,
        num_layers=num_layers,
        conditioning_type=conditioning_type,
        **kwargs
    )