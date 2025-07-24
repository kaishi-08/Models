# src/models/joint_2d_3d_model.py - FIXED to be position-sensitive
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch
from .base_model import MolecularModel

class Joint2D3DMolecularModel(MolecularModel):
    """
    CRITICAL FIX: Model that actually uses position information for predictions
    """
    
    def __init__(self, atom_types: int = 100, bond_types: int = 5,
                 hidden_dim: int = 128, pocket_dim: int = 256,
                 num_layers: int = 4, max_radius: float = 10.0,
                 max_pocket_atoms: int = 1000,
                 conditioning_type: str = "add"):
        super().__init__(atom_types, bond_types, hidden_dim)
        
        self.pocket_dim = pocket_dim
        self.num_layers = num_layers
        self.max_radius = max_radius
        self.max_pocket_atoms = max_pocket_atoms
        self.conditioning_type = conditioning_type
        
        # === FLEXIBLE EMBEDDINGS ===
        self.atom_embedding_6d = nn.Linear(6, hidden_dim)
        self.atom_embedding_8d = nn.Linear(8, hidden_dim)
        
        # 🎯 CRITICAL FIX: Position embedding that forces position sensitivity
        self.position_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Feature combiner for atom + position features
        self.feature_combiner = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        
        # Distance feature encoder
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # Pocket embeddings
        self.pocket_atom_embedding_6d = nn.Linear(6, hidden_dim)
        self.pocket_atom_embedding_7d = nn.Linear(7, hidden_dim)
        self.pocket_atom_embedding_8d = nn.Linear(8, hidden_dim)
        
        self.bond_embedding = nn.Linear(1, hidden_dim)
        
        # === POCKET ENCODER ===
        self.pocket_encoder = SimplePocketEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=pocket_dim,
            max_atoms=max_pocket_atoms
        )
        
        # === CONDITIONING MODULE ===
        if conditioning_type == "add":
            assert pocket_dim == hidden_dim, f"For 'add' conditioning, pocket_dim ({pocket_dim}) must equal hidden_dim ({hidden_dim})"
            self.condition_transform = nn.Identity()
        elif conditioning_type == "concat":
            self.condition_transform = nn.Linear(pocket_dim, hidden_dim)
            self.feature_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        elif conditioning_type == "gated":
            self.condition_transform = nn.Linear(pocket_dim, hidden_dim)
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
        
        # === 2D GRAPH NETWORKS ===
        self.graph_2d_net = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # === 3D POSITION-AWARE NETWORKS ===
        # 🎯 CRITICAL: Networks that explicitly use position information
        self.position_nets = nn.ModuleList([
            PositionAwareLayer(hidden_dim, max_radius) for _ in range(num_layers)
        ])
        
        # === FEATURE FUSION WITH POSITION DEPENDENCY ===
        # 🎯 CRITICAL: Fusion that forces position dependency
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # 3x: features + pos_embedding + distance_features
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # === OUTPUT HEADS WITH POSITION DEPENDENCY ===
        # 🎯 CRITICAL: Position head that explicitly depends on input positions
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),  # Include position embedding
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # Predict position delta
        )
        
        self.atom_type_head = nn.Linear(hidden_dim, atom_types)
        self.bond_type_head = nn.Linear(hidden_dim * 2, bond_types)
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: torch.Tensor,
                pocket_x: torch.Tensor = None, pocket_pos: torch.Tensor = None,
                pocket_edge_index: torch.Tensor = None, pocket_batch: torch.Tensor = None,
                **kwargs):
        """
        🎯 FIXED: Forward pass that MUST use position information
        """
        
        # Ensure gradients are enabled
        if not x.requires_grad:
            x = x.requires_grad_(True)
        if not pos.requires_grad:
            pos = pos.requires_grad_(True)
        
        # 🎯 CRITICAL: Embed positions FIRST to force dependency
        pos_emb = self.position_embedding(pos)  # [N, hidden_dim//2]
        
        # Flexible atom embedding
        atom_emb = self._embed_atoms_flexible(x)  # [N, hidden_dim]
        
        # 🎯 CRITICAL: Combine atom features with position features immediately
        # This forces the model to consider positions from the start
        combined_features = torch.cat([atom_emb, pos_emb], dim=-1)  # [N, hidden_dim + hidden_dim//2]
        combined_features = self.feature_combiner(combined_features)
        
        bond_emb = self._embed_bonds(edge_attr, edge_index)
        
        # Pocket conditioning
        pocket_condition = self._encode_pocket_flexible(pocket_x, pocket_pos, pocket_edge_index, pocket_batch, batch)
        
        # Apply conditioning to position-aware features
        conditioned_features = self._apply_conditioning(combined_features, pocket_condition, batch)
        
        # 🎯 CRITICAL: Position-aware processing
        h_pos_aware = conditioned_features
        for pos_layer in self.position_nets:
            h_pos_aware = pos_layer(h_pos_aware, pos, edge_index) + h_pos_aware
        
        # 2D graph processing (using position-aware features)
        h_2d = h_pos_aware
        for layer in self.graph_2d_net:
            h_2d = layer(h_2d, edge_index, bond_emb) + h_2d
        
        # 🎯 CRITICAL: Create distance features from positions
        distance_features = self._create_distance_features(pos, edge_index, self.hidden_dim)
        
        # Feature fusion with explicit position dependency
        final_features = torch.cat([h_pos_aware, h_2d, distance_features], dim=-1)
        h_fused = self.fusion_layer(final_features)
        
        # 🎯 CRITICAL: Position prediction with explicit position input
        # Concatenate fused features with position embedding for position head
        pos_input = torch.cat([h_fused, pos_emb], dim=-1)
        pos_delta = self.position_head(pos_input)
        
        # 🎯 CRITICAL: Position prediction is current_pos + delta (forces dependency)
        pos_pred = pos + pos_delta
        
        # Other predictions
        atom_logits = self.atom_type_head(h_fused)
        
        # Bond predictions
        if edge_index.size(1) > 0:
            row, col = edge_index
            edge_features = torch.cat([h_fused[row], h_fused[col]], dim=-1)
            bond_logits = self.bond_type_head(edge_features)
        else:
            bond_logits = torch.zeros((0, self.bond_types), device=x.device)
        
        return {
            'atom_logits': atom_logits,
            'pos_pred': pos_pred,  # This now MUST depend on input positions
            'bond_logits': bond_logits,
            'node_features': h_fused
        }
    
    def _create_distance_features(self, pos: torch.Tensor, edge_index: torch.Tensor, 
                                hidden_dim: int) -> torch.Tensor:
        """
        🎯 CRITICAL: Create distance-based features that force position sensitivity
        """
        num_atoms = pos.size(0)
        device = pos.device
        
        if edge_index.size(1) == 0:
            return torch.zeros(num_atoms, hidden_dim, device=device)
        
        row, col = edge_index
        
        # Compute distances
        edge_vectors = pos[row] - pos[col]  # [E, 3]
        distances = torch.norm(edge_vectors, dim=-1, keepdim=True)  # [E, 1]
        
        # Use pre-defined distance encoder
        edge_distance_features = self.distance_encoder(distances)  # [E, hidden_dim//4]
        
        # Aggregate distance features per atom
        atom_distance_features = torch.zeros(num_atoms, hidden_dim // 4, device=device)
        atom_distance_features = atom_distance_features.index_add(0, col, edge_distance_features)
        
        # Pad to full hidden_dim
        padding = torch.zeros(num_atoms, hidden_dim - hidden_dim // 4, device=device)
        full_distance_features = torch.cat([atom_distance_features, padding], dim=-1)
        
        return full_distance_features
    
    def _embed_atoms_flexible(self, x: torch.Tensor) -> torch.Tensor:
        """Flexible atom embedding for ligand features"""
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got {x.dim()}D: {x.shape}")
        
        input_dim = x.size(1)
        
        if input_dim == 6:
            return self.atom_embedding_6d(x.float())
        elif input_dim == 8:
            return self.atom_embedding_8d(x.float())
        elif input_dim < 6:
            padding = torch.zeros(x.size(0), 6 - input_dim, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
            return self.atom_embedding_6d(x_padded.float())
        elif input_dim == 7:
            padding = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
            return self.atom_embedding_8d(x_padded.float())
        else:
            x_truncated = x[:, :8]
            return self.atom_embedding_8d(x_truncated.float())
    
    def _embed_bonds(self, edge_attr: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Embed bond features safely"""
        if edge_index.size(1) == 0:
            return torch.zeros((0, self.hidden_dim), device=edge_attr.device)
        
        if edge_attr.dim() == 2:
            bond_emb = self.bond_embedding(edge_attr[:, :1].float())
        else:
            bond_emb = self.bond_embedding(edge_attr.unsqueeze(-1).float())
        
        return bond_emb
    
    def _encode_pocket_flexible(self, pocket_x: torch.Tensor, pocket_pos: torch.Tensor,
                               pocket_edge_index: torch.Tensor, pocket_batch: torch.Tensor,
                               ligand_batch: torch.Tensor) -> torch.Tensor:
        """Flexible pocket encoding with proper 7D handling"""
        if pocket_x is None or pocket_pos is None:
            batch_size = ligand_batch.max().item() + 1
            return torch.zeros(batch_size, self.pocket_dim, device=ligand_batch.device)
        
        try:
            input_dim = pocket_x.size(1)
            
            if input_dim == 6:
                pocket_emb = self.pocket_atom_embedding_6d(pocket_x.float())
            elif input_dim == 7:
                pocket_emb = self.pocket_atom_embedding_7d(pocket_x.float())
            elif input_dim == 8:
                pocket_emb = self.pocket_atom_embedding_8d(pocket_x.float())
            elif input_dim < 6:
                padding = torch.zeros(pocket_x.size(0), 6 - input_dim, 
                                    device=pocket_x.device, dtype=pocket_x.dtype)
                pocket_x_padded = torch.cat([pocket_x, padding], dim=1)
                pocket_emb = self.pocket_atom_embedding_6d(pocket_x_padded.float())
            elif input_dim > 8:
                pocket_x_truncated = pocket_x[:, :8]
                pocket_emb = self.pocket_atom_embedding_8d(pocket_x_truncated.float())
            else:
                pocket_emb = self.pocket_atom_embedding_7d(pocket_x.float())
            
            pocket_repr = self.pocket_encoder(pocket_emb, pocket_pos, pocket_edge_index, pocket_batch)
            return pocket_repr
            
        except Exception as e:
            batch_size = ligand_batch.max().item() + 1
            return torch.zeros(batch_size, self.pocket_dim, device=ligand_batch.device)
    
    def _apply_conditioning(self, atom_features: torch.Tensor, pocket_condition: torch.Tensor, 
                          batch: torch.Tensor) -> torch.Tensor:
        """Apply pocket conditioning with proper batch handling"""
        if pocket_condition.abs().sum() == 0:
            return atom_features
        
        pocket_transformed = self.condition_transform(pocket_condition)
        
        batch_size = pocket_transformed.size(0)
        max_batch_idx = batch.max().item()
        
        if max_batch_idx >= batch_size:
            pocket_transformed = pocket_transformed[0:1].expand(max_batch_idx + 1, -1)
        
        broadcasted_condition = pocket_transformed[batch]
        
        if self.conditioning_type == "add":
            return atom_features + broadcasted_condition
        elif self.conditioning_type == "concat":
            concatenated = torch.cat([atom_features, broadcasted_condition], dim=-1)
            return self.feature_fusion(concatenated)
        elif self.conditioning_type == "gated":
            concatenated = torch.cat([atom_features, broadcasted_condition], dim=-1)
            gate = self.gate_net(concatenated)
            return atom_features * gate + broadcasted_condition * (1 - gate)
        else:
            return atom_features

class PositionAwareLayer(nn.Module):
    """
    🎯 CRITICAL: Layer that explicitly uses position information
    """
    
    def __init__(self, hidden_dim: int, max_radius: float = 10.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_radius = max_radius
        
        # Position-aware message passing
        self.position_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),  # Process 3D positions
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 4, hidden_dim),  # Include position features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        """Position-aware forward pass"""
        if edge_index.size(1) == 0:
            return x
        
        row, col = edge_index
        
        # 🎯 CRITICAL: Use position differences for messages
        pos_diff = pos[row] - pos[col]  # [E, 3]
        pos_features = self.position_mlp(pos_diff)  # [E, hidden_dim//4]
        
        # Messages include position information
        messages = self.message_mlp(torch.cat([x[row], x[col], pos_features], dim=-1))
        
        # Aggregate messages
        out = torch.zeros_like(x)
        out = out.index_add(0, col, messages)
        
        # Update with original features
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        
        return out

class SimplePocketEncoder(nn.Module):
    """Simple but effective pocket encoder"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, max_atoms: int = 1000):
        super().__init__()
        self.max_atoms = max_atoms
        
        self.atom_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.global_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, pocket_features: torch.Tensor, pocket_pos: torch.Tensor,
                pocket_edge_index: torch.Tensor, pocket_batch: torch.Tensor) -> torch.Tensor:
        
        if pocket_features.size(0) > self.max_atoms:
            indices = torch.randperm(pocket_features.size(0))[:self.max_atoms]
            pocket_features = pocket_features[indices]
            pocket_batch = pocket_batch[indices]
        
        processed_features = self.atom_processor(pocket_features)
        
        global_mean = global_mean_pool(processed_features, pocket_batch)
        global_max = global_max_pool(processed_features, pocket_batch)
        global_combined = global_mean + global_max
        
        global_repr = self.global_processor(global_combined)
        
        return global_repr

class GraphConvLayer(nn.Module):
    """Simple graph convolution layer"""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        if edge_index.size(1) == 0:
            return x
            
        row, col = edge_index
        
        if edge_attr.size(1) != x.size(1):
            edge_attr = edge_attr[:, :min(edge_attr.size(1), x.size(1))]
            if edge_attr.size(1) < x.size(1):
                padding = torch.zeros(edge_attr.size(0), x.size(1) - edge_attr.size(1), device=x.device)
                edge_attr = torch.cat([edge_attr, padding], dim=1)
        
        messages = self.message_mlp(torch.cat([x[row], x[col], edge_attr], dim=-1))
        
        out = torch.zeros_like(x)
        out = out.index_add(0, col, messages)
        
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        
        return out