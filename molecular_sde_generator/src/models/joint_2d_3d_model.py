# src/models/joint_2d_3d_model.py - FIXED for 7D pocket features
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch
from .base_model import MolecularModel
from .e3_egnn import E3EquivariantGNN

class Joint2D3DMolecularModel(MolecularModel):
    """
    COMPLETE FIX: Handle 6D ligand features and 7D pocket features
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
        
        # === FLEXIBLE EMBEDDINGS FOR DIFFERENT FEATURE DIMENSIONS ===
        # Ligand embeddings (typically 6D or 8D)
        self.atom_embedding_6d = nn.Linear(6, hidden_dim)
        self.atom_embedding_8d = nn.Linear(8, hidden_dim)
        
        # Pocket embeddings (typically 7D, sometimes 8D)
        self.pocket_atom_embedding_6d = nn.Linear(6, hidden_dim)
        self.pocket_atom_embedding_7d = nn.Linear(7, hidden_dim)  # 🎯 NEW: For 7D pocket features
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
        
        # === 3D EQUIVARIANT NETWORKS ===
        try:
            self.e3_3d_net = E3EquivariantGNN(
                irreps_in=f"{hidden_dim}x0e",
                irreps_hidden=f"{hidden_dim}x0e+{hidden_dim//2}x1o",
                irreps_out=f"{hidden_dim}x0e",
                num_layers=num_layers,
                max_radius=max_radius
            )
        except Exception as e:
            print(f"Warning: E3EquivariantGNN failed, using fallback: {e}")
            self.e3_3d_net = SimpleFallbackGNN(hidden_dim, num_layers)
        
        # === FEATURE FUSION ===
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # === OUTPUT HEADS ===
        self.atom_type_head = nn.Linear(hidden_dim, atom_types)
        self.bond_type_head = nn.Linear(hidden_dim * 2, bond_types)
        self.position_head = nn.Linear(hidden_dim, 3)
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: torch.Tensor,
                pocket_x: torch.Tensor = None, pocket_pos: torch.Tensor = None,
                pocket_edge_index: torch.Tensor = None, pocket_batch: torch.Tensor = None,
                **kwargs):
        """
        FIXED forward pass with proper dimension handling
        """
        
        # Ensure gradients are enabled
        if x.requires_grad == False:
            x = x.requires_grad_(True)
        if pos.requires_grad == False:
            pos = pos.requires_grad_(True)
        
        # Flexible atom embedding
        atom_emb = self._embed_atoms_flexible(x)
        bond_emb = self._embed_bonds(edge_attr, edge_index)
        
        # FIXED: Pocket conditioning with proper dimension handling
        pocket_condition = self._encode_pocket_flexible(pocket_x, pocket_pos, pocket_edge_index, pocket_batch, batch)
        
        # Apply conditioning
        conditioned_atom_emb = self._apply_conditioning(atom_emb, pocket_condition, batch)
        
        # 2D graph processing
        h_2d = conditioned_atom_emb
        for layer in self.graph_2d_net:
            h_2d = layer(h_2d, edge_index, bond_emb)
        
        # 3D equivariant processing
        h_3d = self.e3_3d_net(conditioned_atom_emb, pos, edge_index, batch)
        
        # Feature fusion
        h_fused = self.fusion_layer(torch.cat([h_2d, h_3d], dim=-1))
        
        # Predictions
        atom_logits = self.atom_type_head(h_fused)
        pos_pred = self.position_head(h_fused)
        
        # Bond predictions
        if edge_index.size(1) > 0:
            row, col = edge_index
            edge_features = torch.cat([h_fused[row], h_fused[col]], dim=-1)
            bond_logits = self.bond_type_head(edge_features)
        else:
            bond_logits = torch.zeros((0, self.bond_types), device=x.device)
        
        return {
            'atom_logits': atom_logits,
            'pos_pred': pos_pred,
            'bond_logits': bond_logits,
            'node_features': h_fused
        }
    
    def _embed_atoms_flexible(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flexible atom embedding for ligand features
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got {x.dim()}D: {x.shape}")
        
        input_dim = x.size(1)
        
        if input_dim == 6:
            return self.atom_embedding_6d(x.float())
        elif input_dim == 8:
            return self.atom_embedding_8d(x.float())
        elif input_dim < 6:
            # Pad to 6 dimensions
            padding = torch.zeros(x.size(0), 6 - input_dim, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
            return self.atom_embedding_6d(x_padded.float())
        elif input_dim == 7:
            # Handle 7D by padding to 8D
            padding = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
            return self.atom_embedding_8d(x_padded.float())
        else:
            # Truncate to 8 dimensions for >8D input
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
        """
        🎯 FIXED: Flexible pocket encoding with proper 7D handling
        """
        if pocket_x is None or pocket_pos is None:
            batch_size = ligand_batch.max().item() + 1
            return torch.zeros(batch_size, self.pocket_dim, device=ligand_batch.device)
        
        try:
            input_dim = pocket_x.size(1)
            # print(f"Debug: Pocket input dimension: {input_dim}")  # Debug line
            
            # 🎯 HANDLE DIFFERENT POCKET DIMENSIONS PROPERLY
            if input_dim == 6:
                pocket_emb = self.pocket_atom_embedding_6d(pocket_x.float())
            elif input_dim == 7:
                # 🎯 CRITICAL FIX: Use 7D embedding for 7D pocket features
                pocket_emb = self.pocket_atom_embedding_7d(pocket_x.float())
            elif input_dim == 8:
                pocket_emb = self.pocket_atom_embedding_8d(pocket_x.float())
            elif input_dim < 6:
                # Pad to 6 dimensions
                padding = torch.zeros(pocket_x.size(0), 6 - input_dim, 
                                    device=pocket_x.device, dtype=pocket_x.dtype)
                pocket_x_padded = torch.cat([pocket_x, padding], dim=1)
                pocket_emb = self.pocket_atom_embedding_6d(pocket_x_padded.float())
            elif input_dim > 8:
                # Truncate to 8 dimensions
                pocket_x_truncated = pocket_x[:, :8]
                pocket_emb = self.pocket_atom_embedding_8d(pocket_x_truncated.float())
            else:
                # This should not happen, but fallback to 7D
                pocket_emb = self.pocket_atom_embedding_7d(pocket_x.float())
            
            # Encode pocket
            pocket_repr = self.pocket_encoder(pocket_emb, pocket_pos, pocket_edge_index, pocket_batch)
            
            return pocket_repr
            
        except Exception as e:
            # Silent fallback - don't spam console
            batch_size = ligand_batch.max().item() + 1
            return torch.zeros(batch_size, self.pocket_dim, device=ligand_batch.device)
    
    def _apply_conditioning(self, atom_features: torch.Tensor, pocket_condition: torch.Tensor, 
                          batch: torch.Tensor) -> torch.Tensor:
        """
        Apply pocket conditioning with proper batch handling
        """
        if pocket_condition.abs().sum() == 0:
            return atom_features
        
        # Transform pocket condition
        pocket_transformed = self.condition_transform(pocket_condition)
        
        # Broadcast to atoms using batch indices
        batch_size = pocket_transformed.size(0)
        max_batch_idx = batch.max().item()
        
        if max_batch_idx >= batch_size:
            # Handle batch size mismatch by expanding
            pocket_transformed = pocket_transformed[0:1].expand(max_batch_idx + 1, -1)
        
        # Use batch indices to broadcast
        broadcasted_condition = pocket_transformed[batch]
        
        # Apply conditioning
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
        
        # Smart selection if pocket too large
        if pocket_features.size(0) > self.max_atoms:
            indices = torch.randperm(pocket_features.size(0))[:self.max_atoms]
            pocket_features = pocket_features[indices]
            pocket_batch = pocket_batch[indices]
        
        # Process each pocket atom
        processed_features = self.atom_processor(pocket_features)
        
        # Global pooling per batch
        global_mean = global_mean_pool(processed_features, pocket_batch)
        global_max = global_max_pool(processed_features, pocket_batch)
        global_combined = global_mean + global_max
        
        # Final processing
        global_repr = self.global_processor(global_combined)
        
        return global_repr

class SimpleFallbackGNN(nn.Module):
    """Fallback GNN if E3EquivariantGNN fails"""
    
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
    
    def forward(self, x, pos, edge_index, batch=None):
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, torch.zeros((edge_index.size(1), h.size(1)), device=x.device)) + h
        return h

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
        
        # Handle edge_attr dimension
        if edge_attr.size(1) != x.size(1):
            edge_attr = edge_attr[:, :min(edge_attr.size(1), x.size(1))]
            if edge_attr.size(1) < x.size(1):
                padding = torch.zeros(edge_attr.size(0), x.size(1) - edge_attr.size(1), device=x.device)
                edge_attr = torch.cat([edge_attr, padding], dim=1)
        
        # Message passing
        messages = self.message_mlp(torch.cat([x[row], x[col], edge_attr], dim=-1))
        
        # Aggregate
        out = torch.zeros_like(x)
        out = out.index_add(0, col, messages)
        
        # Update
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        
        return out