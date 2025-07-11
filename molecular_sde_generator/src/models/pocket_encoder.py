# src/models/pocket_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch

class ProteinPocketEncoder(nn.Module):
    """Protein pocket encoder using E(3) equivariant networks"""
    
    def __init__(self, node_features: int = 20, edge_features: int = 4,
                 hidden_dim: int = 128, num_layers: int = 4, 
                 output_dim: int = 256, max_radius: float = 10.0):
        super().__init__()
        
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.edge_embedding = nn.Linear(edge_features, hidden_dim)
        
        # E(3) equivariant network for pocket encoding
        self.e3_net = E3EquivariantGNN(
            irreps_in=f"{hidden_dim}x0e",
            irreps_hidden=f"{hidden_dim}x0e+{hidden_dim//2}x1o",
            irreps_out=f"{hidden_dim}x0e",
            num_layers=num_layers,
            max_radius=max_radius
        )
        
        # Attention mechanism for pocket-ligand interaction
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor = None):
        """
        Args:
            x: Node features [N, node_features]
            pos: Node positions [N, 3]
            edge_index: Edge indices [2, E]
            batch: Batch indices [N]
        """
        # Embed node features
        h = self.node_embedding(x)
        
        # Apply E(3) equivariant network
        h = self.e3_net(h, pos, edge_index, batch)
        
        # Convert to dense format for attention
        h_dense, mask = to_dense_batch(h, batch)
        
        # Self-attention within pocket
        h_att, _ = self.attention(h_dense, h_dense, h_dense, key_padding_mask=~mask)
        
        # Global pooling
        h_att = h_att[mask]  # Back to sparse format
        pocket_repr = global_mean_pool(h_att, batch)
        
        # Output projection
        return self.output_projection(pocket_repr)

class CrossAttentionPocketConditioner(nn.Module):
    """Cross-attention mechanism for pocket-ligand conditioning"""
    
    def __init__(self, ligand_dim: int = 128, pocket_dim: int = 256, 
                 hidden_dim: int = 128, num_heads: int = 8):
        super().__init__()
        
        self.ligand_proj = nn.Linear(ligand_dim, hidden_dim)
        self.pocket_proj = nn.Linear(pocket_dim, hidden_dim)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(hidden_dim, ligand_dim)
        
    def forward(self, ligand_features: torch.Tensor, pocket_features: torch.Tensor,
                ligand_batch: torch.Tensor):
        """
        Args:
            ligand_features: [N_ligand, ligand_dim]
            pocket_features: [N_batch, pocket_dim]
            ligand_batch: [N_ligand] batch assignment for ligand atoms
        """
        # Project features
        ligand_proj = self.ligand_proj(ligand_features)
        pocket_proj = self.pocket_proj(pocket_features)
        
        # Convert to dense format
        ligand_dense, ligand_mask = to_dense_batch(ligand_proj, ligand_batch)
        
        # Expand pocket features to match ligand batch size
        pocket_expanded = pocket_proj[ligand_batch].unsqueeze(1)
        
        # Cross-attention: ligand queries, pocket keys/values
        attended_ligand, _ = self.cross_attention(
            ligand_dense, pocket_expanded, pocket_expanded,
            key_padding_mask=None
        )
        
        # Back to sparse format
        attended_ligand = attended_ligand[ligand_mask]
        
        # Output projection
        return self.output_proj(attended_ligand)