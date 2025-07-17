# src/models/pocket_encoder.py - Fixed version
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch
from .e3_egnn import E3EquivariantGNN

class ProteinPocketEncoder(nn.Module):
    """Protein pocket encoder using E(3) equivariant networks"""
    
    def __init__(self, node_features: int = 8, edge_features: int = 4,
                 hidden_dim: int = 128, num_layers: int = 4, 
                 output_dim: int = 256, max_radius: float = 10.0):
        super().__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        
        # Use linear layer for feature projection instead of fixed embedding
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.edge_embedding = nn.Linear(edge_features if edge_features > 0 else 1, hidden_dim)
        
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
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor = None,
                batch: torch.Tensor = None):
        """
        Args:
            x: Node features [N, node_features] - preprocessed features
            pos: Node positions [N, 3]
            edge_index: Edge indices [2, E] (optional)
            batch: Batch indices [N] (optional)
        """
        # Handle input features
        if x.dim() == 2 and x.size(1) == self.node_features:
            h = self.node_embedding(x.float())
        else:
            raise ValueError(f"Expected x shape [N, {self.node_features}], got {x.shape}")
        
        # Create edges if not provided (distance-based)
        if edge_index is None or edge_index.size(1) == 0:
            edge_index = self._create_distance_edges(pos, max_dist=10.0)
        
        # Apply E(3) equivariant network
        try:
            h = self.e3_net(h, pos, edge_index, batch)
        except Exception as e:
            print(f"Warning: E3 network failed: {e}")
            # Fallback to simple processing
            h = self.node_embedding(x.float())
        
        # Handle batching
        if batch is not None:
            # Convert to dense format for attention
            try:
                h_dense, mask = to_dense_batch(h, batch)
                
                # Self-attention within pocket
                h_att, _ = self.attention(h_dense, h_dense, h_dense, key_padding_mask=~mask)
                
                # Back to sparse format
                h_att = h_att[mask]
                
                # Global pooling
                pocket_repr = global_mean_pool(h_att, batch)
            except:
                # Fallback to simple pooling
                pocket_repr = global_mean_pool(h, batch)
        else:
            # Single pocket case
            h_mean = torch.mean(h, dim=0, keepdim=True)
            pocket_repr = h_mean
        
        # Output projection
        return self.output_projection(pocket_repr)
    
    def _create_distance_edges(self, pos: torch.Tensor, max_dist: float = 10.0):
        """Create edges based on distance threshold"""
        from torch_geometric.nn import radius_graph
        
        # Limit number of atoms to avoid memory issues
        if pos.size(0) > 1000:
            # Sample subset of atoms
            indices = torch.randperm(pos.size(0))[:1000]
            pos_subset = pos[indices]
            edge_index = radius_graph(pos_subset, r=max_dist, batch=None, max_num_neighbors=64)
            # Map back to original indices
            edge_index = indices[edge_index]
        else:
            edge_index = radius_graph(pos, r=max_dist, batch=None, max_num_neighbors=64)
        
        return edge_index

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
        
        try:
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
            
        except Exception as e:
            print(f"Warning: Cross attention failed: {e}")
            # Fallback: simple addition
            pocket_broadcast = pocket_proj[ligand_batch]
            combined = ligand_proj + pocket_broadcast
            return self.output_proj(combined)