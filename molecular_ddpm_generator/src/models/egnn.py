# src/models/egnn.py - E(3) Equivariant Graph Neural Networks
import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph

class EGNNLayer(nn.Module):
    """E(3) Equivariant Graph Neural Network Layer"""
    
    def __init__(self, hidden_dim, edge_dim=1, residual=True, attention=False):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.residual = residual
        self.attention = attention
        
        # Edge model
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node model
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Coordinate model
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
            
    def forward(self, h, pos, edge_index, edge_attr=None):
        if edge_index.size(1) == 0:
            return h, pos
            
        row, col = edge_index
        
        # Edge features with distance
        radial = pos[row] - pos[col]
        radial_norm = torch.norm(radial, dim=-1, keepdim=True)
        radial_norm = torch.clamp(radial_norm, min=1e-8)
        
        edge_input = [h[row], h[col], radial_norm]
        if edge_attr is not None:
            edge_input.append(edge_attr)
        
        edge_feat = torch.cat(edge_input, dim=-1)
        
        # Edge model
        m_ij = self.edge_mlp(edge_feat)
        
        # Coordinate update
        coord_diff = self.coord_mlp(m_ij)
        radial_normalized = radial / radial_norm
        coord_update = coord_diff * radial_normalized
        
        pos_new = pos.clone()
        pos_new.index_add_(0, row, coord_update)
        
        # Node feature update
        agg = torch.zeros(h.size(0), self.hidden_dim, device=h.device)
        
        if self.attention:
            att = self.att_mlp(m_ij)
            m_ij = m_ij * att
            
        agg.index_add_(0, row, m_ij)
        
        h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
        
        if self.residual:
            h_new = h + h_new
            
        return h_new, pos_new

class EGNNBackbone(nn.Module):
    """EGNN backbone for 3D processing"""
    
    def __init__(self, hidden_dim=256, num_layers=3, cutoff=10.0):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cutoff = cutoff
        
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(
                hidden_dim=hidden_dim,
                residual=True,
                attention=(i > 0)
            ) for i in range(num_layers)
        ])
        
    def forward(self, h, pos, batch):
        """Forward pass through EGNN layers"""
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=32)
        
        if edge_index.size(1) == 0:
            return h, pos
        
        h_current = h
        pos_current = pos
        
        for egnn_layer in self.egnn_layers:
            h_current, pos_current = egnn_layer(h_current, pos_current, edge_index)
        
        return h_current, pos_current

def create_egnn_backbone(hidden_dim=256, num_layers=3, cutoff=10.0):
    """Factory function to create EGNN backbone"""
    return EGNNBackbone(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        cutoff=cutoff
    )