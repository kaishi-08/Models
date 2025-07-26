# src/models/egnn_backbone.py - EGNN Replacement cho SchNet
import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph
from torch_geometric.utils import to_dense_batch
import math

class EGNNLayer(nn.Module):
    """🎯 EGNN Layer - Perfect for Molecular Generation"""
    
    def __init__(self, hidden_dim, edge_dim=1, act_fn=nn.SiLU(), 
                 residual=True, attention=False, normalize=False, tanh=False):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.tanh = tanh
        
        # Edge model φ_e
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + 1, hidden_dim),  # +1 for distance
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )
        
        # Node model φ_h  
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Coordinate model φ_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 1)
        )
        
        # Attention mechanism (optional)
        if attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        
        # Layer normalization
        if normalize:
            self.layer_norm = nn.LayerNorm(hidden_dim)
            
    def forward(self, h, pos, edge_index, edge_attr=None):
        """
        EGNN Forward pass
        
        Args:
            h: Node features [N, hidden_dim]
            pos: Coordinates [N, 3]
            edge_index: [2, E]
            edge_attr: Edge features [E, edge_dim]
            
        Returns:
            h_new: Updated node features [N, hidden_dim]
            pos_new: Updated coordinates [N, 3]
        """
        
        row, col = edge_index
        
        # 🔧 Edge features với distance
        radial = pos[row] - pos[col]  # [E, 3]
        radial_norm = torch.norm(radial, dim=-1, keepdim=True)  # [E, 1]
        
        # Avoid division by zero
        radial_norm = torch.clamp(radial_norm, min=1e-8)
        
        # Edge input: [h_i, h_j, edge_attr, ||x_i - x_j||]
        edge_input = [h[row], h[col], radial_norm]
        if edge_attr is not None:
            edge_input.append(edge_attr)
        
        edge_feat = torch.cat(edge_input, dim=-1)
        
        # 🔧 Edge model
        m_ij = self.edge_mlp(edge_feat)  # [E, hidden_dim]
        
        # 🔧 Coordinate update
        coord_diff = self.coord_mlp(m_ij)  # [E, 1]
        
        if self.tanh:
            coord_diff = torch.tanh(coord_diff)
            
        # Normalize radial direction
        radial_normalized = radial / radial_norm
        
        # Update coordinates: x_i = x_i + Σ_j (φ_x(m_ij) * (x_i - x_j)/||x_i - x_j||)
        coord_update = coord_diff * radial_normalized  # [E, 3]
        
        pos_new = pos.clone()
        pos_new.index_add_(0, row, coord_update)
        
        # 🔧 Node feature update
        # Aggregate messages: Σ_j m_ij
        agg = torch.zeros(h.size(0), self.hidden_dim, device=h.device)
        
        if self.attention:
            att = self.att_mlp(m_ij)  # [E, 1]
            m_ij = m_ij * att
            
        agg.index_add_(0, row, m_ij)
        
        # Update node features
        h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
        
        # Residual connection
        if self.residual:
            h_new = h + h_new
            
        # Layer normalization
        if self.normalize:
            h_new = self.layer_norm(h_new)
            
        return h_new, pos_new


class EGNNBackbone(nn.Module):
    """🎯 EGNN Backbone - Thay thế SchNet hoàn toàn"""
    
    def __init__(self, hidden_dim=256, num_layers=4, cutoff=10.0, 
                 residual=True, attention=True, normalize=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers  
        self.cutoff = cutoff
        
        # EGNN layers
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(
                hidden_dim=hidden_dim,
                residual=residual,
                attention=attention and (i > 0),  # Attention từ layer 2
                normalize=normalize
            ) for i in range(num_layers)
        ])
        
        print(f"✅ EGNN Backbone: {num_layers} layers, cutoff={cutoff}Å")
        print(f"   Features: residual={residual}, attention={attention}")
    
    def forward(self, h, pos, batch):
        """
        🎯 EGNN Forward - Returns ATOM-LEVEL features
        
        Args:
            h: Initial node features [N, hidden_dim]
            pos: Coordinates [N, 3]  
            batch: Batch indices [N]
            
        Returns:
            h_final: Final node features [N, hidden_dim]
            pos_final: Final coordinates [N, 3]
        """
        
        # Build edge index based on distance cutoff
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, 
                                 max_num_neighbors=32)
        
        if edge_index.size(1) == 0:
            # No edges - return original features
            return h, pos
        
        h_current = h
        pos_current = pos
        
        # Apply EGNN layers
        for i, egnn_layer in enumerate(self.egnn_layers):
            h_current, pos_current = egnn_layer(
                h_current, pos_current, edge_index
            )
        
        # 🎯 GUARANTEED: Returns [N_atoms, hidden_dim] - NO aggregation!
        assert h_current.size(0) == h.size(0), f"Atom count changed: {h_current.size(0)} != {h.size(0)}"
        assert h_current.size(1) == self.hidden_dim, f"Feature dim wrong: {h_current.size(1)} != {self.hidden_dim}"
        
        return h_current, pos_current


# 🔧 FIXED Joint2D3D với EGNN
class Joint2D3DEGNNModel(nn.Module):
    """Joint 2D-3D Model với EGNN backbone thay vì SchNet"""
    
    def __init__(self, atom_types=11, bond_types=4, hidden_dim=256, 
                 num_layers=4, cutoff=10.0):
        super().__init__()
        
        self.atom_types = atom_types
        self.bond_types = bond_types 
        self.hidden_dim = hidden_dim
        
        # Embeddings
        self.atom_embedding = nn.Linear(6, hidden_dim)  # Flexible input
        
        # 🎯 EGNN thay vì SchNet
        self.egnn_3d = EGNNBackbone(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            cutoff=cutoff,
            residual=True,
            attention=True,
            normalize=True
        )
        
        # 2D processing
        self.gnn_2d_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output heads
        self.atom_head = nn.Linear(hidden_dim, atom_types)
        self.position_head = nn.Linear(hidden_dim, 3)
        
        print(f"🎯 Joint2D3D-EGNN Model initialized")
        print(f"   EGNN replaces SchNet - guaranteed atom-level features!")
    
    def forward(self, x, pos, edge_index, edge_attr, batch, **kwargs):
        """Forward pass with EGNN"""
        
        # Embeddings
        h = self.atom_embedding(x.float())
        
        # 2D processing
        h_2d = h
        for layer in self.gnn_2d_layers:
            h_2d = h_2d + layer(h_2d, edge_index, edge_attr)
        
        # 🎯 3D processing với EGNN
        h_3d, pos_updated = self.egnn_3d(h, pos, batch)
        
        # Fusion
        h_fused = self.fusion(torch.cat([h_2d, h_3d], dim=-1))
        
        # Outputs
        atom_logits = self.atom_head(h_fused)
        pos_pred = pos_updated + self.position_head(h_fused)
        
        return {
            'atom_logits': atom_logits,
            'pos_pred': pos_pred,
            'node_features': h_fused
        }


class GraphConvLayer(nn.Module):
    """Simple 2D graph convolution"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.SiLU()
        
    def forward(self, x, edge_index, edge_attr):
        if edge_index.size(1) == 0:
            return self.activation(self.linear(x))
        
        row, col = edge_index
        messages = x[col]  # Simple message passing
        
        # Aggregate
        out = torch.zeros_like(x)
        out.index_add_(0, row, messages)
        
        return self.activation(self.linear(x + out))


# 🎯 Factory function
def create_egnn_model(hidden_dim=256, num_layers=4):
    """Create EGNN model to replace SchNet"""
    return Joint2D3DEGNNModel(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        cutoff=10.0
    )