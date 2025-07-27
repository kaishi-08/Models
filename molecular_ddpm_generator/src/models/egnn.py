# src/models/egnn.py - REFINED: EGNN Backbone cho Molecular Generation
import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph
from torch_geometric.utils import to_dense_batch
import math

class EGNNLayer(nn.Module):
    """🎯 REFINED: EGNN Layer - E(n) Equivariant (like Pocket2Mol/SBDDiff)"""
    
    def __init__(self, hidden_dim, edge_dim=1, act_fn=nn.SiLU(), 
                 residual=True, attention=False, normalize=False, tanh=False):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.tanh = tanh
        
        # Edge model φ_e - processes edge features and distances
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + 1, hidden_dim),  # +1 for distance
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )
        
        # Node model φ_h - updates node features
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Coordinate model φ_x - updates 3D coordinates (equivariant!)
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
        EGNN Forward pass - E(n) Equivariant
        
        Args:
            h: Node features [N, hidden_dim]
            pos: Coordinates [N, 3] (equivariant)
            edge_index: [2, E]
            edge_attr: Edge features [E, edge_dim]
            
        Returns:
            h_new: Updated node features [N, hidden_dim]
            pos_new: Updated coordinates [N, 3] (equivariant)
        """
        
        row, col = edge_index
        
        # 🔧 Edge features with distance (equivariant)
        radial = pos[row] - pos[col]  # [E, 3] - relative positions
        radial_norm = torch.norm(radial, dim=-1, keepdim=True)  # [E, 1] - distances
        
        # Avoid division by zero
        radial_norm = torch.clamp(radial_norm, min=1e-8)
        
        # Edge input: [h_i, h_j, edge_attr, ||x_i - x_j||]
        edge_input = [h[row], h[col], radial_norm]
        if edge_attr is not None:
            edge_input.append(edge_attr)
        
        edge_feat = torch.cat(edge_input, dim=-1)
        
        # 🔧 Edge model - compute edge messages
        m_ij = self.edge_mlp(edge_feat)  # [E, hidden_dim]
        
        # 🔧 Coordinate update (equivariant!)
        coord_diff = self.coord_mlp(m_ij)  # [E, 1]
        
        if self.tanh:
            coord_diff = torch.tanh(coord_diff)
            
        # Normalize radial direction (unit vectors)
        radial_normalized = radial / radial_norm
        
        # Update coordinates: x_i = x_i + Σ_j (φ_x(m_ij) * (x_i - x_j)/||x_i - x_j||)
        coord_update = coord_diff * radial_normalized  # [E, 3]
        
        pos_new = pos.clone()
        pos_new.index_add_(0, row, coord_update)  # Aggregate coordinate updates
        
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
    """🎯 REFINED: EGNN Backbone - Full equivariant processing"""
    
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
                attention=attention and (i > 0),  # Attention from layer 2
                normalize=normalize,
                tanh=(i == num_layers - 1)  # Tanh on last layer for stability
            ) for i in range(num_layers)
        ])
        
        print(f"✅ EGNN Backbone: {num_layers} layers, cutoff={cutoff}Å")
        print(f"   Features: residual={residual}, attention={attention}, E(n)-equivariant")
    
    def forward(self, h, pos, batch):
        """
        🎯 EGNN Forward - Returns equivariant atom-level features
        
        Args:
            h: Initial node features [N, hidden_dim]
            pos: Coordinates [N, 3] (will be updated equivariantly)
            batch: Batch indices [N]
            
        Returns:
            h_final: Final node features [N, hidden_dim]
            pos_final: Final coordinates [N, 3] (equivariant updates)
        """
        
        # Build edge index based on distance cutoff
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, 
                                 max_num_neighbors=32)
        
        if edge_index.size(1) == 0:
            # No edges - return original features
            return h, pos
        
        h_current = h
        pos_current = pos
        
        # Apply EGNN layers sequentially
        for i, egnn_layer in enumerate(self.egnn_layers):
            h_current, pos_current = egnn_layer(
                h_current, pos_current, edge_index
            )
        
        # 🎯 GUARANTEED: Returns [N_atoms, hidden_dim] - NO aggregation!
        assert h_current.size(0) == h.size(0), f"Atom count changed: {h_current.size(0)} != {h.size(0)}"
        assert h_current.size(1) == self.hidden_dim, f"Feature dim wrong: {h_current.size(1)} != {self.hidden_dim}"
        
        return h_current, pos_current


class GraphConvLayer(nn.Module):
    """REFINED: Simple 2D graph convolution for chemical topology"""
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.edge_linear = nn.Linear(1, out_dim)
        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x, edge_index, edge_attr):
        if edge_index.size(1) == 0:
            return self.norm(self.activation(self.linear(x)))
        
        row, col = edge_index
        
        # Node transformation
        x_transformed = self.linear(x)
        
        # Message passing
        messages = x_transformed[col]
        
        # Add edge information if available
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            edge_features = self.edge_linear(edge_attr[:, :1])
            messages = messages + edge_features
        
        # Aggregate messages
        out = torch.zeros_like(x_transformed)
        out.index_add_(0, row, messages)
        
        # Add self-connection
        out = out + x_transformed
        
        return self.norm(self.activation(out))


class Joint2D3DEGNNModel(nn.Module):
    """🎯 REFINED: Joint 2D-3D Model với EGNN backbone"""
    
    def __init__(self, atom_types=11, bond_types=4, hidden_dim=256, 
                 num_layers=4, cutoff=10.0, pocket_dim=256,
                 conditioning_type="add"):
        super().__init__()
        
        self.atom_types = atom_types
        self.bond_types = bond_types 
        self.hidden_dim = hidden_dim
        self.conditioning_type = conditioning_type
        
        # Flexible embeddings for different input dimensions
        self.atom_embedding_6d = nn.Linear(6, hidden_dim)
        self.atom_embedding_7d = nn.Linear(7, hidden_dim)
        self.atom_embedding_8d = nn.Linear(8, hidden_dim)
        
        # 🎯 EGNN 3D Backend (main innovation)
        self.egnn_3d = EGNNBackbone(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            cutoff=cutoff,
            residual=True,
            attention=True,
            normalize=True
        )
        
        # 2D chemical topology processing
        self.gnn_2d_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # 2D-3D fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Pocket conditioning (if needed)
        if conditioning_type == "add":
            self.pocket_projection = nn.Linear(pocket_dim, hidden_dim)
        elif conditioning_type == "concat":
            self.pocket_projection = nn.Linear(pocket_dim, hidden_dim)
            self.fusion_with_pocket = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Output heads
        self.atom_head = nn.Linear(hidden_dim, atom_types)
        self.position_head = nn.Linear(hidden_dim, 3)
        self.bond_head = nn.Linear(hidden_dim * 2, bond_types)
        
        print(f"🎯 Joint2D3D-EGNN Model initialized")
        print(f"   EGNN replaces SchNet - guaranteed atom-level equivariant features!")
    
    def forward(self, x, pos, edge_index, edge_attr, batch, 
                pocket_x=None, pocket_pos=None, pocket_batch=None, **kwargs):
        """Forward pass with EGNN backend"""
        
        # Flexible atom embeddings
        h = self._embed_atoms_flexible(x)
        
        # 2D chemical topology processing
        h_2d = h
        for layer in self.gnn_2d_layers:
            h_2d = h_2d + layer(h_2d, edge_index, edge_attr)
        
        # 🎯 3D processing với EGNN (equivariant!)
        h_3d, pos_updated = self.egnn_3d(h, pos, batch)
        
        # 2D-3D fusion
        h_fused = self.fusion(torch.cat([h_2d, h_3d], dim=-1))
        
        # Pocket conditioning (optional)
        if pocket_x is not None and hasattr(self, 'pocket_projection'):
            pocket_repr = self._process_pocket(pocket_x, pocket_pos, pocket_batch)
            h_fused = self._apply_pocket_conditioning(h_fused, pocket_repr, batch)
        
        # Generate outputs
        atom_logits = self.atom_head(h_fused)
        pos_pred = pos_updated + self.position_head(h_fused)
        
        # Bond predictions
        if edge_index.size(1) > 0:
            row, col = edge_index
            edge_features = torch.cat([h_fused[row], h_fused[col]], dim=-1)
            bond_logits = self.bond_head(edge_features)
        else:
            bond_logits = torch.zeros((0, self.bond_types), device=x.device)
        
        return {
            'atom_logits': atom_logits,
            'pos_pred': pos_pred,
            'bond_logits': bond_logits,
            'node_features': h_fused
        }
    
    def _embed_atoms_flexible(self, x):
        """Flexible atom embedding for different input dimensions"""
        input_dim = x.size(1)
        
        if input_dim == 6:
            return self.atom_embedding_6d(x.float())
        elif input_dim == 7:
            return self.atom_embedding_7d(x.float())
        elif input_dim == 8:
            return self.atom_embedding_8d(x.float())
        elif input_dim < 6:
            padding = torch.zeros(x.size(0), 6 - input_dim, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
            return self.atom_embedding_6d(x_padded.float())
        else:
            # Truncate to 8D if larger
            x_truncated = x[:, :8]
            return self.atom_embedding_8d(x_truncated.float())
    
    def _process_pocket(self, pocket_x, pocket_pos, pocket_batch):
        """Simple pocket processing"""
        pocket_emb = self._embed_atoms_flexible(pocket_x)
        # Simple global pooling
        if pocket_batch is not None:
            from torch_geometric.nn import global_mean_pool
            pocket_repr = global_mean_pool(pocket_emb, pocket_batch)
        else:
            pocket_repr = torch.mean(pocket_emb, dim=0, keepdim=True)
        return self.pocket_projection(pocket_repr)
    
    def _apply_pocket_conditioning(self, h_fused, pocket_repr, batch):
        """Apply pocket conditioning"""
        if self.conditioning_type == "add":
            # Broadcast pocket features to all atoms
            pocket_broadcasted = pocket_repr[batch]
            return h_fused + pocket_broadcasted
        elif self.conditioning_type == "concat":
            pocket_broadcasted = pocket_repr[batch]
            combined = torch.cat([h_fused, pocket_broadcasted], dim=-1)
            return self.fusion_with_pocket(combined)
        else:
            return h_fused


# 🎯 Factory functions
def create_egnn_model(hidden_dim=256, num_layers=4, cutoff=10.0):
    """Create EGNN model for molecular generation"""
    return Joint2D3DEGNNModel(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        cutoff=cutoff
    )

def create_joint2d3d_egnn_model(hidden_dim=256, num_layers=6, **kwargs):
    """Create Joint2D3D EGNN model (main factory)"""
    return Joint2D3DEGNNModel(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        **kwargs
    )