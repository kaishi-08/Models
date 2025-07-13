import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from .e3_egnn import E3EquivariantGNN
from .pocket_encoder import ProteinPocketEncoder, CrossAttentionPocketConditioner

class Joint2D3DMolecularModel(nn.Module):
    """Joint 2D/3D molecular generation model with pocket conditioning"""
    
    def __init__(self, atom_types: int = 100, bond_types: int = 5,
                 hidden_dim: int = 128, pocket_dim: int = 256,
                 num_layers: int = 4, max_radius: float = 10.0):
        super().__init__()
        
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.hidden_dim = hidden_dim
        
        # Atom type embedding
        self.atom_embedding = nn.Embedding(atom_types, hidden_dim)
        
        # Bond type embedding
        self.bond_embedding = nn.Embedding(bond_types, hidden_dim)
        
        # 2D Graph Neural Network for connectivity
        self.graph_2d_net = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # 3D E(3) Equivariant Network for geometry
        self.e3_3d_net = E3EquivariantGNN(
            irreps_in=f"{hidden_dim}x0e",
            irreps_hidden=f"{hidden_dim}x0e+{hidden_dim//2}x1o",
            irreps_out=f"{hidden_dim}x0e",
            num_layers=num_layers,
            max_radius=max_radius
        )
        
        # Pocket encoder
        self.pocket_encoder = ProteinPocketEncoder(
            hidden_dim=hidden_dim,
            output_dim=pocket_dim
        )
        
        # Cross-attention for pocket conditioning
        self.pocket_conditioner = CrossAttentionPocketConditioner(
            ligand_dim=hidden_dim,
            pocket_dim=pocket_dim,
            hidden_dim=hidden_dim
        )
        
        # 2D/3D fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output heads
        self.atom_type_head = nn.Linear(hidden_dim, atom_types)
        self.bond_type_head = nn.Linear(hidden_dim * 2, bond_types)
        self.position_head = nn.Linear(hidden_dim, 3)
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: torch.Tensor,
                pocket_x: torch.Tensor = None, pocket_pos: torch.Tensor = None,
                pocket_edge_index: torch.Tensor = None, pocket_batch: torch.Tensor = None):
        """
        Args:
            x: Atom types [N, 1]
            pos: Atom positions [N, 3]
            edge_index: Bond connectivity [2, E]
            edge_attr: Bond types [E, 1]
            batch: Batch indices [N]
            pocket_*: Protein pocket features
        """
        # Embed atom and bond features
        atom_emb = self.atom_embedding(x.squeeze(-1))
        bond_emb = self.bond_embedding(edge_attr.squeeze(-1))
        
        # 2D graph convolution
        h_2d = atom_emb
        for layer in self.graph_2d_net:
            h_2d = layer(h_2d, edge_index, bond_emb)
        
        # 3D equivariant convolution
        h_3d = self.e3_3d_net(atom_emb, pos, edge_index, batch)
        
        # Pocket conditioning
        if pocket_x is not None:
            pocket_repr = self.pocket_encoder(
                pocket_x, pocket_pos, pocket_edge_index, pocket_batch
            )
            h_2d = self.pocket_conditioner(h_2d, pocket_repr, batch)
            h_3d = self.pocket_conditioner(h_3d, pocket_repr, batch)
        
        # Fuse 2D and 3D features
        h_fused = self.fusion_layer(torch.cat([h_2d, h_3d], dim=-1))
        
        # Output predictions
        atom_logits = self.atom_type_head(h_fused)
        pos_pred = self.position_head(h_fused)
        
        # Bond predictions (needs edge features)
        row, col = edge_index
        edge_features = torch.cat([h_fused[row], h_fused[col]], dim=-1)
        bond_logits = self.bond_type_head(edge_features)
        
        return {
            'atom_logits': atom_logits,
            'pos_pred': pos_pred,
            'bond_logits': bond_logits,
            'node_features': h_fused
        }

class GraphConvLayer(nn.Module):
    """Graph convolution layer for 2D molecular graphs"""
    
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
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor):
        row, col = edge_index
        
        # Message passing
        messages = self.message_mlp(torch.cat([x[row], x[col], edge_attr], dim=-1))
        
        # Aggregate messages
        out = torch.zeros_like(x)
        out = out.index_add(0, col, messages)
        
        # Update node features
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        
        return out + x  # Residual connection