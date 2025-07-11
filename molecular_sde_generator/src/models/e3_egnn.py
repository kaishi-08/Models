# src/models/e3_egnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from e3nn import o3
from e3nn.nn import Gate, NormActivation
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists

class E3EquivariantLayer(MessagePassing):
    """E(3) Equivariant Graph Neural Network Layer"""
    
    def __init__(self, irreps_in, irreps_out, irreps_sh, max_radius=10.0, 
                 number_of_basis=10, radial_layers=1, radial_neurons=100):
        super().__init__(aggr='add', node_dim=0)
        
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.max_radius = max_radius
        
        # Spherical harmonics for edge features
        self.sh = o3.SphericalHarmonics(self.irreps_sh, normalize=True, normalization='component')
        
        # Radial basis functions
        self.distance_expansion = GaussianSmearing(0.0, max_radius, number_of_basis)
        
        # Radial MLP
        self.radial_mlp = nn.Sequential(
            nn.Linear(number_of_basis, radial_neurons),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(radial_neurons, radial_neurons), nn.ReLU()) 
              for _ in range(radial_layers - 1)]
        )
        
        # Tensor product and linear layers
        irreps_mid = []
        instructions = []
        
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_sh):
                for ir_out in self.irreps_out:
                    if ir_out in ir_in * ir_edge:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvw', True))
        
        self.irreps_mid = o3.Irreps(irreps_mid)
        self.tp = o3.TensorProduct(self.irreps_in, self.irreps_sh, self.irreps_mid, 
                                  instructions, shared_weights=False, internal_weights=False)
        
        self.fc = nn.Linear(radial_neurons, self.tp.weight_numel)
        self.linear = o3.Linear(self.irreps_mid, self.irreps_out)
        
    def forward(self, x, pos, edge_index, edge_attr=None):
        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        edge_length = edge_vec.norm(dim=1, keepdim=True)
        
        # Compute spherical harmonics
        edge_sh = self.sh(edge_vec)
        
        # Compute radial features
        edge_length_embedded = self.distance_expansion(edge_length.squeeze())
        edge_weight = self.radial_mlp(edge_length_embedded)
        edge_weight = self.fc(edge_weight)
        
        # Message passing
        out = self.propagate(edge_index, x=x, edge_sh=edge_sh, edge_weight=edge_weight)
        return self.linear(out)
    
    def message(self, x_j, edge_sh, edge_weight):
        return self.tp(x_j, edge_sh, edge_weight)

class GaussianSmearing(nn.Module):
    """Gaussian smearing for radial basis functions"""
    
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)
        
    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class E3EquivariantGNN(nn.Module):
    """Multi-layer E(3) Equivariant Graph Neural Network"""
    
    def __init__(self, irreps_in="8x0e", irreps_hidden="16x0e+8x1o", 
                 irreps_out="8x0e", irreps_sh="1x0e+1x1o+1x2e", 
                 num_layers=4, max_radius=10.0):
        super().__init__()
        
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)
        
        # Input embedding
        self.embedding = o3.Linear(self.irreps_in, self.irreps_hidden)
        
        # E(3) equivariant layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = E3EquivariantLayer(
                irreps_in=self.irreps_hidden,
                irreps_out=self.irreps_hidden,
                irreps_sh=self.irreps_sh,
                max_radius=max_radius
            )
            self.layers.append(layer)
        
        # Output projection
        self.output = o3.Linear(self.irreps_hidden, self.irreps_out)
        
    def forward(self, x, pos, edge_index, batch=None):
        # Input embedding
        x = self.embedding(x)
        
        # Apply E(3) equivariant layers
        for layer in self.layers:
            x = layer(x, pos, edge_index) + x  # Residual connection
        
        # Output projection
        return self.output(x)