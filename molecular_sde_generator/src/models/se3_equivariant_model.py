# src/models/se3_equivariant_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

class SE3EquivariantLayer(nn.Module):
    """
    SE(3) equivariant layer inspired by GCDM and DiffSBDD
    Handles both scalar (invariant) and vector (equivariant) features
    """
    
    def __init__(self, scalar_dim: int, vector_dim: int, hidden_dim: int):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.hidden_dim = hidden_dim
        
        # Scalar feature processing
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Vector feature processing  
        self.vector_net = VectorLinear(vector_dim, hidden_dim)
        
        # Interaction between scalar and vector
        self.scalar_vector_interaction = nn.Linear(hidden_dim, hidden_dim)
        self.vector_scalar_interaction = VectorLinear(hidden_dim, scalar_dim)
        
        # Output projections
        self.scalar_out = nn.Linear(hidden_dim, scalar_dim)
        self.vector_out = VectorLinear(hidden_dim, vector_dim)
        
    def forward(self, scalar_features: torch.Tensor, 
                vector_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            scalar_features: [N, scalar_dim] - SE(3) invariant features
            vector_features: [N, vector_dim, 3] - SE(3) equivariant features
        """
        
        # Process scalar features
        scalar_hidden = self.scalar_net(scalar_features)
        
        # Process vector features  
        vector_hidden = self.vector_net(vector_features)
        
        # Scalar-vector interactions
        # Vector norm contributes to scalar features (maintains invariance)
        vector_norms = torch.norm(vector_hidden, dim=-1)  # [N, hidden_dim]
        scalar_enhanced = scalar_hidden + self.scalar_vector_interaction(vector_norms)
        
        # Scalar features modulate vector magnitudes (maintains equivariance)
        scalar_weights = self.vector_scalar_interaction.get_scalar_weights(scalar_hidden)
        vector_enhanced = vector_hidden * scalar_weights.unsqueeze(-1)
        
        # Output
        scalar_out = self.scalar_out(scalar_enhanced) + scalar_features
        vector_out = self.vector_out(vector_enhanced) + vector_features
        
        return scalar_out, vector_out

class VectorLinear(nn.Module):
    """Linear layer for vector features that maintains SE(3) equivariance"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight matrix for vector transformation
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, vector_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vector_input: [N, in_features, 3]
        Returns:
            vector_output: [N, out_features, 3]
        """
        # Linear transformation on vector features
        # Each 3D vector is transformed independently
        return torch.einsum('nij,oi->noj', vector_input, self.weight)
    
    def get_scalar_weights(self, scalar_input: torch.Tensor) -> torch.Tensor:
        """Get scalar weights for vector modulation"""
        return torch.sigmoid(nn.Linear(scalar_input.size(-1), self.out_features)(scalar_input))

class SE3TransformerLayer(nn.Module):
    """
    SE(3) Transformer layer with attention mechanism
    Based on SE(3)-Transformer architecture
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0
        
        # Query, key, value projections for scalars
        self.q_scalar = nn.Linear(hidden_dim, hidden_dim)
        self.k_scalar = nn.Linear(hidden_dim, hidden_dim)
        self.v_scalar = nn.Linear(hidden_dim, hidden_dim)
        
        # Vector projections
        self.q_vector = VectorLinear(hidden_dim // 3, hidden_dim // 3)
        self.k_vector = VectorLinear(hidden_dim // 3, hidden_dim // 3)
        self.v_vector = VectorLinear(hidden_dim // 3, hidden_dim // 3)
        
        # Output projections
        self.out_scalar = nn.Linear(hidden_dim, hidden_dim)
        self.out_vector = VectorLinear(hidden_dim // 3, hidden_dim // 3)
        
        self.layer_norm_scalar = nn.LayerNorm(hidden_dim)
        self.layer_norm_vector = VectorLayerNorm(hidden_dim // 3)
        
    def forward(self, scalar_features: torch.Tensor,
                vector_features: torch.Tensor,
                edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """SE(3) equivariant attention"""
        
        batch_size, seq_len = scalar_features.shape[:2]
        
        # Multi-head attention for scalars
        q_s = self.q_scalar(scalar_features).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_s = self.k_scalar(scalar_features).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v_s = self.v_scalar(scalar_features).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Attention weights (SE(3) invariant)
        attention_weights = torch.einsum('bqhd,bkhd->bhqk', q_s, k_s) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Apply attention to scalars
        attended_scalar = torch.einsum('bhqk,bkhd->bqhd', attention_weights, v_s)
        attended_scalar = attended_scalar.contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Apply same attention to vectors (maintains equivariance)
        vector_reshaped = vector_features.view(batch_size, seq_len, -1, 3)
        attended_vector = torch.einsum('bhqk,bkd...->bqhd...', attention_weights, vector_reshaped)
        attended_vector = attended_vector.mean(dim=2)  # Average over heads
        
        # Output projections
        scalar_out = self.out_scalar(attended_scalar)
        vector_out = self.out_vector(attended_vector)
        
        # Residual connections and layer norm
        scalar_out = self.layer_norm_scalar(scalar_out + scalar_features)
        vector_out = self.layer_norm_vector(vector_out + vector_features)
        
        return scalar_out, vector_out

class VectorLayerNorm(nn.Module):
    """Layer normalization for vector features"""
    
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        
    def forward(self, vector_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vector_input: [N, normalized_shape, 3]
        """
        # Compute norms
        norms = torch.norm(vector_input, dim=-1, keepdim=True)  # [N, normalized_shape, 1]
        
        # Normalize
        normalized = vector_input / (norms + 1e-8)
        
        # Apply learnable weights
        return normalized * self.weight.unsqueeze(-1).unsqueeze(0)

class SE3EquivariantMolecularModel(nn.Module):
    """
    Complete SE(3) equivariant model for molecular generation
    Integrates scalar and vector features with proper equivariance
    """
    
    def __init__(self, 
                 atom_types: int = 11,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8):
        super().__init__()
        
        self.atom_types = atom_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input embeddings
        self.atom_embedding = nn.Embedding(atom_types, hidden_dim)
        self.position_embedding = VectorLinear(1, hidden_dim // 3)
        
        # SE(3) equivariant layers
        self.se3_layers = nn.ModuleList([
            SE3EquivariantLayer(hidden_dim, hidden_dim // 3, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # SE(3) transformer layers
        self.transformer_layers = nn.ModuleList([
            SE3TransformerLayer(hidden_dim, num_heads)
            for _ in range(2)  # Add 2 transformer layers
        ])
        
        # Output heads
        self.atom_type_head = nn.Linear(hidden_dim, atom_types)
        self.position_head = VectorLinear(hidden_dim // 3, 1)
        
        # Chirality prediction (important for drug molecules)
        self.chirality_head = nn.Linear(hidden_dim, 2)  # R/S
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor,
                edge_index: torch.Tensor, batch: torch.Tensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [N] - atom types
            pos: [N, 3] - positions
            edge_index: [2, E] - graph connectivity
            batch: [N] - batch assignment
        """
        
        # Initial embeddings
        scalar_features = self.atom_embedding(x.long())  # [N, hidden_dim]
        vector_features = self.position_embedding(pos.unsqueeze(-2))  # [N, hidden_dim//3, 3]
        
        # SE(3) equivariant processing
        for layer in self.se3_layers:
            scalar_features, vector_features = layer(scalar_features, vector_features)
        
        # SE(3) transformer processing
        for transformer in self.transformer_layers:
            scalar_features, vector_features = transformer(
                scalar_features, vector_features, edge_index
            )
        
        # Output predictions
        atom_logits = self.atom_type_head(scalar_features)
        position_delta = self.position_head(vector_features).squeeze(-2)  # [N, 3]
        
        # Predict new positions
        pos_pred = pos + position_delta
        
        # Chirality prediction
        chirality_logits = self.chirality_head(scalar_features)
        
        return {
            'atom_logits': atom_logits,
            'pos_pred': pos_pred,
            'position_delta': position_delta,
            'chirality_logits': chirality_logits,
            'scalar_features': scalar_features,
            'vector_features': vector_features
        }
    
    def get_equivariance_loss(self, scalar_features: torch.Tensor,
                             vector_features: torch.Tensor,
                             pos: torch.Tensor) -> torch.Tensor:
        """
        Compute loss to enforce SE(3) equivariance
        """
        # Generate random rotation and translation
        batch_size = scalar_features.size(0)
        device = scalar_features.device
        
        # Random rotation matrix
        angles = torch.randn(3, device=device) * 0.1
        R = self._rotation_matrix(angles)
        
        # Random translation
        t = torch.randn(3, device=device) * 0.1
        
        # Transform positions
        pos_transformed = torch.matmul(pos, R.T) + t
        
        # Forward pass with transformed positions
        with torch.no_grad():
            outputs_transformed = self.forward(
                torch.arange(self.atom_types, device=device)[:scalar_features.size(0)],
                pos_transformed,
                torch.zeros(2, 0, dtype=torch.long, device=device),
                torch.zeros(scalar_features.size(0), dtype=torch.long, device=device)
            )
        
        # Scalar features should be invariant
        scalar_invariance_loss = F.mse_loss(
            scalar_features, outputs_transformed['scalar_features']
        )
        
        # Vector features should transform equivariantly
        vector_transformed_expected = torch.matmul(vector_features, R.T)
        vector_equivariance_loss = F.mse_loss(
            vector_transformed_expected, outputs_transformed['vector_features']
        )
        
        return scalar_invariance_loss + vector_equivariance_loss
    
    def _rotation_matrix(self, angles: torch.Tensor) -> torch.Tensor:
        """Generate 3D rotation matrix from Euler angles"""
        x, y, z = angles
        
        # Rotation matrices for each axis
        Rx = torch.tensor([[1, 0, 0],
                          [0, torch.cos(x), -torch.sin(x)],
                          [0, torch.sin(x), torch.cos(x)]], device=angles.device)
        
        Ry = torch.tensor([[torch.cos(y), 0, torch.sin(y)],
                          [0, 1, 0],
                          [-torch.sin(y), 0, torch.cos(y)]], device=angles.device)
        
        Rz = torch.tensor([[torch.cos(z), -torch.sin(z), 0],
                          [torch.sin(z), torch.cos(z), 0],
                          [0, 0, 1]], device=angles.device)
        
        return Rz @ Ry @ Rx