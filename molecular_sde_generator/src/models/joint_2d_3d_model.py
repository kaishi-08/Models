# molecular_sde_generator/src/models/joint_2d_3d_model.py
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch
from .base_model import MolecularModel
from .e3_egnn import E3EquivariantGNN

class Joint2D3DMolecularModel(MolecularModel):
    """
    Joint 2D/3D molecular generation with SIMPLE but EFFECTIVE pocket conditioning
    
    Approach: Additive Conditioning (Tránh hoàn toàn dimension mismatch)
    - Pocket → Global representation vector
    - Broadcast to all ligand atoms  
    - Add to ligand features (không cần cross-attention)
    """
    
    def __init__(self, atom_types: int = 100, bond_types: int = 5,
                 hidden_dim: int = 128, pocket_dim: int = 256,
                 num_layers: int = 4, max_radius: float = 10.0,
                 max_pocket_atoms: int = 1000,
                 conditioning_type: str = "add"):  # "add", "concat", or "gated"
        super().__init__(atom_types, bond_types, hidden_dim)
        
        self.pocket_dim = pocket_dim
        self.num_layers = num_layers
        self.max_radius = max_radius
        self.max_pocket_atoms = max_pocket_atoms
        self.conditioning_type = conditioning_type
        
        # Feature dimensions
        self.atom_feature_dim = 8
        
        # === EMBEDDINGS ===
        self.atom_embedding = nn.Linear(self.atom_feature_dim, hidden_dim)
        self.bond_embedding = nn.Linear(1, hidden_dim)
        
        # === POCKET ENCODER (Simple but Effective) ===
        self.pocket_atom_embedding = nn.Linear(self.atom_feature_dim, hidden_dim)
        self.pocket_encoder = SimplePocketEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=pocket_dim,
            max_atoms=max_pocket_atoms
        )
        
        # === CONDITIONING MODULE ===
        if conditioning_type == "add":
            # Simple addition - requires pocket_dim == hidden_dim
            assert pocket_dim == hidden_dim, f"For 'add' conditioning, pocket_dim ({pocket_dim}) must equal hidden_dim ({hidden_dim})"
            self.condition_transform = nn.Identity()
            
        elif conditioning_type == "concat":
            # Concatenation approach
            self.condition_transform = nn.Linear(pocket_dim, hidden_dim)
            self.feature_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
            
        elif conditioning_type == "gated":
            # Gated conditioning (more sophisticated)
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
                pocket_edge_index: torch.Tensor = None, pocket_batch: torch.Tensor = None):
        """
        Forward pass với additive conditioning
        
        Args:
            x: Ligand atom features [N, feature_dim]
            pos: Ligand positions [N, 3]
            edge_index: Ligand bonds [2, E]
            edge_attr: Bond features [E, feature_dim]
            batch: Batch indices for ligand [N]
            pocket_*: Protein pocket data
        """
        
        # === STEP 1: LIGAND EMBEDDINGS ===
        atom_emb = self._embed_atoms(x)
        bond_emb = self._embed_bonds(edge_attr, edge_index)
        
        # === STEP 2: POCKET CONDITIONING ===
        pocket_condition = self._encode_pocket(pocket_x, pocket_pos, pocket_edge_index, pocket_batch, batch)
        
        # === STEP 3: APPLY CONDITIONING ===
        conditioned_atom_emb = self._apply_conditioning(atom_emb, pocket_condition)
        
        # === STEP 4: 2D GRAPH PROCESSING ===
        h_2d = conditioned_atom_emb
        for layer in self.graph_2d_net:
            h_2d = layer(h_2d, edge_index, bond_emb)
        
        # === STEP 5: 3D EQUIVARIANT PROCESSING ===
        h_3d = self.e3_3d_net(conditioned_atom_emb, pos, edge_index, batch)
        
        # === STEP 6: FEATURE FUSION ===
        h_fused = self.fusion_layer(torch.cat([h_2d, h_3d], dim=-1))
        
        # === STEP 7: PREDICTIONS ===
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
    
    def _embed_atoms(self, x: torch.Tensor) -> torch.Tensor:
        """Embed atom features safely"""
        if x.dim() == 2 and x.size(1) == self.atom_feature_dim:
            return self.atom_embedding(x.float())
        elif x.dim() == 2 and x.size(1) == 1:
            # Convert indices to one-hot if needed
            one_hot = torch.zeros(x.size(0), self.atom_feature_dim, device=x.device)
            one_hot[torch.arange(x.size(0)), x.squeeze(-1).long().clamp(0, self.atom_feature_dim-1)] = 1.0
            return self.atom_embedding(one_hot)
        else:
            raise ValueError(f"Unexpected x shape: {x.shape}")
    
    def _embed_bonds(self, edge_attr: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Embed bond features safely"""
        if edge_index.size(1) == 0:
            return torch.zeros((0, self.hidden_dim), device=edge_attr.device)
        
        if edge_attr.dim() == 2:
            bond_emb = self.bond_embedding(edge_attr[:, :1].float())
        else:
            bond_emb = self.bond_embedding(edge_attr.unsqueeze(-1).float())
        
        return bond_emb
    
    def _encode_pocket(self, pocket_x: torch.Tensor, pocket_pos: torch.Tensor,
                      pocket_edge_index: torch.Tensor, pocket_batch: torch.Tensor,
                      ligand_batch: torch.Tensor) -> torch.Tensor:
        """
        Encode pocket into global condition vectors
        
        Returns:
            condition_vectors: [batch_size, pocket_dim] - one vector per batch
        """
        if pocket_x is None or pocket_pos is None:
            # No pocket conditioning
            batch_size = ligand_batch.max().item() + 1
            return torch.zeros(batch_size, self.pocket_dim, device=ligand_batch.device)
        
        try:
            # Embed pocket atoms
            pocket_emb = self.pocket_atom_embedding(pocket_x.float())
            
            # Encode pocket
            pocket_repr = self.pocket_encoder(pocket_emb, pocket_pos, pocket_edge_index, pocket_batch)
            
            return pocket_repr
            
        except Exception as e:
            #print(f"Pocket encoding failed: {e}")
            batch_size = ligand_batch.max().item() + 1
            return torch.zeros(batch_size, self.pocket_dim, device=ligand_batch.device)
    
    def _apply_conditioning(self, atom_features: torch.Tensor, pocket_condition: torch.Tensor) -> torch.Tensor:
        """
        Apply pocket conditioning to atom features
        
        Args:
            atom_features: [N_atoms, hidden_dim]
            pocket_condition: [batch_size, pocket_dim]
            
        Returns:
            conditioned_features: [N_atoms, hidden_dim]
        """
        if pocket_condition.abs().sum() == 0:
            # No conditioning
            return atom_features
        
        # Transform pocket condition to match hidden_dim
        pocket_transformed = self.condition_transform(pocket_condition)  # [batch_size, hidden_dim]
        
        # Broadcast to all atoms (assuming single batch for now)
        # For multi-batch, you'd use: pocket_transformed[batch_indices]
        if pocket_transformed.size(0) == 1:
            # Single batch case
            broadcasted_condition = pocket_transformed.expand(atom_features.size(0), -1)
        else:
            # Multi-batch case - need batch indices
            # For now, use first batch
            broadcasted_condition = pocket_transformed[0:1].expand(atom_features.size(0), -1)
        
        # Apply conditioning based on type
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
    """
    Simple but effective pocket encoder
    Pocket atoms → Global representation vector
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, max_atoms: int = 1000):
        super().__init__()
        self.max_atoms = max_atoms
        
        # Simple MLP for atom-wise processing
        self.atom_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Global pooling and final projection
        self.global_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, pocket_features: torch.Tensor, pocket_pos: torch.Tensor,
                pocket_edge_index: torch.Tensor, pocket_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pocket_features: [N_pocket, input_dim]
            pocket_pos: [N_pocket, 3] 
            pocket_edge_index: [2, E_pocket]
            pocket_batch: [N_pocket]
            
        Returns:
            global_repr: [batch_size, output_dim]
        """
        
        # Smart selection if pocket too large
        if pocket_features.size(0) > self.max_atoms:
            indices = torch.randperm(pocket_features.size(0))[:self.max_atoms]
            pocket_features = pocket_features[indices]
            pocket_batch = pocket_batch[indices]
        
        # Process each pocket atom
        processed_features = self.atom_processor(pocket_features)
        
        # Global pooling per batch
        # Combine mean and max pooling for richer representation
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
            nn.Linear(in_dim * 2 + in_dim, out_dim),  # node1 + node2 + edge
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