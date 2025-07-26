# src/models/joint_2d_3d_model.py - SchNet VERSION
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, SchNet
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.pool import radius_graph
from .base_model import MolecularModel

# SchNet Import
try:
    from torch_geometric.nn import SchNet
    SCHNET_AVAILABLE = True
    print("✅ SchNet available - using stable backend")
except ImportError:
    print("❌ SchNet not available. Install torch-geometric>=2.0")
    SCHNET_AVAILABLE = False

# Improved pocket encoder import
try:
    from .pocket_encoder import create_improved_pocket_encoder
    IMPROVED_POCKET_AVAILABLE = True
except ImportError:
    print("Warning: ImprovedProteinPocketEncoder not available, using fallback")
    IMPROVED_POCKET_AVAILABLE = False

def safe_global_pool(x, batch, pool_type='mean'):
    """Safe global pooling with CUDA fallback"""
    try:
        if pool_type == 'mean':
            return global_mean_pool(x, batch)
        else:
            return global_max_pool(x, batch)
    except Exception as e:
        # CPU fallback
        if x.is_cuda:
            x_cpu = x.cpu()
            batch_cpu = batch.cpu()
            if pool_type == 'mean':
                result = global_mean_pool(x_cpu, batch_cpu)
            else:
                result = global_max_pool(x_cpu, batch_cpu)
            return result.cuda()
        else:
            # Manual pooling fallback
            unique_batch = torch.unique(batch)
            pooled = []
            for b in unique_batch:
                mask = batch == b
                if pool_type == 'mean':
                    pooled.append(torch.mean(x[mask], dim=0))
                else:
                    pooled.append(torch.max(x[mask], dim=0)[0])
            return torch.stack(pooled)

class Joint2D3DSchNetModel(MolecularModel):
    """Joint 2D-3D Model with SchNet Backend for molecular generation"""
    
    def __init__(self, atom_types: int = 11, bond_types: int = 4,
                 hidden_dim: int = 256, pocket_dim: int = 256,
                 num_layers: int = 6, max_radius: float = 10.0,
                 max_pocket_atoms: int = 1000,
                 conditioning_type: str = "add",
                 pocket_selection_strategy: str = "adaptive"):
        super().__init__(atom_types, bond_types, hidden_dim)
        
        self.pocket_dim = pocket_dim
        self.num_layers = num_layers
        self.max_radius = max_radius
        self.max_pocket_atoms = max_pocket_atoms
        self.conditioning_type = conditioning_type
        self.pocket_selection_strategy = pocket_selection_strategy
        
        if not SCHNET_AVAILABLE:
            raise ImportError("SchNet not available. Install torch-geometric>=2.0")
        
        # Flexible embeddings for different input dimensions
        self.atom_embedding_6d = nn.Linear(6, hidden_dim)
        self.atom_embedding_7d = nn.Linear(7, hidden_dim)
        self.atom_embedding_8d = nn.Linear(8, hidden_dim)
        
        # Bond embedding for 2D chemical information
        self.bond_embedding = nn.Linear(bond_types, hidden_dim)
        
        # Pocket embeddings
        self.pocket_atom_embedding_6d = nn.Linear(6, hidden_dim)
        self.pocket_atom_embedding_7d = nn.Linear(7, hidden_dim)
        self.pocket_atom_embedding_8d = nn.Linear(8, hidden_dim)
        
        # SchNet for 3D geometric processing
        self.schnet_3d = SchNet(
            hidden_channels=hidden_dim,
            num_filters=hidden_dim,
            num_interactions=num_layers,
            num_gaussians=50,
            cutoff=max_radius,
            readout='add'
        ).cpu()
        
        # 2D Chemical topology processing
        self.gnn_2d_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # 2D-3D fusion
        self.fusion_layer = Enhanced2D3DFusion(hidden_dim)
        
        # Enhanced pocket encoder
        if IMPROVED_POCKET_AVAILABLE:
            self.pocket_encoder = create_improved_pocket_encoder(
                hidden_dim=hidden_dim,
                output_dim=pocket_dim,
                selection_strategy=pocket_selection_strategy
            )
            print(f"Using ImprovedProteinPocketEncoder with strategy: {pocket_selection_strategy}")
        else:
            self.pocket_encoder = SchNetPocketEncoder(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=pocket_dim,
                max_atoms=max_pocket_atoms
            )
            print("Using SchNet-based pocket encoder")
        
        # Conditioning module
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
        
        # Output heads
        self.atom_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, atom_types)
        )
        
        self.bond_type_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, bond_types)
        )
        
        # Position head
        self.position_head = nn.Linear(hidden_dim, 3)
        
        print(f"Joint2D3D Model initialized with SchNet backend")
        print(f"   Layers: {num_layers}, Hidden: {hidden_dim}, Radius: {max_radius}A")
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: torch.Tensor,
                pocket_x: torch.Tensor = None, pocket_pos: torch.Tensor = None,
                pocket_edge_index: torch.Tensor = None, pocket_batch: torch.Tensor = None,
                **kwargs):
        """Forward pass with SchNet processing"""
        
        try:
            # Initial embeddings
            atom_emb = self._embed_atoms_flexible(x)
            
            # 2D Processing: Chemical topology
            h_2d = self._process_2d_chemistry(atom_emb, edge_index, edge_attr, batch)
            
            # 3D Processing: SchNet geometric processing  
            h_3d, pos_updated = self._process_3d_schnet(atom_emb, pos, batch)
            
            # 2D-3D fusion
            h_fused = self.fusion_layer(h_2d, h_3d)
            
            # Pocket conditioning
            pocket_condition = self._encode_pocket_flexible(
                pocket_x, pocket_pos, pocket_edge_index, pocket_batch, batch,
                ligand_pos=pos
            )
            
            if pocket_condition is not None:
                h_conditioned = self._apply_conditioning(h_fused, pocket_condition, batch)
            else:
                h_conditioned = h_fused
            
            # Output predictions
            atom_logits = self.atom_type_head(h_conditioned)
            
            # Bond predictions
            if edge_index.size(1) > 0:
                row, col = edge_index
                edge_features = torch.cat([h_conditioned[row], h_conditioned[col]], dim=-1)
                bond_logits = self.bond_type_head(edge_features)
            else:
                bond_logits = torch.zeros((0, self.bond_types), device=x.device)
            
            # Position prediction
            pos_pred = pos_updated + self.position_head(h_conditioned)
            
            return {
                'atom_logits': atom_logits,
                'pos_pred': pos_pred,
                'bond_logits': bond_logits,
                'node_features': h_conditioned
            }
            
        except Exception as e:
            print(f"Joint2D3D SchNet forward error: {e}")
            # Fallback output
            h_fallback = self._embed_atoms_flexible(x)
            return {
                'atom_logits': torch.zeros(x.size(0), self.atom_types, device=x.device),
                'pos_pred': pos,
                'bond_logits': torch.zeros((0, self.bond_types), device=x.device),
                'node_features': h_fallback
            }
    
    def _process_2d_chemistry(self, atom_emb: torch.Tensor, edge_index: torch.Tensor,
                             edge_attr: torch.Tensor, batch: torch.Tensor):
        """2D chemical topology processing"""
        h_current = atom_emb.clone()
        
        # Process bond features
        if edge_attr.dim() == 2 and edge_attr.size(1) > 1:
            bond_features = edge_attr.float()
        else:
            bond_features = edge_attr.float()
            if bond_features.dim() == 1:
                bond_features = bond_features.unsqueeze(-1)
        
        # Apply 2D graph convolutions
        for i, gnn_layer in enumerate(self.gnn_2d_layers):
            try:
                h_prev = h_current.clone()
                h_current = gnn_layer(h_current, edge_index, bond_features)
                h_current = h_current + h_prev  # Residual connection
            except Exception as e:
                print(f"2D GNN layer {i} error: {e}")
                continue
        
        return h_current
    
    def _process_3d_schnet(self, atom_emb: torch.Tensor, pos: torch.Tensor, 
                        batch: torch.Tensor):
        """SchNet 3D processing - Always on CPU"""
        
        z = torch.argmax(atom_emb, dim=-1) + 1
        z = torch.clamp(z, 1, self.atom_types)
        
        try:
            # Save original device
            original_device = pos.device
            
            # Always move to CPU for SchNet processing
            z_cpu = z.cpu()
            pos_cpu = pos.cpu()
            batch_cpu = batch.cpu()
            
            # SchNet processing on CPU (model is already on CPU)
            h_3d_cpu = self.schnet_3d(z=z_cpu, pos=pos_cpu, batch=batch_cpu)
            
            # Move result back to original device
            h_3d = h_3d_cpu.to(original_device)
            pos_updated = pos.clone()
            
            return h_3d, pos_updated
            
        except Exception as e:
            print(f"SchNet CPU processing failed: {e}")
            print("Using atom embeddings fallback")
            return atom_emb, pos
    
    
    def _embed_atoms_flexible(self, x: torch.Tensor) -> torch.Tensor:
        """Flexible atom embedding"""
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got {x.dim()}D: {x.shape}")
        
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
        elif input_dim > 8:
            x_truncated = x[:, :8]
            return self.atom_embedding_8d(x_truncated.float())
        else:
            padding = torch.zeros(x.size(0), 8 - input_dim, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
            return self.atom_embedding_8d(x_padded.float())
    
    def _encode_pocket_flexible(self, pocket_x: torch.Tensor, pocket_pos: torch.Tensor,
                               pocket_edge_index: torch.Tensor, pocket_batch: torch.Tensor,
                               ligand_batch: torch.Tensor, ligand_pos: torch.Tensor = None) -> torch.Tensor:
        """Enhanced pocket encoding"""
        if pocket_x is None or pocket_pos is None:
            return None
        
        try:
            if IMPROVED_POCKET_AVAILABLE and hasattr(self.pocket_encoder, 'forward'):
                pocket_repr = self.pocket_encoder(
                    x=pocket_x, 
                    pos=pocket_pos, 
                    edge_index=pocket_edge_index, 
                    batch=pocket_batch,
                    ligand_pos=ligand_pos
                )
            else:
                pocket_repr = self.pocket_encoder(
                    pocket_x, pocket_pos, pocket_edge_index, pocket_batch
                )
            return pocket_repr
            
        except Exception as e:
            print(f"Warning: Pocket encoding failed: {e}")
            batch_size = ligand_batch.max().item() + 1
            return torch.zeros(batch_size, self.pocket_dim, device=ligand_batch.device)
    
    def _apply_conditioning(self, atom_features: torch.Tensor, pocket_condition: torch.Tensor, 
                          batch: torch.Tensor) -> torch.Tensor:
        """Apply pocket conditioning"""
        if pocket_condition is None or pocket_condition.abs().sum() == 0:
            return atom_features
        
        pocket_transformed = self.condition_transform(pocket_condition)
        
        batch_size = pocket_transformed.size(0)
        max_batch_idx = batch.max().item()
        
        if max_batch_idx >= batch_size:
            pocket_transformed = pocket_transformed[0:1].expand(max_batch_idx + 1, -1)
        
        broadcasted_condition = pocket_transformed[batch]
        
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


class GraphConvLayer(nn.Module):
    """Simple graph convolution layer for 2D chemistry"""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.edge_linear = nn.Linear(1, out_dim)
        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        """Simple message passing"""
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


class Enhanced2D3DFusion(nn.Module):
    """Enhanced 2D-3D feature fusion"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, features_2d: torch.Tensor, features_3d: torch.Tensor) -> torch.Tensor:
        """Fuse 2D and 3D features"""
        # Ensure same dimensions
        if features_2d.size(0) != features_3d.size(0):
            min_size = min(features_2d.size(0), features_3d.size(0))
            features_2d = features_2d[:min_size]
            features_3d = features_3d[:min_size]
        
        # Concatenate and fuse
        combined = torch.cat([features_2d, features_3d], dim=-1)
        fused = self.fusion_mlp(combined)
        
        # Residual connection with 2D features
        return self.norm(fused + features_2d)


class SchNetPocketEncoder(nn.Module):
    """SchNet-based pocket encoder fallback"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 max_atoms: int = 1000):
        super().__init__()
        self.max_atoms = max_atoms
        
        # Flexible embeddings
        self.pocket_embedding_6d = nn.Linear(6, hidden_dim)
        self.pocket_embedding_7d = nn.Linear(7, hidden_dim)
        self.pocket_embedding_8d = nn.Linear(8, hidden_dim)
        
        # Simple pocket processor
        self.pocket_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection
        self.global_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, pocket_x: torch.Tensor, pocket_pos: torch.Tensor,
                pocket_edge_index: torch.Tensor, pocket_batch: torch.Tensor) -> torch.Tensor:
        
        # Smart atom selection
        if pocket_x.size(0) > self.max_atoms:
            indices = torch.randperm(pocket_x.size(0))[:self.max_atoms]
            pocket_x = pocket_x[indices]
            pocket_pos = pocket_pos[indices]
            if pocket_batch is not None:
                pocket_batch = pocket_batch[indices]
        
        # Flexible embedding
        input_dim = pocket_x.size(1)
        if input_dim == 6:
            pocket_emb = self.pocket_embedding_6d(pocket_x.float())
        elif input_dim == 7:
            pocket_emb = self.pocket_embedding_7d(pocket_x.float())
        elif input_dim == 8:
            pocket_emb = self.pocket_embedding_8d(pocket_x.float())
        else:
            # Pad or truncate to 7D
            if input_dim < 7:
                padding = torch.zeros(pocket_x.size(0), 7 - input_dim, 
                                    device=pocket_x.device, dtype=pocket_x.dtype)
                pocket_x_padded = torch.cat([pocket_x, padding], dim=1)
                pocket_emb = self.pocket_embedding_7d(pocket_x_padded.float())
            else:
                pocket_x_truncated = pocket_x[:, :7]
                pocket_emb = self.pocket_embedding_7d(pocket_x_truncated.float())
        
        # Process features
        processed_emb = self.pocket_processor(pocket_emb)
        
        # Global pooling
        if pocket_batch is not None:
            global_mean = safe_global_pool(processed_emb, pocket_batch, 'mean')
            global_max = safe_global_pool(processed_emb, pocket_batch, 'max')
            global_combined = global_mean + global_max
        else:
            global_combined = torch.mean(processed_emb, dim=0, keepdim=True)
        
        # Final output
        pocket_repr = self.global_processor(global_combined)
        
        return pocket_repr


# Factory function
def create_joint2d3d_schnet_model(hidden_dim: int = 256, num_layers: int = 6,
                                 pocket_selection_strategy: str = "adaptive"):
    """Create Joint2D3D model with SchNet backend"""
    if not SCHNET_AVAILABLE:
        raise ImportError("SchNet not available! Update torch-geometric: pip install torch-geometric>=2.0")
        
    return Joint2D3DSchNetModel(
        atom_types=11,
        bond_types=4,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        pocket_selection_strategy=pocket_selection_strategy
    )

# Backward compatibility
create_joint2d3d_egnn_model = create_joint2d3d_schnet_model
Joint2D3DMolecularModel = Joint2D3DSchNetModel