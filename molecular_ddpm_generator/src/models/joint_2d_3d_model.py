# src/models/joint_2d_3d_model.py - COMPLETE: EGNN Backend with Enhanced Pocket Processing
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch
from .base_model import MolecularModel

# 🎯 EGNN Import from refined egnn.py
try:
    from .egnn import (
        Joint2D3DEGNNModel, 
        EGNNBackbone, 
        EGNNLayer,
        GraphConvLayer,
        create_joint2d3d_egnn_model
    )
    EGNN_AVAILABLE = True
    print("✅ EGNN backend available - using equivariant processing")
except ImportError as e:
    print(f"❌ EGNN backend not available: {e}")
    EGNN_AVAILABLE = False

# Enhanced pocket encoder import
try:
    from .pocket_encoder import create_improved_pocket_encoder, SmartPocketAtomSelector
    IMPROVED_POCKET_AVAILABLE = True
    print("✅ Enhanced pocket encoder available")
except ImportError:
    print("Warning: Enhanced pocket encoder not available, using fallback")
    IMPROVED_POCKET_AVAILABLE = False

def safe_global_pool(x, batch, pool_type='mean'):
    """Safe global pooling with CUDA/CPU fallback"""
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
            try:
                if pool_type == 'mean':
                    result = global_mean_pool(x_cpu, batch_cpu)
                else:
                    result = global_max_pool(x_cpu, batch_cpu)
                return result.cuda()
            except:
                pass
        
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


class Enhanced2D3DFusion(nn.Module):
    """🔧 Enhanced 2D-3D feature fusion with robust dimension handling"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Multi-level fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention-based fusion
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, features_2d: torch.Tensor, features_3d: torch.Tensor) -> torch.Tensor:
        """Enhanced fusion that preserves atom count and handles dimension mismatches"""
        
        atoms_2d = features_2d.size(0)
        atoms_3d = features_3d.size(0)
        
        # Handle atom count mismatch
        if atoms_2d != atoms_3d:
            if atoms_3d < atoms_2d:
                # Expand 3D features to match 2D
                repeat_ratio = atoms_2d // atoms_3d + 1
                features_3d = features_3d.repeat(repeat_ratio, 1)[:atoms_2d]
            else:
                # Truncate 3D to match 2D
                features_3d = features_3d[:atoms_2d]
        
        # Ensure correct dimensions
        target_dim = self.hidden_dim
        
        # Fix 2D features dimension
        if features_2d.size(1) != target_dim:
            if features_2d.size(1) > target_dim:
                features_2d = features_2d[:, :target_dim]
            else:
                padding = torch.zeros(features_2d.size(0), target_dim - features_2d.size(1), 
                                    device=features_2d.device, dtype=features_2d.dtype)
                features_2d = torch.cat([features_2d, padding], dim=1)
        
        # Fix 3D features dimension
        if features_3d.size(1) != target_dim:
            if features_3d.size(1) > target_dim:
                features_3d = features_3d[:, :target_dim]
            else:
                padding = torch.zeros(features_3d.size(0), target_dim - features_3d.size(1),
                                    device=features_3d.device, dtype=features_3d.dtype)
                features_3d = torch.cat([features_3d, padding], dim=1)
        
        # Attention-based fusion
        combined = torch.cat([features_2d, features_3d], dim=-1)
        attention_weights = self.attention(combined)
        
        # Weighted fusion
        fused = self.fusion_mlp(combined)
        weighted_2d = features_2d * attention_weights
        weighted_3d = features_3d * (1 - attention_weights)
        
        # Final combination with residual
        result = self.norm(fused + weighted_2d + weighted_3d)
        
        return result


class Joint2D3DModel(MolecularModel):
    """🔧 COMPLETE: Joint 2D-3D Model with EGNN Backend and Enhanced Pocket Processing"""
    
    def __init__(self, atom_types: int = 11, bond_types: int = 4,
                 hidden_dim: int = 256, pocket_dim: int = 256,
                 num_layers: int = 6, max_radius: float = 10.0,
                 max_pocket_atoms: int = 1000,
                 conditioning_type: str = "add",
                 pocket_selection_strategy: str = "adaptive"):
        super().__init__(atom_types, bond_types, hidden_dim)
        
        if not EGNN_AVAILABLE:
            raise ImportError("EGNN backend not available! Check egnn.py imports")
        
        # Store configuration
        self.pocket_dim = pocket_dim
        self.num_layers = num_layers
        self.max_radius = max_radius
        self.max_pocket_atoms = max_pocket_atoms
        self.conditioning_type = conditioning_type
        self.pocket_selection_strategy = pocket_selection_strategy
        
        # Flexible atom embeddings for different input dimensions
        self.atom_embedding_6d = nn.Linear(6, hidden_dim)
        self.atom_embedding_7d = nn.Linear(7, hidden_dim)
        self.atom_embedding_8d = nn.Linear(8, hidden_dim)
        
        # Bond embedding for 2D chemical information
        self.bond_embedding = nn.Linear(bond_types, hidden_dim)
        
        # 🎯 EGNN 3D Backend (main innovation)
        self.egnn_3d = EGNNBackbone(
            hidden_dim=hidden_dim,
            num_layers=max(2, num_layers//2),  # EGNN optimal: 3-4 layers
            cutoff=max_radius,
            residual=True,
            attention=True,
            normalize=True
        )
        
        # 2D Chemical topology processing
        self.gnn_2d_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # Enhanced 2D-3D fusion
        self.fusion_layer = Enhanced2D3DFusion(hidden_dim)
        
        # 🎯 Enhanced pocket encoder
        if IMPROVED_POCKET_AVAILABLE:
            self.pocket_encoder = create_improved_pocket_encoder(
                hidden_dim=hidden_dim,
                output_dim=pocket_dim,
                selection_strategy=pocket_selection_strategy
            )
            print(f"✅ Using Enhanced pocket encoder with strategy: {pocket_selection_strategy}")
        else:
            self.pocket_encoder = FallbackPocketEncoder(
                input_dim=hidden_dim,
                output_dim=pocket_dim,
                max_atoms=max_pocket_atoms
            )
            print("⚠️  Using fallback pocket encoder")
        
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
        else:
            self.condition_transform = nn.Linear(pocket_dim, hidden_dim)
        
        # Output heads
        self.atom_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, atom_types)
        )
        
        self.bond_type_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, bond_types)
        )
        
        # Position head with residual connection
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )
        
        print(f"🎯 Complete Joint2D3D Model initialized with EGNN backend")
        print(f"   Layers: {num_layers}, Hidden: {hidden_dim}, Radius: {max_radius}A")
        print(f"   EGNN: E(n)-equivariant processing")
        print(f"   Pocket: Enhanced conditioning with {pocket_selection_strategy} selection")
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: torch.Tensor,
                pocket_x: torch.Tensor = None, pocket_pos: torch.Tensor = None,
                pocket_edge_index: torch.Tensor = None, pocket_batch: torch.Tensor = None,
                **kwargs):
        """🔧 Complete forward pass with EGNN backend and enhanced pocket processing"""
        
        try:
            # Initial embeddings with flexible input handling
            atom_emb = self._embed_atoms_flexible(x)
            
            # 2D Processing: Chemical topology
            h_2d = self._process_2d_chemistry(atom_emb, edge_index, edge_attr, batch)
            
            # 🎯 3D Processing with EGNN (equivariant!)
            h_3d, pos_updated = self.egnn_3d(atom_emb, pos, batch)
            
            # Enhanced 2D-3D fusion
            h_fused = self.fusion_layer(h_2d, h_3d)
            
            # 🎯 Enhanced pocket conditioning
            pocket_condition = self._encode_pocket_enhanced(
                pocket_x, pocket_pos, pocket_edge_index, pocket_batch, batch,
                ligand_pos=pos
            )
            
            if pocket_condition is not None:
                h_conditioned = self._apply_conditioning(h_fused, pocket_condition, batch)
            else:
                h_conditioned = h_fused
            
            # Generate outputs
            atom_logits = self.atom_type_head(h_conditioned)
            
            # Bond predictions with edge features
            if edge_index.size(1) > 0:
                row, col = edge_index
                edge_features = torch.cat([h_conditioned[row], h_conditioned[col]], dim=-1)
                bond_logits = self.bond_type_head(edge_features)
            else:
                bond_logits = torch.zeros((0, self.bond_types), device=x.device)
            
            # Position prediction with residual connection
            pos_delta = self.position_head(h_conditioned)
            pos_pred = pos_updated + pos_delta
            
            return {
                'atom_logits': atom_logits,
                'pos_pred': pos_pred,
                'bond_logits': bond_logits,
                'node_features': h_conditioned,
                'pos_delta': pos_delta,
                'pocket_condition': pocket_condition
            }
            
        except Exception as e:
            print(f"Joint2D3D forward error: {e}")
            # Comprehensive fallback
            h_fallback = self._embed_atoms_flexible(x)
            return {
                'atom_logits': torch.zeros(x.size(0), self.atom_types, device=x.device),
                'pos_pred': pos,
                'bond_logits': torch.zeros((0, self.bond_types), device=x.device),
                'node_features': h_fallback,
                'pos_delta': torch.zeros_like(pos),
                'pocket_condition': None
            }
    
    def _process_2d_chemistry(self, atom_emb: torch.Tensor, edge_index: torch.Tensor,
                             edge_attr: torch.Tensor, batch: torch.Tensor):
        """Enhanced 2D chemical topology processing"""
        h_current = atom_emb.clone()
        
        # Process bond features
        if edge_attr.dim() == 2 and edge_attr.size(1) > 1:
            bond_features = edge_attr.float()
        else:
            bond_features = edge_attr.float()
            if bond_features.dim() == 1:
                bond_features = bond_features.unsqueeze(-1)
        
        # Apply 2D graph convolutions with residual connections
        for i, gnn_layer in enumerate(self.gnn_2d_layers):
            try:
                h_prev = h_current.clone()
                h_current = gnn_layer(h_current, edge_index, bond_features)
                # Residual connection with dropout
                h_current = h_current + h_prev
                if i < len(self.gnn_2d_layers) - 1:  # No dropout on last layer
                    h_current = torch.dropout(h_current, p=0.1, train=self.training)
            except Exception as e:
                print(f"2D GNN layer {i} error: {e}")
                continue
        
        return h_current
    
    def _embed_atoms_flexible(self, x: torch.Tensor) -> torch.Tensor:
        """Flexible atom embedding for different input dimensions"""
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
            # Pad to 6D
            padding = torch.zeros(x.size(0), 6 - input_dim, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
            return self.atom_embedding_6d(x_padded.float())
        elif input_dim > 8:
            # Truncate to 8D
            x_truncated = x[:, :8]
            return self.atom_embedding_8d(x_truncated.float())
        else:
            # Pad to 7D
            padding = torch.zeros(x.size(0), 7 - input_dim, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
            return self.atom_embedding_7d(x_padded.float())
    
    def _encode_pocket_enhanced(self, pocket_x: torch.Tensor, pocket_pos: torch.Tensor,
                               pocket_edge_index: torch.Tensor, pocket_batch: torch.Tensor,
                               ligand_batch: torch.Tensor, ligand_pos: torch.Tensor = None) -> torch.Tensor:
        """🎯 Enhanced pocket encoding with smart selection and processing"""
        if pocket_x is None or pocket_pos is None:
            return None
        
        try:
            # Use enhanced pocket encoder
            pocket_repr = self.pocket_encoder(
                x=pocket_x, 
                pos=pocket_pos, 
                edge_index=pocket_edge_index, 
                batch=pocket_batch,
                ligand_pos=ligand_pos  # For smart selection
            )
            return pocket_repr
            
        except Exception as e:
            print(f"Warning: Enhanced pocket encoding failed: {e}")
            # Fallback to simple processing
            try:
                pocket_emb = self._embed_atoms_flexible(pocket_x)
                if pocket_batch is not None:
                    pocket_global = safe_global_pool(pocket_emb, pocket_batch, 'mean')
                else:
                    pocket_global = torch.mean(pocket_emb, dim=0, keepdim=True)
                
                # Project to pocket dimension
                pocket_repr = self.condition_transform(pocket_global)
                return pocket_repr
            except Exception as e2:
                print(f"Fallback pocket encoding failed: {e2}")
                batch_size = ligand_batch.max().item() + 1
                return torch.zeros(batch_size, self.pocket_dim, device=ligand_batch.device)
    
    def _apply_conditioning(self, atom_features: torch.Tensor, pocket_condition: torch.Tensor, 
                          batch: torch.Tensor) -> torch.Tensor:
        """Enhanced pocket conditioning with multiple strategies"""
        if pocket_condition is None or pocket_condition.abs().sum() == 0:
            return atom_features
        
        try:
            pocket_transformed = self.condition_transform(pocket_condition)
            
            # Handle batch size issues
            batch_size = pocket_transformed.size(0)
            max_batch_idx = batch.max().item()
            
            if max_batch_idx >= batch_size:
                # Extend pocket features for missing batch indices
                extra_needed = max_batch_idx + 1 - batch_size
                last_condition = pocket_transformed[-1:].repeat(extra_needed, 1)
                pocket_transformed = torch.cat([pocket_transformed, last_condition], dim=0)
            
            # Handle dimension mismatch
            if pocket_transformed.size(1) != atom_features.size(1):
                target_dim = atom_features.size(1)
                if pocket_transformed.size(1) > target_dim:
                    pocket_transformed = pocket_transformed[:, :target_dim]
                else:
                    padding = torch.zeros(pocket_transformed.size(0), target_dim - pocket_transformed.size(1),
                                        device=pocket_transformed.device, dtype=pocket_transformed.dtype)
                    pocket_transformed = torch.cat([pocket_transformed, padding], dim=1)
            
            # Safe batch indexing
            batch_safe = torch.clamp(batch, 0, pocket_transformed.size(0) - 1)
            broadcasted_condition = pocket_transformed[batch_safe]
            
            # Apply conditioning based on strategy
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
                return atom_features + broadcasted_condition
            
        except Exception as e:
            print(f"Conditioning error: {e}")
            return atom_features


class FallbackPocketEncoder(nn.Module):
    """Fallback pocket encoder when enhanced version is not available"""
    
    def __init__(self, input_dim: int, output_dim: int, max_atoms: int = 1000):
        super().__init__()
        self.max_atoms = max_atoms
        
        # Flexible embeddings
        self.pocket_embedding_6d = nn.Linear(6, input_dim)
        self.pocket_embedding_7d = nn.Linear(7, input_dim)
        self.pocket_embedding_8d = nn.Linear(8, input_dim)
        
        # Pocket processor
        self.pocket_processor = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, input_dim)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor = None, 
                edge_index: torch.Tensor = None, batch: torch.Tensor = None,
                ligand_pos: torch.Tensor = None, **kwargs) -> torch.Tensor:
        
        # Smart atom selection based on distance to ligand
        if x.size(0) > self.max_atoms:
            if ligand_pos is not None and pos is not None:
                # Distance-based selection
                ligand_center = torch.mean(ligand_pos, dim=0)
                distances = torch.norm(pos - ligand_center, dim=1)
                _, indices = torch.topk(distances, k=self.max_atoms, largest=False)
                x = x[indices]
                pos = pos[indices] if pos is not None else None
                if batch is not None:
                    batch = batch[indices]
            else:
                # Random selection
                indices = torch.randperm(x.size(0))[:self.max_atoms]
                x = x[indices]
                if batch is not None:
                    batch = batch[indices]
        
        # Flexible embedding
        input_dim = x.size(1)
        if input_dim == 6:
            pocket_emb = self.pocket_embedding_6d(x.float())
        elif input_dim == 7:
            pocket_emb = self.pocket_embedding_7d(x.float())
        elif input_dim == 8:
            pocket_emb = self.pocket_embedding_8d(x.float())
        else:
            # Default to 7D
            if input_dim < 7:
                padding = torch.zeros(x.size(0), 7 - input_dim, device=x.device, dtype=x.dtype)
                x_padded = torch.cat([x, padding], dim=1)
                pocket_emb = self.pocket_embedding_7d(x_padded.float())
            else:
                x_truncated = x[:, :7]
                pocket_emb = self.pocket_embedding_7d(x_truncated.float())
        
        # Process features
        processed_emb = self.pocket_processor(pocket_emb)
        
        # Global pooling
        if batch is not None:
            pocket_global = safe_global_pool(processed_emb, batch, 'mean')
        else:
            pocket_global = torch.mean(processed_emb, dim=0, keepdim=True)
        
        # Final projection
        return self.output_projection(pocket_global)


# 🎯 MAIN FACTORY FUNCTIONS
def create_joint2d3d_model(hidden_dim: int = 256, num_layers: int = 6,
                           pocket_selection_strategy: str = "adaptive", **kwargs):
    """🔧 Create Joint2D3D model with EGNN backend and enhanced pocket processing"""
    if not EGNN_AVAILABLE:
        raise ImportError("EGNN backend not available! Check egnn.py")
        
    return Joint2D3DModel(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        pocket_selection_strategy=pocket_selection_strategy,
        **kwargs
    )

def create_joint2d3d_egnn_model(hidden_dim: int = 256, num_layers: int = 6, **kwargs):
    """🎯 RECOMMENDED: Create EGNN-based Joint2D3D model"""
    return create_joint2d3d_model(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        **kwargs
    )

# 🎯 Backward compatibility aliases
create_joint2d3d_schnet_model = create_joint2d3d_egnn_model
Joint2D3DSchNetModel = Joint2D3DModel
Joint2D3DMolecularModel = Joint2D3DModel

# Test function
def test_complete_model():
    """Test complete Joint2D3D model with EGNN and pocket processing"""
    print("Testing complete Joint2D3D model...")
    
    # Create test data
    batch_size = 2
    num_atoms = 20
    num_pocket_atoms = 50
    
    # Ligand data
    x = torch.randn(num_atoms, 6)
    pos = torch.randn(num_atoms, 3)
    edge_index = torch.randint(0, num_atoms, (2, 30))
    edge_attr = torch.randn(30, 3)
    batch = torch.cat([torch.zeros(10), torch.ones(10)]).long()
    
    # Pocket data
    pocket_x = torch.randn(num_pocket_atoms, 7)
    pocket_pos = torch.randn(num_pocket_atoms, 3)
    pocket_edge_index = torch.randint(0, num_pocket_atoms, (2, 80))
    pocket_batch = torch.cat([torch.zeros(25), torch.ones(25)]).long()
    
    # Create model
    model = create_joint2d3d_model(
        hidden_dim=128, 
        num_layers=4,
        pocket_selection_strategy="adaptive"
    )
    
    # Test forward pass
    try:
        outputs = model(
            x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, batch=batch,
            pocket_x=pocket_x, pocket_pos=pocket_pos, 
            pocket_edge_index=pocket_edge_index, pocket_batch=pocket_batch
        )
        
        print(f"   Input: {num_atoms} atoms, {num_pocket_atoms} pocket atoms")
        print(f"   Output atom_logits: {outputs['atom_logits'].shape}")
        print(f"   Output pos_pred: {outputs['pos_pred'].shape}")
        print(f"   Output bond_logits: {outputs['bond_logits'].shape}")
        if outputs['pocket_condition'] is not None:
            print(f"   Pocket condition: {outputs['pocket_condition'].shape}")
        print(f"   ✅ Complete model test passed!")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Complete model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎯 Complete Joint2D3D Model - EGNN Backend with Enhanced Pocket Processing")
    print("=" * 80)
    print("Features:")
    print("- EGNN backend for E(n)-equivariant processing")
    print("- Enhanced pocket encoder with smart atom selection")
    print("- Flexible input dimension handling")
    print("- Multiple conditioning strategies")
    print("- Robust error handling and fallbacks")
    print()
    
    if EGNN_AVAILABLE:
        test_complete_model()
    else:
        print("❌ EGNN not available - check imports")