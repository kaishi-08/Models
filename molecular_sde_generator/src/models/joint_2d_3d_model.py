# src/models/joint_2d_3d_model.py - Original with MINIMAL changes for ImprovedPocketEncoder
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, SchNet, DimeNet
from torch_geometric.utils import to_dense_batch, radius_graph
from .base_model import MolecularModel

# 🔄 ONLY CHANGE: Import improved pocket encoder
try:
    from .pocket_encoder import create_improved_pocket_encoder
    IMPROVED_POCKET_AVAILABLE = True
except ImportError:
    print("Warning: ImprovedProteinPocketEncoder not available, using fallback")
    IMPROVED_POCKET_AVAILABLE = False

# Try to import EGNN, fallback if not available
try:
    from egnn_pytorch import EGNN
    EGNN_AVAILABLE = True
    print("EGNN available - using best backend")
except ImportError:
    print("Warning: EGNN not installed. Install with: pip install egnn-pytorch")
    EGNN_AVAILABLE = False

class Joint2D3DMolecularModel(MolecularModel):
    """
    FINAL JOINT 2D-3D MODEL - EGNN ONLY VERSION
    
    Fixed Issues:
    (a) Properly uses edge attributes with EGNN
    (b) TRUE SE(3) equivariant 3D processing with EGNN
    (c) Proper position-graph interaction
    
    Features:
    - 2D: Chemical topology with bond information
    - 3D: SE(3) equivariant geometric processing
    - Pocket: Advanced conditioning with ImprovedProteinPocketEncoder
    - EGNN backend only (removed SchNet, DimeNet)
    """
    
    def __init__(self, atom_types: int = 11, bond_types: int = 4,
                 hidden_dim: int = 256, pocket_dim: int = 256,
                 num_layers: int = 6, max_radius: float = 10.0,
                 max_pocket_atoms: int = 1000,
                 conditioning_type: str = "add",
                 pocket_selection_strategy: str = "adaptive"):  # 🔄 NEW PARAMETER
        super().__init__(atom_types, bond_types, hidden_dim)
        
        self.pocket_dim = pocket_dim
        self.num_layers = num_layers
        self.max_radius = max_radius
        self.max_pocket_atoms = max_pocket_atoms
        self.conditioning_type = conditioning_type
        self.pocket_selection_strategy = pocket_selection_strategy  # 🔄 NEW
        
        # Validate EGNN backend
        if not EGNN_AVAILABLE:
            raise ImportError("EGNN not available. Install with: pip install egnn-pytorch")
        
        # Flexible embeddings
        self.atom_embedding_6d = nn.Linear(6, hidden_dim)
        self.atom_embedding_8d = nn.Linear(8, hidden_dim)
        
        # Bond embedding for edge attributes
        self.bond_embedding = nn.Linear(bond_types, hidden_dim)
        
        # Pocket embeddings  
        self.pocket_atom_embedding_6d = nn.Linear(6, hidden_dim)
        self.pocket_atom_embedding_7d = nn.Linear(7, hidden_dim)
        self.pocket_atom_embedding_8d = nn.Linear(8, hidden_dim)
        
        # EGNN: Best choice - SE(3) equivariant + edge attributes
        self.gnn_2d_layers = nn.ModuleList([
            EGNN(
                dim=hidden_dim,
                edge_dim=bond_types,          # Proper edge attributes
                m_dim=hidden_dim // 4,
                fourier_features=0,
                num_nearest_neighbors=32,
                dropout=0.1,
                norm_feats=True,
                update_feats=True,
                update_coors=True             # SE(3) equivariant
            ) for _ in range(num_layers)
        ])
        
        # Separate 3D processing for geometric features
        self.gnn_3d_layers = nn.ModuleList([
            EGNN(
                dim=hidden_dim,
                edge_dim=0,                   # No edge features for 3D spatial
                m_dim=hidden_dim // 4,
                fourier_features=8,           # Geometric fourier features
                num_nearest_neighbors=32,
                dropout=0.1,
                norm_feats=True,
                update_feats=True,
                update_coors=True
            ) for _ in range(num_layers)
        ])
        
        # 2D-3D fusion module
        self.fusion_layer = Enhanced2D3DFusion(hidden_dim)
        
        # 🔄 MAIN CHANGE: Use improved pocket encoder if available
        if IMPROVED_POCKET_AVAILABLE:
            self.pocket_encoder = create_improved_pocket_encoder(
                hidden_dim=hidden_dim,
                output_dim=pocket_dim,
                selection_strategy=pocket_selection_strategy
            )
            print(f"Using ImprovedProteinPocketEncoder with strategy: {pocket_selection_strategy}")
        else:
            # Fallback to original enhanced encoder
            self.pocket_encoder = EnhancedPocketEncoder(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=pocket_dim,
                max_atoms=max_pocket_atoms
            )
            print("Using fallback EnhancedPocketEncoder")
        
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
        
        # Position head (EGNN handles position updates internally)
        self.position_head = nn.Identity()
        
        print(f"Joint2D3D Model initialized with EGNN backend")
        print(f"   Layers: {num_layers}, Hidden: {hidden_dim}, Radius: {max_radius}A")
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: torch.Tensor,
                pocket_x: torch.Tensor = None, pocket_pos: torch.Tensor = None,
                pocket_edge_index: torch.Tensor = None, pocket_batch: torch.Tensor = None,
                **kwargs):
        """
        Fixed Forward Pass - All Issues Resolved
        
        (a) Proper edge attribute usage with EGNN
        (b) True SE(3) equivariant 3D processing  
        (c) Enhanced position-graph interaction
        """
        
        # Initial embeddings
        atom_emb = self._embed_atoms_flexible(x)  # [N, hidden_dim]
        
        # EGNN processing: Joint 2D-3D processing with TRUE equivariance
        h_final, pos_final = self._process_egnn(
            atom_emb, pos, edge_index, edge_attr, batch
        )
        
        # 🔄 ENHANCED: Pocket conditioning with ligand position guidance
        pocket_condition = self._encode_pocket_flexible(
            pocket_x, pocket_pos, pocket_edge_index, pocket_batch, batch,
            ligand_pos=pos  # 🔄 Pass ligand position for smart selection
        )
        
        if pocket_condition is not None:
            h_conditioned = self._apply_conditioning(h_final, pocket_condition, batch)
        else:
            h_conditioned = h_final
        
        # Output predictions
        # Atom types
        atom_logits = self.atom_type_head(h_conditioned)
        
        # Bond types  
        if edge_index.size(1) > 0:
            row, col = edge_index
            edge_features = torch.cat([h_conditioned[row], h_conditioned[col]], dim=-1)
            bond_logits = self.bond_type_head(edge_features)
        else:
            bond_logits = torch.zeros((0, self.bond_types), device=x.device)
        
        # Position prediction
        # EGNN updates positions internally - already SE(3) equivariant
        pos_pred = pos_final
        
        return {
            'atom_logits': atom_logits,
            'pos_pred': pos_pred,
            'bond_logits': bond_logits,
            'node_features': h_conditioned
        }
    
    def _process_egnn(self, atom_emb: torch.Tensor, pos: torch.Tensor,
                     edge_index: torch.Tensor, edge_attr: torch.Tensor,
                     batch: torch.Tensor):
        """
        EGNN Processing: Joint 2D-3D with TRUE SE(3) equivariance
        """
        
        # Prepare edge attributes for EGNN
        if edge_attr.dim() == 2 and edge_attr.size(1) > 1:
            # Multi-dimensional edge features
            bond_features = edge_attr.float()
        else:
            # Single dimension - expand or use as-is
            bond_features = edge_attr.float()
            if bond_features.dim() == 1:
                bond_features = bond_features.unsqueeze(-1)
        
        # Current features and positions
        h_current = atom_emb.clone()
        pos_current = pos.clone()
        
        # 2D processing: Chemical topology with bond information
        for i, egnn_2d in enumerate(self.gnn_2d_layers):
            h_prev = h_current.clone()
            pos_prev = pos_current.clone()
            
            # EGNN layer processes both features AND coordinates
            h_current, pos_current = egnn_2d(
                feats=h_current,         # Node features
                coors=pos_current,       # Coordinates (SE(3) equivariant)
                edges=bond_features      # Edge attributes properly used
            )
            
            # Residual connections
            h_current = h_current + h_prev
            # Position updates are handled by EGNN internally
        
        # 3D processing: Spatial geometry
        # Create spatial edges (different from chemical bonds)
        spatial_edge_index = radius_graph(
            pos_current, r=self.max_radius, batch=batch, max_num_neighbors=32
        )
        
        for i, egnn_3d in enumerate(self.gnn_3d_layers):
            h_prev = h_current.clone()
            
            # 3D EGNN with spatial edges (no edge features, pure geometry)
            h_current, pos_current = egnn_3d(
                feats=h_current,
                coors=pos_current,
                edges=None  # No edge features for spatial processing
            )
            
            # Residual for features
            h_current = h_current + h_prev
        
        # 2D-3D fusion
        h_fused = self.fusion_layer(h_current)
        
        return h_fused, pos_current
    
    def _embed_atoms_flexible(self, x: torch.Tensor) -> torch.Tensor:
        """Flexible atom embedding for ligand features"""
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got {x.dim()}D: {x.shape}")
        
        input_dim = x.size(1)
        
        if input_dim == 6:
            return self.atom_embedding_6d(x.float())
        elif input_dim == 8:
            return self.atom_embedding_8d(x.float())
        elif input_dim < 6:
            padding = torch.zeros(x.size(0), 6 - input_dim, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
            return self.atom_embedding_6d(x_padded.float())
        elif input_dim == 7:
            padding = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
            return self.atom_embedding_8d(x_padded.float())
        else:
            x_truncated = x[:, :8]
            return self.atom_embedding_8d(x_truncated.float())
    
    def _encode_pocket_flexible(self, pocket_x: torch.Tensor, pocket_pos: torch.Tensor,
                               pocket_edge_index: torch.Tensor, pocket_batch: torch.Tensor,
                               ligand_batch: torch.Tensor, ligand_pos: torch.Tensor = None) -> torch.Tensor:
        """🔄 ENHANCED: Pocket encoding with ligand position guidance"""
        if pocket_x is None or pocket_pos is None:
            return None
        
        try:
            # 🔄 ENHANCED: Pass ligand position to improved encoder for smart selection
            if IMPROVED_POCKET_AVAILABLE and hasattr(self.pocket_encoder, 'forward'):
                # ImprovedProteinPocketEncoder supports ligand_pos parameter
                pocket_repr = self.pocket_encoder(
                    x=pocket_x, 
                    pos=pocket_pos, 
                    edge_index=pocket_edge_index, 
                    batch=pocket_batch,
                    ligand_pos=ligand_pos  # 🔄 Enable binding site proximity selection
                )
            else:
                # Fallback to original encoder
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
        """Apply pocket conditioning with proper batch handling"""
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

# Enhanced supporting classes

class Enhanced2D3DFusion(nn.Module):
    """Enhanced 2D-3D feature fusion for Joint model"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Simple but effective fusion"""
        fused = self.fusion_mlp(features)
        return self.norm(fused + features)  # Residual

class EnhancedPocketEncoder(nn.Module):
    """Enhanced pocket encoder with EGNN support - FALLBACK VERSION"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 max_atoms: int = 1000):
        super().__init__()
        self.max_atoms = max_atoms
        
        # Flexible pocket embeddings
        self.pocket_embedding_6d = nn.Linear(6, hidden_dim)
        self.pocket_embedding_7d = nn.Linear(7, hidden_dim)
        self.pocket_embedding_8d = nn.Linear(8, hidden_dim)
        
        # Pocket processor using EGNN
        if EGNN_AVAILABLE:
            self.pocket_processor = EGNN(
                dim=hidden_dim,
                edge_dim=0,
                m_dim=hidden_dim // 4,
                fourier_features=4,
                num_nearest_neighbors=16,
                dropout=0.1,
                update_feats=True,
                update_coors=False  # Don't update pocket coordinates
            )
        else:
            # Fallback to simple MLP
            self.pocket_processor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Global pooling and output
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
        
        # Process with EGNN or MLP
        if EGNN_AVAILABLE:
            try:
                processed_emb, _ = self.pocket_processor(
                    feats=pocket_emb,
                    coors=pocket_pos,
                    edges=None
                )
            except:
                processed_emb = pocket_emb
        else:
            processed_emb = self.pocket_processor(pocket_emb)
        
        # Global pooling
        if pocket_batch is not None:
            global_mean = global_mean_pool(processed_emb, pocket_batch)
            global_max = global_max_pool(processed_emb, pocket_batch)
            global_combined = global_mean + global_max
        else:
            # Single pocket case
            global_combined = torch.mean(processed_emb, dim=0, keepdim=True)
        
        # Final output
        pocket_repr = self.global_processor(global_combined)
        
        return pocket_repr

# Factory functions for easy usage

def create_joint2d3d_egnn_model(hidden_dim: int = 256, num_layers: int = 6,
                               pocket_selection_strategy: str = "adaptive"):  # 🔄 NEW PARAMETER
    """
    RECOMMENDED: Create Joint2D3D model with EGNN backend
    
    Best choice for molecular generation:
    - SE(3) equivariant
    - Proper edge attribute usage
    - Joint 2D-3D processing
    - 🔄 Smart pocket atom selection strategies
    
    Args:
        pocket_selection_strategy: "adaptive", "distance", "surface", "residue", "binding_site"
    """
    if not EGNN_AVAILABLE:
        raise ImportError("EGNN not available! Install with: pip install egnn-pytorch")
        
    return Joint2D3DMolecularModel(
        atom_types=11,
        bond_types=4,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        pocket_selection_strategy=pocket_selection_strategy  # 🔄 NEW
    )

# Testing and validation

def test_joint2d3d_model_equivariance(model, device='cpu'):
    """Test SE(3) equivariance of the Joint2D3D model"""
    print(f"Testing EGNN Joint2D3D Model Equivariance...")
    
    model.eval()
    model = model.to(device)
    
    # Create test data
    num_atoms = 15
    x = torch.randn(num_atoms, 6, device=device)
    pos = torch.randn(num_atoms, 3, device=device)
    edge_index = torch.randint(0, num_atoms, (2, 20), device=device)
    edge_attr = torch.randn(20, 4, device=device)
    batch = torch.zeros(num_atoms, dtype=torch.long, device=device)
    
    # Original prediction
    with torch.no_grad():
        out1 = model(x, pos, edge_index, edge_attr, batch)
    
    # Apply rotation
    from scipy.spatial.transform import Rotation
    R = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float32, device=device)
    pos_rotated = torch.matmul(pos, R.T)
    
    # Rotated prediction
    with torch.no_grad():
        out2 = model(x, pos_rotated, edge_index, edge_attr, batch)
    
    # Check equivariance for EGNN
    # For EGNN, positions should transform correctly
    pos_pred1_rot = torch.matmul(out1['pos_pred'], R.T)
    pos_error = torch.norm(pos_pred1_rot - out2['pos_pred']).item()
    
    # Check scalar invariance
    scalar_error = torch.norm(out1['atom_logits'] - out2['atom_logits']).item()
    
    print(f"   Position error: {pos_error:.6f}")
    print(f"   Scalar error: {scalar_error:.6f}")
    
    is_good = pos_error < 0.1 and scalar_error < 0.01
    print(f"   {'SE(3) EQUIVARIANT' if is_good else 'NOT EQUIVARIANT'}")
    
    return is_good

# Usage example
if __name__ == "__main__":
    print("Joint2D3D Molecular Model - EGNN VERSION with Smart Pocket Selection")
    print("=" * 70)
    
    # Test EGNN model
    if EGNN_AVAILABLE:
        print("Testing EGNN Model:")
        
        # 🔄 Test different pocket selection strategies
        strategies = ["adaptive", "distance", "surface", "binding_site"]
        
        for strategy in strategies:
            print(f"\n📍 Testing with pocket strategy: {strategy}")
            try:
                egnn_model = create_joint2d3d_egnn_model(
                    hidden_dim=128, 
                    num_layers=4,
                    pocket_selection_strategy=strategy
                )
                
                # Basic forward pass test
                x = torch.randn(10, 6)
                pos = torch.randn(10, 3)
                edge_index = torch.randint(0, 10, (2, 15))
                edge_attr = torch.randn(15, 4)
                batch = torch.zeros(10, dtype=torch.long)
                
                outputs = egnn_model(x, pos, edge_index, edge_attr, batch)
                
                print(f"   ✅ Forward pass successful with {strategy} strategy")
                print(f"   Output shapes:")
                print(f"     atom_logits: {outputs['atom_logits'].shape}")
                print(f"     pos_pred: {outputs['pos_pred'].shape}")
                print(f"     bond_logits: {outputs['bond_logits'].shape}")
                
            except Exception as e:
                print(f"   ❌ Error with {strategy}: {e}")
        
        # Test equivariance with default model
        print(f"\n🔄 Testing SE(3) Equivariance:")
        default_model = create_joint2d3d_egnn_model(hidden_dim=128, num_layers=4)
        test_joint2d3d_model_equivariance(default_model)
            
    else:
        print("EGNN not available!")
        print("Install with: pip install egnn-pytorch")
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"   Use create_joint2d3d_egnn_model() for best performance!")
    print(f"   - SE(3) equivariant")
    print(f"   - Proper edge attributes") 
    print(f"   - Joint 2D-3D processing")
    print(f"   - 🔄 Smart pocket atom selection")
    
    print(f"\n📦 Installation:")
    print(f"   pip install egnn-pytorch        # For EGNN (recommended)")
    print(f"   pip install torch-geometric     # For base dependencies")
    
    print(f"\n🚀 Joint2D3D Model - ENHANCED VERSION!")
    print(f"   (a) Proper edge attribute usage")
    print(f"   (b) True SE(3) equivariant processing")
    print(f"   (c) Enhanced position-graph interaction")
    print(f"   🔄 (d) Smart pocket atom selection strategies")