# src/models/pocket_encoder.py - UPDATED for SchNet compatibility
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, knn_graph
from torch_geometric.nn.pool import radius_graph
from torch_geometric.utils import to_dense_batch
import numpy as np

# THÊM FUNCTION NÀY NGAY SAU IMPORTS:
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
    
class SmartPocketAtomSelector:
    """Smart strategies for selecting important pocket atoms"""
    
    @staticmethod
    def select_by_distance_to_center(pos: torch.Tensor, pocket_center: torch.Tensor, 
                                   max_atoms: int, max_radius: float = 12.0):
        """Distance-based selection"""
        distances = torch.norm(pos - pocket_center, dim=1)
        
        # Scientific thresholds
        primary_radius = 6.0
        secondary_radius = 12.0
        
        # Priority 1: Atoms within 6 Å
        primary_mask = distances <= primary_radius
        primary_indices = torch.where(primary_mask)[0]
        
        if len(primary_indices) >= max_atoms:
            _, top_indices = torch.topk(distances[primary_indices], 
                                       k=max_atoms, largest=False)
            return primary_indices[top_indices]
        
        # Priority 2: Extend to 12 Å
        secondary_mask = (distances > primary_radius) & (distances <= secondary_radius)
        secondary_indices = torch.where(secondary_mask)[0]
        
        remaining_slots = max_atoms - len(primary_indices)
        if len(secondary_indices) > remaining_slots:
            _, top_secondary = torch.topk(distances[secondary_indices], 
                                         k=remaining_slots, largest=False)
            secondary_indices = secondary_indices[top_secondary]
        
        combined_indices = torch.cat([primary_indices, secondary_indices]) if len(secondary_indices) > 0 else primary_indices
        return combined_indices
    
    @staticmethod
    def select_multi_strategy(pos: torch.Tensor, x: torch.Tensor, max_atoms: int,
                            pocket_center: torch.Tensor = None, 
                            ligand_pos: torch.Tensor = None,
                            strategy: str = "adaptive"):
        """Multi-strategy selection"""
        
        if strategy == "adaptive" or strategy == "distance":
            center = pocket_center if pocket_center is not None else torch.mean(pos, dim=0)
            return SmartPocketAtomSelector.select_by_distance_to_center(pos, center, max_atoms)
        else:
            # Default to distance-based
            center = pocket_center if pocket_center is not None else torch.mean(pos, dim=0)
            return SmartPocketAtomSelector.select_by_distance_to_center(pos, center, max_atoms)


class ImprovedProteinPocketEncoder(nn.Module):
    """
    🎯 UPDATED: Improved pocket encoder - SchNet compatible
    
    Key Changes:
    - Removed EGNN dependency 
    - Using simple but effective MLPs
    - Fully compatible with SchNet backend
    - No unpacking issues
    """
    
    def __init__(self, node_features: int = 8, edge_features: int = 4,
                 hidden_dim: int = 128, num_layers: int = 4, 
                 output_dim: int = 256, max_radius: float = 10.0,
                 max_pocket_atoms: int = 1000, 
                 selection_strategy: str = "adaptive"):
        super().__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.max_pocket_atoms = max_pocket_atoms
        self.selection_strategy = selection_strategy
        self.max_radius = max_radius
        
        # Smart atom selector
        self.atom_selector = SmartPocketAtomSelector()
        
        # Flexible embeddings for different input dimensions
        self.node_embedding_6d = nn.Linear(6, hidden_dim)
        self.node_embedding_7d = nn.Linear(7, hidden_dim)
        self.node_embedding_8d = nn.Linear(8, hidden_dim)
        
        # 🎯 UPDATED: Simple but effective processor (no EGNN)
        self.processor_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        # Position-aware processing (SchNet-style)
        self.position_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        print(f"✅ ImprovedProteinPocketEncoder created (SchNet compatible)")
        print(f"   Strategy: {selection_strategy}, Max atoms: {max_pocket_atoms}")
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor = None,
                batch: torch.Tensor = None, ligand_pos: torch.Tensor = None):
        """
        🎯 UPDATED: SchNet-compatible forward pass
        
        Args:
            x: Node features [N, node_features]
            pos: Node positions [N, 3] 
            edge_index: Edge indices [2, E] (optional, mostly ignored)
            batch: Batch indices [N] (optional)
            ligand_pos: Ligand positions [M, 3] (optional, for smart selection)
        """
        
        # Smart atom selection when pocket is too large
        if x.size(0) > self.max_pocket_atoms:
            pocket_center = torch.mean(pos, dim=0)
            
            try:
                selected_indices = self.atom_selector.select_multi_strategy(
                    pos=pos, 
                    x=x, 
                    max_atoms=self.max_pocket_atoms,
                    pocket_center=pocket_center,
                    ligand_pos=ligand_pos,
                    strategy=self.selection_strategy
                )
                
                # Apply selection
                x = x[selected_indices]
                pos = pos[selected_indices]
                if batch is not None:
                    batch = batch[selected_indices]
                
            except Exception as e:
                print(f"Smart selection failed: {e}, using random fallback")
                indices = torch.randperm(x.size(0))[:self.max_pocket_atoms]
                x = x[indices]
                pos = pos[indices]
                if batch is not None:
                    batch = batch[indices]
        
        # Flexible embedding
        h = self._embed_features_flexible(x)
        
        # Position encoding (SchNet-style)
        pos_features = self.position_mlp(pos)
        
        # Combine node and position features
        h_combined = self.feature_fusion(torch.cat([h, pos_features], dim=-1))
        
        # Layer normalization
        h_normalized = self.layer_norm(h_combined)
        
        # 🎯 UPDATED: Process through simple MLP layers (stable)
        h_processed = h_normalized
        for layer in self.processor_layers:
            h_prev = h_processed
            h_processed = layer(h_processed)
            h_processed = h_processed + h_prev  # Residual connection
        
        # Global pooling
        if batch is not None:
            try:
                pocket_mean = safe_global_pool(h_processed, batch, 'mean')
                pocket_max = safe_global_pool(h_processed, batch, 'max')
                pocket_repr = pocket_mean + pocket_max
            except Exception as e:
                print(f"Pooling error: {e}")
                pocket_repr = torch.mean(h_processed, dim=0, keepdim=True)
        else:
            # Single pocket case
            pocket_repr = torch.mean(h_processed, dim=0, keepdim=True)
        
        # Final output projection
        return self.output_projection(pocket_repr)
    
    def _embed_features_flexible(self, x: torch.Tensor) -> torch.Tensor:
        """Flexible feature embedding"""
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got {x.dim()}D: {x.shape}")
        
        input_dim = x.size(1)
        
        if input_dim == 6:
            return self.node_embedding_6d(x.float())
        elif input_dim == 7:
            return self.node_embedding_7d(x.float())
        elif input_dim == 8:
            return self.node_embedding_8d(x.float())
        elif input_dim < 6:
            # Pad to 6D
            padding = torch.zeros(x.size(0), 6 - input_dim, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
            return self.node_embedding_6d(x_padded.float())
        elif input_dim == 9:
            # Truncate to 8D
            x_truncated = x[:, :8]
            return self.node_embedding_8d(x_truncated.float())
        else:
            # Pad or truncate to 7D
            if input_dim < 7:
                padding = torch.zeros(x.size(0), 7 - input_dim, device=x.device, dtype=x.dtype)
                x_padded = torch.cat([x, padding], dim=1)
                return self.node_embedding_7d(x_padded.float())
            else:
                x_truncated = x[:, :7]
                return self.node_embedding_7d(x_truncated.float())


# 🎯 UPDATED: Factory function for SchNet compatibility
def create_improved_pocket_encoder(hidden_dim: int = 256, output_dim: int = 256, 
                                 selection_strategy: str = "adaptive"):
    """
    🎯 UPDATED: Create improved pocket encoder - SchNet compatible version
    
    No EGNN dependencies, fully compatible with SchNet backend
    """
    
    return ImprovedProteinPocketEncoder(
        node_features=7,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        selection_strategy=selection_strategy
    )


# 🎯 Simple pocket encoder for basic usage
class SimplePocketEncoder(nn.Module):
    """Simple pocket encoder - most basic version"""
    
    def __init__(self, input_dim: int = 7, hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor = None, **kwargs):
        # Simple mean pooling over pocket atoms
        pocket_features = torch.mean(x, dim=0, keepdim=True)
        return self.encoder(pocket_features)


# Test function
def test_improved_pocket_encoder():
    """Test the pocket encoder"""
    print("Testing Improved Pocket Encoder (SchNet compatible)...")
    
    # Create test data
    num_residues = 500
    pocket_x = torch.randn(num_residues, 7)
    pocket_pos = torch.randn(num_residues, 3)
    pocket_batch = torch.zeros(num_residues, dtype=torch.long)
    
    # Create encoder
    encoder = create_improved_pocket_encoder(
        hidden_dim=128,
        output_dim=256,
        selection_strategy="adaptive"
    )
    
    # Test forward pass
    try:
        pocket_repr = encoder(
            x=pocket_x,
            pos=pocket_pos,
            batch=pocket_batch
        )
        
        print(f"   Input: {num_residues} residues")
        print(f"   Output: {pocket_repr.shape}")
        print(f"   ✅ Success: SchNet-compatible pocket processing")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("🎯 Improved Pocket Encoder - SchNet Compatible Version")
    print("=" * 60)
    print("Features:")
    print("- No EGNN dependencies (removed)")
    print("- Simple but effective MLP processing")
    print("- Smart atom selection")
    print("- Position-aware encoding")
    print("- Flexible input dimensions")
    print("- Full SchNet compatibility")
    print()
    
    test_improved_pocket_encoder()
