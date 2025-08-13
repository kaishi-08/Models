import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, knn_graph
from torch_geometric.nn.pool import radius_graph
from torch_geometric.utils import to_dense_batch
import numpy as np

# Safe global pooling function
def safe_global_pool(x, batch, pool_type='mean'):
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
    
    @staticmethod
    def select_by_distance_to_ligand(pocket_pos: torch.Tensor, ligand_pos: torch.Tensor, 
                                   max_atoms: int, primary_radius: float = 6.0, 
                                   secondary_radius: float = 12.0):
        ligand_center = torch.mean(ligand_pos, dim=0)
        distances = torch.norm(pocket_pos - ligand_center, dim=1)
        
        # Priority 1: Atoms within primary radius (direct interaction)
        primary_mask = distances <= primary_radius
        primary_indices = torch.where(primary_mask)[0]
        
        if len(primary_indices) >= max_atoms:
            _, top_indices = torch.topk(distances[primary_indices], 
                                       k=max_atoms, largest=False)
            return primary_indices[top_indices]
        
        # Priority 2: Extend to secondary radius
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
    def select_by_binding_site_prediction(pocket_pos: torch.Tensor, pocket_x: torch.Tensor,
                                        ligand_pos: torch.Tensor, max_atoms: int):
        """Binding site prediction based on chemical features and geometry"""
        ligand_center = torch.mean(ligand_pos, dim=0)
        
        # Distance component
        distances = torch.norm(pocket_pos - ligand_center, dim=1)
        distance_scores = torch.exp(-distances / 8.0)  # 8Å decay
        
        # Chemical feature component (if available)
        if pocket_x.size(1) >= 7:  # Has chemical features
            # Assume chemical features: [res_type, hydrophobic, charged, polar, aromatic, ...]
            chemical_scores = torch.zeros(pocket_x.size(0), device=pocket_x.device)
            
            # Prefer hydrophobic, aromatic, and charged residues for binding
            if pocket_x.size(1) > 2:  # Has hydrophobic
                chemical_scores += pocket_x[:, 2] * 0.3  # Hydrophobic
            if pocket_x.size(1) > 3:  # Has charged
                chemical_scores += pocket_x[:, 3] * 0.4  # Charged
            if pocket_x.size(1) > 5:  # Has aromatic
                chemical_scores += pocket_x[:, 5] * 0.3  # Aromatic
        else:
            chemical_scores = torch.ones(pocket_x.size(0), device=pocket_x.device)
        
        # Combined score
        combined_scores = distance_scores * 0.7 + chemical_scores * 0.3
        
        # Select top atoms
        _, selected_indices = torch.topk(combined_scores, k=min(max_atoms, len(combined_scores)), largest=True)
        return selected_indices
    
    @staticmethod
    def select_by_surface_accessibility(pocket_pos: torch.Tensor, ligand_pos: torch.Tensor,
                                      max_atoms: int, probe_radius: float = 1.4):
        """Select surface-accessible atoms (simplified version)"""
        ligand_center = torch.mean(ligand_pos, dim=0)
        distances = torch.norm(pocket_pos - ligand_center, dim=1)
        
        # Simple surface approximation: atoms not too crowded
        pairwise_distances = torch.cdist(pocket_pos, pocket_pos)
        neighbor_counts = (pairwise_distances < probe_radius * 2).sum(dim=1) - 1  # Exclude self
        
        # Prefer atoms with fewer neighbors (more surface-like) but close to ligand
        surface_scores = 1.0 / (neighbor_counts.float() + 1.0)
        distance_scores = torch.exp(-distances / 10.0)
        
        combined_scores = surface_scores * 0.4 + distance_scores * 0.6
        
        _, selected_indices = torch.topk(combined_scores, k=min(max_atoms, len(combined_scores)), largest=True)
        return selected_indices
    
    @staticmethod
    def select_adaptive(pocket_pos: torch.Tensor, pocket_x: torch.Tensor,
                       ligand_pos: torch.Tensor, max_atoms: int, strategy: str = "adaptive"):
        """Adaptive selection combining multiple strategies"""
        
        if strategy == "distance":
            return SmartPocketAtomSelector.select_by_distance_to_ligand(
                pocket_pos, ligand_pos, max_atoms
            )
        elif strategy == "binding_site":
            return SmartPocketAtomSelector.select_by_binding_site_prediction(
                pocket_pos, pocket_x, ligand_pos, max_atoms
            )
        elif strategy == "surface":
            return SmartPocketAtomSelector.select_by_surface_accessibility(
                pocket_pos, ligand_pos, max_atoms
            )
        else:  # adaptive
            # Use binding site prediction if chemical features available, else distance
            if pocket_x.size(1) >= 6:
                return SmartPocketAtomSelector.select_by_binding_site_prediction(
                    pocket_pos, pocket_x, ligand_pos, max_atoms
                )
            else:
                return SmartPocketAtomSelector.select_by_distance_to_ligand(
                    pocket_pos, ligand_pos, max_atoms
                )


class EGNNPocketLayer(nn.Module):
    """🎯 EGNN-style layer for pocket processing (simplified for pocket atoms)"""
    
    def __init__(self, hidden_dim, edge_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Edge model (simplified for pocket-pocket interactions)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + 1, hidden_dim),  # +1 for distance
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node update model
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Position update (optional for pocket atoms)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, h, pos, edge_index):
        """Simple EGNN-style update for pocket atoms"""
        if edge_index.size(1) == 0:
            return h, pos
        
        row, col = edge_index
        
        # Edge features with distance
        radial = pos[row] - pos[col]
        radial_norm = torch.norm(radial, dim=-1, keepdim=True)
        radial_norm = torch.clamp(radial_norm, min=1e-8)
        
        # Edge processing
        edge_input = torch.cat([h[row], h[col], radial_norm], dim=-1)
        m_ij = self.edge_mlp(edge_input)
        
        # Node update
        agg = torch.zeros(h.size(0), self.hidden_dim, device=h.device)
        agg.index_add_(0, row, m_ij)
        
        h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
        h_new = h + h_new  # Residual connection
        
        # Minimal position update (optional)
        coord_diff = self.coord_mlp(m_ij)
        coord_diff = torch.tanh(coord_diff) * 0.1  # Small updates
        radial_normalized = radial / radial_norm
        coord_update = coord_diff * radial_normalized
        
        pos_new = pos.clone()
        pos_new.index_add_(0, row, coord_update)
        
        return h_new, pos_new


class ImprovedProteinPocketEncoder(nn.Module):
    """
    🎯 ENHANCED: Improved pocket encoder with EGNN-style processing and smart selection
    
    Key improvements:
    - Smart atom selection strategies
    - EGNN-compatible processing 
    - Multi-scale pocket representation
    - Chemical-aware encoding
    - Flexible input handling
    """
    
    def __init__(self, node_features: int = 8, edge_features: int = 4,
                 hidden_dim: int = 128, num_layers: int = 3, 
                 output_dim: int = 256, max_radius: float = 10.0,
                 max_pocket_atoms: int = 1000, 
                 selection_strategy: str = "adaptive",
                 use_egnn_layers: bool = True):
        super().__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_pocket_atoms = max_pocket_atoms
        self.selection_strategy = selection_strategy
        self.max_radius = max_radius
        self.use_egnn_layers = use_egnn_layers
        
        # Smart atom selector
        self.atom_selector = SmartPocketAtomSelector()
        
        # Flexible embeddings for different input dimensions
        self.node_embedding_6d = nn.Linear(6, hidden_dim)
        self.node_embedding_7d = nn.Linear(7, hidden_dim)
        self.node_embedding_8d = nn.Linear(8, hidden_dim)
        self.node_embedding_9d = nn.Linear(9, hidden_dim)
        
        # 🎯 EGNN-style processing layers (optional)
        if use_egnn_layers:
            self.egnn_layers = nn.ModuleList([
                EGNNPocketLayer(hidden_dim) for _ in range(num_layers)
            ])
        else:
            # Simple MLP layers
            self.mlp_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.LayerNorm(hidden_dim)
                ) for _ in range(num_layers)
            ])
        
        # Multi-scale feature extraction
        self.local_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.global_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Position-aware processing
        self.position_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # node + position + processed
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection with attention
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=0)
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        print(f"✅ Enhanced PocketEncoder created")
        print(f"   Strategy: {selection_strategy}, Max atoms: {max_pocket_atoms}")
        print(f"   EGNN-style: {use_egnn_layers}, Layers: {num_layers}")
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor = None,
                batch: torch.Tensor = None, ligand_pos: torch.Tensor = None, **kwargs):
        
        if x.size(0) > self.max_pocket_atoms and ligand_pos is not None:
            try:
                selected_indices = self.atom_selector.select_adaptive(
                    pocket_pos=pos, 
                    pocket_x=x, 
                    ligand_pos=ligand_pos,
                    max_atoms=self.max_pocket_atoms,
                    strategy=self.selection_strategy
                )
                
                x = x[selected_indices]
                pos = pos[selected_indices]
                if batch is not None:
                    batch = batch[selected_indices]
                if edge_index is not None:
                    edge_mask = torch.isin(edge_index[0], selected_indices) & torch.isin(edge_index[1], selected_indices)
                    if edge_mask.any():
                        edge_index = edge_index[:, edge_mask]
                        # Remap indices
                        old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(selected_indices)}
                        edge_index = torch.tensor([[old_to_new[edge_index[0, i].item()], old_to_new[edge_index[1, i].item()]] 
                                                 for i in range(edge_index.size(1)) 
                                                 if edge_index[0, i].item() in old_to_new and edge_index[1, i].item() in old_to_new]).t()
                    else:
                        edge_index = torch.zeros((2, 0), dtype=torch.long, device=x.device)
                
            except Exception as e:
                print(f"Smart selection failed: {e}, using random fallback")
                indices = torch.randperm(x.size(0))[:self.max_pocket_atoms]
                x = x[indices]
                pos = pos[indices]
                if batch is not None:
                    batch = batch[indices]
        
        h = self._embed_features_flexible(x)
        
        # Position encoding
        pos_features = self.position_mlp(pos)
        
        # Combine node and position features
        h_with_pos = self.feature_fusion(torch.cat([h, pos_features, h], dim=-1))
        h_normalized = self.layer_norm(h_with_pos)
        
        # 🎯 Process through EGNN layers or MLP layers
        if self.use_egnn_layers:
            h_processed, pos_updated = self._process_with_egnn(h_normalized, pos, edge_index)
        else:
            h_processed = self._process_with_mlp(h_normalized)
            pos_updated = pos
        
        # Multi-scale processing
        h_local = self.local_processor(h_processed)
        h_global = self.global_processor(h_processed)
        h_combined = h_local + h_global
        
        # Global pooling with attention
        if batch is not None:
            try:
                # Attention-weighted pooling
                attention_weights = self.attention_weights(h_combined)
                weighted_features = h_combined * attention_weights
                
                pocket_mean = safe_global_pool(weighted_features, batch, 'mean')
                pocket_max = safe_global_pool(h_combined, batch, 'max')
                pocket_repr = pocket_mean + pocket_max * 0.5
            except Exception as e:
                print(f"Attention pooling error: {e}")
                pocket_repr = safe_global_pool(h_combined, batch, 'mean')
        else:
            # Single pocket case with attention
            attention_weights = self.attention_weights(h_combined)
            weighted_features = h_combined * attention_weights
            pocket_repr = torch.sum(weighted_features, dim=0, keepdim=True)
        
        # Final output projection
        return self.output_projection(pocket_repr)
    
    def _embed_features_flexible(self, x: torch.Tensor) -> torch.Tensor:
        """Flexible feature embedding for different input dimensions"""
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got {x.dim()}D: {x.shape}")
        
        input_dim = x.size(1)
        
        if input_dim == 6:
            return self.node_embedding_6d(x.float())
        elif input_dim == 7:
            return self.node_embedding_7d(x.float())
        elif input_dim == 8:
            return self.node_embedding_8d(x.float())
        elif input_dim == 9:
            return self.node_embedding_9d(x.float())
        elif input_dim < 6:
            # Pad to 6D
            padding = torch.zeros(x.size(0), 6 - input_dim, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
            return self.node_embedding_6d(x_padded.float())
        elif input_dim > 9:
            # Truncate to 9D
            x_truncated = x[:, :9]
            return self.node_embedding_9d(x_truncated.float())
        else:
            # Pad to 8D
            padding = torch.zeros(x.size(0), 8 - input_dim, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
            return self.node_embedding_8d(x_padded.float())
    
    def _process_with_egnn(self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        """Process with EGNN-style layers"""
        h_current = h
        pos_current = pos
        
        # Build edge index if not provided
        if edge_index is None or edge_index.size(1) == 0:
            # Create edges based on distance
            if h.size(0) > 1:
                edge_index = radius_graph(pos_current, r=self.max_radius, 
                                        max_num_neighbors=min(32, h.size(0)-1))
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=h.device)
        
        # Apply EGNN layers
        for layer in self.egnn_layers:
            h_prev = h_current
            h_current, pos_current = layer(h_current, pos_current, edge_index)
            h_current = h_current + h_prev  # Residual connection
        
        return h_current, pos_current
    
    def _process_with_mlp(self, h: torch.Tensor):
        """Process with simple MLP layers"""
        h_current = h
        
        for layer in self.mlp_layers:
            h_prev = h_current
            h_current = layer(h_current)
            h_current = h_current + h_prev  # Residual connection
        
        return h_current


# 🎯 Factory function for creating improved pocket encoder
def create_improved_pocket_encoder(hidden_dim: int = 256, output_dim: int = 256, 
                                 selection_strategy: str = "adaptive",
                                 use_egnn_layers: bool = True, **kwargs):
  
    return ImprovedProteinPocketEncoder(
        node_features=8,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        selection_strategy=selection_strategy,
        use_egnn_layers=use_egnn_layers,
        **kwargs
    )


# 🎯 Simple pocket encoder for basic usage
class SimplePocketEncoder(nn.Module):
    """Simple pocket encoder fallback"""
    
    def __init__(self, input_dim: int = 7, hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor = None, **kwargs):
        # Simple mean pooling over pocket atoms
        pocket_features = torch.mean(x, dim=0, keepdim=True)
        return self.encoder(pocket_features)


# Test function
def test_improved_pocket_encoder():
    """Test the enhanced pocket encoder"""
    print("Testing Enhanced Pocket Encoder...")
    
    # Create test data
    num_residues = 800
    num_ligand_atoms = 25
    
    pocket_x = torch.randn(num_residues, 7)
    pocket_pos = torch.randn(num_residues, 3) * 20  # 20Å spread
    pocket_batch = torch.zeros(num_residues, dtype=torch.long)
    
    ligand_pos = torch.randn(num_ligand_atoms, 3) * 5  # Ligand in center
    
    # Test different strategies
    strategies = ["adaptive", "distance", "binding_site", "surface"]
    
    for strategy in strategies:
        print(f"\n   Testing strategy: {strategy}")
        
        try:
            # Create encoder
            encoder = create_improved_pocket_encoder(
                hidden_dim=128,
                output_dim=256,
                selection_strategy=strategy,
                use_egnn_layers=True
            )
            
            # Test forward pass
            pocket_repr = encoder(
                x=pocket_x,
                pos=pocket_pos,
                batch=pocket_batch,
                ligand_pos=ligand_pos
            )
            
            print(f"     Input: {num_residues} residues → Output: {pocket_repr.shape}")
            print(f"     ✅ Strategy '{strategy}' successful")
            
        except Exception as e:
            print(f"     ❌ Strategy '{strategy}' failed: {e}")
    
    # Test fallback
    print(f"\n   Testing simple fallback...")
    try:
        simple_encoder = SimplePocketEncoder(input_dim=7, output_dim=256)
        simple_repr = simple_encoder(pocket_x)
        print(f"     Simple encoder: {pocket_x.shape} → {simple_repr.shape}")
        print(f"     ✅ Simple encoder successful")
    except Exception as e:
        print(f"     ❌ Simple encoder failed: {e}")


if __name__ == "__main__":
    test_improved_pocket_encoder()