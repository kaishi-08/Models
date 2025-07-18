# src/models/pocket_encoder.py - Complete version with all classes
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch
try:
    from torch_geometric.nn import knn_graph, radius_graph
except ImportError:
    # Fallback for older versions
    from torch_geometric.nn import knn_graph
    from torch_geometric.nn import radius_graph

try:
    from .e3_egnn import E3EquivariantGNN
except ImportError:
    # Fallback if e3_egnn is not available
    print("Warning: E3EquivariantGNN not available, using simple fallback")
    class E3EquivariantGNN(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.fallback_net = nn.Linear(kwargs.get('hidden_dim', 128), kwargs.get('hidden_dim', 128))
        
        def forward(self, x, pos, edge_index, batch=None):
            return self.fallback_net(x)

class SmartPocketAtomSelector:
    """Smart strategies for selecting important pocket atoms"""
    
    @staticmethod
    def select_by_distance_to_center(pos: torch.Tensor, pocket_center: torch.Tensor, 
                                   max_atoms: int, max_radius: float = 12.0):
        """Select atoms based on distance to pocket center"""
        distances = torch.norm(pos - pocket_center, dim=1)
        
        # First filter by radius
        radius_mask = distances <= max_radius
        if radius_mask.sum() <= max_atoms:
            return torch.where(radius_mask)[0]
        
        # Then select closest atoms
        _, indices = torch.topk(distances, k=max_atoms, largest=False)
        return indices
    
    @staticmethod
    def select_by_surface_accessibility(pos: torch.Tensor, x: torch.Tensor, 
                                      max_atoms: int, probe_radius: float = 1.4):
        """Select surface-accessible atoms (more likely to interact with ligand)"""
        # Simplified surface accessibility calculation
        try:
            distances = torch.cdist(pos, pos)  # [N, N]
            
            # Count neighbors within probe radius + atom radius
            neighbor_counts = (distances < (probe_radius + 2.0)).sum(dim=1) - 1  # Exclude self
            
            # Surface atoms have fewer neighbors
            surface_scores = 1.0 / (neighbor_counts.float() + 1.0)
            
            # Select atoms with highest surface accessibility
            _, indices = torch.topk(surface_scores, k=min(max_atoms, len(surface_scores)), largest=True)
            return indices
        except Exception as e:
            print(f"Surface selection failed: {e}, using distance-based fallback")
            center = torch.mean(pos, dim=0)
            return SmartPocketAtomSelector.select_by_distance_to_center(pos, center, max_atoms)
    
    @staticmethod
    def select_by_residue_importance(pos: torch.Tensor, x: torch.Tensor, 
                                   max_atoms: int, pocket_center: torch.Tensor):
        """Select atoms from important residues"""
        try:
            # Assume x contains residue type information in x[:, 1]
            if x.size(1) < 2:
                return SmartPocketAtomSelector.select_by_distance_to_center(
                    pos, pocket_center, max_atoms
                )
            
            residue_types = x[:, 1]  # Residue type feature
            
            # Important residues for binding (aromatic, charged, etc.)
            # Based on amino acid properties
            important_residues = {1, 3, 5, 6, 8, 11, 13, 17, 18, 19}  # ARG, ASP, GLU, HIS, LYS, PHE, TRP, TYR, VAL
            
            # Score atoms
            scores = torch.zeros(len(pos), device=pos.device)
            
            for i, res_type in enumerate(residue_types):
                # Base score from residue importance
                if res_type.item() in important_residues:
                    scores[i] += 2.0
                else:
                    scores[i] += 1.0
                
                # Distance bonus (closer to center = higher score)
                dist_to_center = torch.norm(pos[i] - pocket_center)
                scores[i] += 1.0 / (1.0 + dist_to_center / 10.0)
            
            # Select highest scoring atoms
            _, indices = torch.topk(scores, k=min(max_atoms, len(scores)), largest=True)
            return indices
        except Exception as e:
            print(f"Residue selection failed: {e}, using distance-based fallback")
            return SmartPocketAtomSelector.select_by_distance_to_center(pos, pocket_center, max_atoms)
    
    @staticmethod
    def select_by_binding_site_proximity(pos: torch.Tensor, x: torch.Tensor,
                                       ligand_pos: torch.Tensor, max_atoms: int,
                                       interaction_radius: float = 8.0):
        """Select atoms close to known/predicted binding site"""
        try:
            if ligand_pos is None or len(ligand_pos) == 0:
                # Fallback to center-based selection
                pocket_center = torch.mean(pos, dim=0)
                return SmartPocketAtomSelector.select_by_distance_to_center(
                    pos, pocket_center, max_atoms
                )
            
            # Find pocket atoms close to any ligand atom
            ligand_center = torch.mean(ligand_pos, dim=0)
            distances_to_ligand = torch.norm(pos - ligand_center, dim=1)
            
            # Combine distance and surface accessibility
            surface_scores = SmartPocketAtomSelector._compute_surface_scores(pos)
            
            # Combined scoring: closer to ligand + surface accessible
            interaction_scores = 1.0 / (1.0 + distances_to_ligand / 5.0) + surface_scores * 0.5
            
            # Select highest scoring atoms
            _, indices = torch.topk(interaction_scores, k=min(max_atoms, len(interaction_scores)), largest=True)
            return indices
        except Exception as e:
            print(f"Binding site selection failed: {e}, using distance-based fallback")
            pocket_center = torch.mean(pos, dim=0)
            return SmartPocketAtomSelector.select_by_distance_to_center(pos, pocket_center, max_atoms)
    
    @staticmethod
    def _compute_surface_scores(pos: torch.Tensor, neighbor_radius: float = 3.0):
        """Compute surface accessibility scores"""
        try:
            distances = torch.cdist(pos, pos)
            neighbor_counts = (distances < neighbor_radius).sum(dim=1) - 1
            return 1.0 / (neighbor_counts.float() + 1.0)
        except:
            # Fallback to uniform scores
            return torch.ones(len(pos), device=pos.device)
    
    @staticmethod
    def select_multi_strategy(pos: torch.Tensor, x: torch.Tensor, max_atoms: int,
                            pocket_center: torch.Tensor = None, 
                            ligand_pos: torch.Tensor = None,
                            strategy: str = "adaptive"):
        """Multi-strategy selection based on available information"""
        
        if strategy == "adaptive":
            # Choose best strategy based on available information
            if ligand_pos is not None and len(ligand_pos) > 0:
                return SmartPocketAtomSelector.select_by_binding_site_proximity(
                    pos, x, ligand_pos, max_atoms
                )
            elif pocket_center is not None:
                return SmartPocketAtomSelector.select_by_residue_importance(
                    pos, x, max_atoms, pocket_center
                )
            else:
                center = torch.mean(pos, dim=0)
                return SmartPocketAtomSelector.select_by_distance_to_center(
                    pos, center, max_atoms
                )
        
        elif strategy == "distance":
            center = pocket_center if pocket_center is not None else torch.mean(pos, dim=0)
            return SmartPocketAtomSelector.select_by_distance_to_center(
                pos, center, max_atoms
            )
        
        elif strategy == "surface":
            return SmartPocketAtomSelector.select_by_surface_accessibility(
                pos, x, max_atoms
            )
        
        elif strategy == "residue":
            center = pocket_center if pocket_center is not None else torch.mean(pos, dim=0)
            return SmartPocketAtomSelector.select_by_residue_importance(
                pos, x, max_atoms, center
            )
        
        elif strategy == "binding_site":
            return SmartPocketAtomSelector.select_by_binding_site_proximity(
                pos, x, ligand_pos, max_atoms
            )
        
        else:
            print(f"Unknown strategy: {strategy}, using adaptive")
            return SmartPocketAtomSelector.select_multi_strategy(
                pos, x, max_atoms, pocket_center, ligand_pos, "adaptive"
            )

class ProteinPocketEncoder(nn.Module):
    """Protein pocket encoder with smart atom selection"""
    
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
        
        # Use linear layer for feature projection
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.edge_embedding = nn.Linear(edge_features if edge_features > 0 else 1, hidden_dim)
        
        # E(3) equivariant network for pocket encoding
        try:
            self.e3_net = E3EquivariantGNN(
                irreps_in=f"{hidden_dim}x0e",
                irreps_hidden=f"{hidden_dim}x0e+{hidden_dim//2}x1o",
                irreps_out=f"{hidden_dim}x0e",
                num_layers=num_layers,
                max_radius=max_radius
            )
        except Exception as e:
            print(f"Warning: E3EquivariantGNN initialization failed: {e}")
            # Fallback to simple MLP
            self.e3_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Attention mechanism for pocket-ligand interaction
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor = None,
                batch: torch.Tensor = None, ligand_pos: torch.Tensor = None):
        """
        Args:
            x: Node features [N, node_features]
            pos: Node positions [N, 3] 
            edge_index: Edge indices [2, E] (optional)
            batch: Batch indices [N] (optional)
            ligand_pos: Ligand positions for binding site guidance [M, 3] (optional)
        """
        
        # Smart atom selection when pocket is too large
        if x.size(0) > self.max_pocket_atoms:
            # Compute pocket center
            pocket_center = torch.mean(pos, dim=0)
            
            # Smart selection based on strategy
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
                
                print(f"Smart selection ({self.selection_strategy}): {len(selected_indices)} atoms selected from {x.size(0)} total")
                
            except Exception as e:
                print(f"Smart selection failed: {e}, using random fallback")
                # Fallback to random selection as last resort
                indices = torch.randperm(x.size(0))[:self.max_pocket_atoms]
                x = x[indices]
                pos = pos[indices]
                if batch is not None:
                    batch = batch[indices]
        
        # Handle input features
        if x.dim() == 2 and x.size(1) == self.node_features:
            h = self.node_embedding(x.float())
        else:
            raise ValueError(f"Expected x shape [N, {self.node_features}], got {x.shape}")
        
        # Create edges if not provided
        if edge_index is None or edge_index.size(1) == 0:
            edge_index = self._create_distance_edges(pos, max_dist=self.max_radius)
        
        # Apply E(3) equivariant network
        try:
            if hasattr(self.e3_net, 'forward') and callable(self.e3_net.forward):
                # Check if E3 network expects batch parameter
                if 'batch' in self.e3_net.forward.__code__.co_varnames:
                    h = self.e3_net(h, pos, edge_index, batch)
                else:
                    h = self.e3_net(h, pos, edge_index)
            else:
                # Fallback to simple processing
                h = self.e3_net(h)
        except Exception as e:
            print(f"Warning: E3 network failed: {e}")
            h = self.node_embedding(x.float())
        
        # Handle batching and attention
        if batch is not None:
            try:
                h_dense, mask = to_dense_batch(h, batch)
                h_att, _ = self.attention(h_dense, h_dense, h_dense, key_padding_mask=~mask)
                h_att = h_att[mask]
                pocket_repr = global_mean_pool(h_att, batch)
            except:
                pocket_repr = global_mean_pool(h, batch)
        else:
            h_mean = torch.mean(h, dim=0, keepdim=True)
            pocket_repr = h_mean
        
        return self.output_projection(pocket_repr)
    
    def _create_distance_edges(self, pos: torch.Tensor, max_dist: float = 10.0):
        """Create edges based on distance threshold with smart selection"""
        try:
            # Use k-nearest neighbors for more stable connectivity
            if pos.size(0) > 500:
                # For large pockets, use KNN instead of radius
                k = min(32, pos.size(0) - 1)  # Each atom connects to 32 nearest neighbors
                edge_index = knn_graph(pos, k=k, batch=None)
            else:
                edge_index = radius_graph(pos, r=max_dist, batch=None, max_num_neighbors=64)
            
            return edge_index
        except Exception as e:
            print(f"Edge creation failed: {e}, using simple fallback")
            # Fallback: create empty edge index
            return torch.zeros((2, 0), dtype=torch.long, device=pos.device)

class CrossAttentionPocketConditioner(nn.Module):
    """Cross-attention mechanism for pocket-ligand conditioning"""
    
    def __init__(self, ligand_dim: int = 128, pocket_dim: int = 256, 
                 hidden_dim: int = 128, num_heads: int = 8):
        super().__init__()
        
        self.ligand_proj = nn.Linear(ligand_dim, hidden_dim)
        self.pocket_proj = nn.Linear(pocket_dim, hidden_dim)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(hidden_dim, ligand_dim)
        
    def forward(self, ligand_features: torch.Tensor, pocket_features: torch.Tensor,
                ligand_batch: torch.Tensor):
        """
        Args:
            ligand_features: [N_ligand, ligand_dim]
            pocket_features: [N_batch, pocket_dim]
            ligand_batch: [N_ligand] batch assignment for ligand atoms
        """
        # Project features
        ligand_proj = self.ligand_proj(ligand_features)
        pocket_proj = self.pocket_proj(pocket_features)
        
        try:
            # Convert to dense format
            ligand_dense, ligand_mask = to_dense_batch(ligand_proj, ligand_batch)
            
            # Expand pocket features to match ligand batch size
            pocket_expanded = pocket_proj[ligand_batch].unsqueeze(1)
            
            # Cross-attention: ligand queries, pocket keys/values
            attended_ligand, _ = self.cross_attention(
                ligand_dense, pocket_expanded, pocket_expanded,
                key_padding_mask=None
            )
            
            # Back to sparse format
            attended_ligand = attended_ligand[ligand_mask]
            
            # Output projection
            return self.output_proj(attended_ligand)
            
        except Exception as e:
            print(f"Warning: Cross attention failed: {e}")
            # Fallback: simple addition
            try:
                pocket_broadcast = pocket_proj[ligand_batch]
                combined = ligand_proj + pocket_broadcast
                return self.output_proj(combined)
            except:
                # Ultimate fallback: just return ligand features
                return ligand_features