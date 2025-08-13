# src/models/egnn.py - CORRECTED EGNN Implementation based on DiffSBDD/Official EGNN
# 🎯 FIXED: Proper SE(3) Equivariant EGNN with rotation, translation, reflection equivariance
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import radius_graph
from typing import Dict, Optional, Tuple, Union

def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result

def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]  # Relative positions
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)  # Squared distances
    norm = torch.sqrt(radial + 1e-8)  # Distances
    coord_diff = coord_diff / (norm + norm_constant)  # Normalized relative positions
    return radial, coord_diff

def coord2cross(x, edge_index, batch_mask, norm_constant=1):
    """Compute cross products for reflection equivariance"""
    # Compute batch means
    mean = unsorted_segment_sum(x, batch_mask,
                              num_segments=batch_mask.max() + 1,
                              normalization_factor=None,
                              aggregation_method='mean')
    row, col = edge_index
    # Cross product of centered positions
    cross = torch.cross(x[row] - mean[batch_mask[row]],
                       x[col] - mean[batch_mask[col]], dim=1)
    norm = torch.linalg.norm(cross, dim=1, keepdim=True)
    cross = cross / (norm + norm_constant)
    return cross

class SinusoidsEmbeddingNew(nn.Module):
    """Sinusoidal embedding for distances"""
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies) / max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()

class GCL(nn.Module):    
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        # Edge MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        # Node MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        # Attention (optional)
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                 normalization_factor=self.normalization_factor,
                                 aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)  # Residual connection
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij

class EquivariantUpdate(nn.Module):
    """SE(3) Equivariant coordinate update"""
    
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0,
                 reflection_equiv=True):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        self.reflection_equiv = reflection_equiv
        input_edge = hidden_nf * 2 + edges_in_d
        
        # Coordinate MLP (outputs scalar to scale relative positions)
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)  # Small initialization
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)
        
        # Cross product MLP (for reflection non-equivariance)
        if not self.reflection_equiv:
            layer_cross = nn.Linear(hidden_nf, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer_cross.weight, gain=0.001)
            self.cross_product_mlp = nn.Sequential(
                nn.Linear(input_edge, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
                layer_cross
            )
        else:
            self.cross_product_mlp = None
            
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, coord_cross,
                   edge_attr, edge_mask, update_coords_mask=None):
        """SE(3) equivariant coordinate update"""
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        
        # Main coordinate update (equivariant)
        coord_weights = self.coord_mlp(input_tensor)
        if self.tanh:
            trans = coord_diff * torch.tanh(coord_weights) * self.coords_range
        else:
            trans = coord_diff * coord_weights

        # Cross product term (breaks reflection equivariance)
        if not self.reflection_equiv:
            phi_cross = self.cross_product_mlp(input_tensor)
            if self.tanh:
                phi_cross = torch.tanh(phi_cross) * self.coords_range
            trans = trans + coord_cross * phi_cross

        # Apply edge mask
        if edge_mask is not None:
            trans = trans * edge_mask

        # Aggregate coordinate updates
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                 normalization_factor=self.normalization_factor,
                                 aggregation_method=self.aggregation_method)

        # Apply coordinate update mask
        if update_coords_mask is not None:
            agg = update_coords_mask * agg

        coord = coord + agg
        return coord

    def forward(self, h, coord, edge_index, coord_diff, coord_cross,
               edge_attr=None, node_mask=None, edge_mask=None,
               update_coords_mask=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, coord_cross,
                               edge_attr, edge_mask,
                               update_coords_mask=update_coords_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord

class EquivariantBlock(nn.Module):
    """Equivariant block combining GCL and EquivariantUpdate"""
    
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum', reflection_equiv=True):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.reflection_equiv = reflection_equiv

        # Multiple GCL layers for node feature updates
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, 
                                            edges_in_d=edge_feat_nf,
                                            act_fn=act_fn, attention=attention,
                                            normalization_factor=self.normalization_factor,
                                            aggregation_method=self.aggregation_method))
        
        # Equivariant coordinate update
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, 
                                                     act_fn=nn.SiLU(), tanh=tanh,
                                                     coords_range=self.coords_range_layer,
                                                     normalization_factor=self.normalization_factor,
                                                     aggregation_method=self.aggregation_method,
                                                     reflection_equiv=self.reflection_equiv))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None,
               edge_attr=None, update_coords_mask=None, batch_mask=None):
        # Compute relative positions and distances
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        
        # Compute cross products if not reflection equivariant
        if self.reflection_equiv:
            coord_cross = None
        else:
            coord_cross = coord2cross(x, edge_index, batch_mask, self.norm_constant)
        
        # Apply sinusoidal embedding to distances
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        
        # Combine with edge attributes
        if edge_attr is not None:
            edge_attr = torch.cat([distances, edge_attr], dim=1)
        else:
            edge_attr = distances
        
        # Update node features through GCL layers
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr,
                                             node_mask=node_mask, edge_mask=edge_mask)
        
        # Update coordinates equivariantly
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, coord_cross, edge_attr,
                                     node_mask, edge_mask, update_coords_mask=update_coords_mask)

        # Apply node mask
        if node_mask is not None:
            h = h * node_mask
        return h, x

class EGNN(nn.Module):
    
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, 
                 attention=False, norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, 
                 norm_constant=1, inv_sublayers=2, sin_embedding=False, normalization_factor=100, 
                 aggregation_method='sum', reflection_equiv=True):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range / n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.reflection_equiv = reflection_equiv

        # Sinusoidal embedding for distances
        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim
        else:
            self.sin_embedding = None
            edge_feat_nf = 1  # Just squared distance

        edge_feat_nf = edge_feat_nf + in_edge_nf

        # Input/output embeddings
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        
        # Equivariant blocks
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, 
                                                             device=device, act_fn=act_fn, 
                                                             n_layers=inv_sublayers, attention=attention, 
                                                             norm_diff=norm_diff, tanh=tanh,
                                                             coords_range=coords_range, 
                                                             norm_constant=norm_constant,
                                                             sin_embedding=self.sin_embedding,
                                                             normalization_factor=self.normalization_factor,
                                                             aggregation_method=self.aggregation_method,
                                                             reflection_equiv=self.reflection_equiv))
        self.to(self.device)
        
        print(f"   Layers: {n_layers}, Hidden: {hidden_nf}")
        print(f"   SE(3) equivariant: {'✓' if reflection_equiv else '✗'}")
        print(f"   Reflection equivariant: {'✓' if reflection_equiv else '✗'}")
        print(f"   Sin embedding: {'✓' if sin_embedding else '✗'}")

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, 
               update_coords_mask=None, batch_mask=None, edge_attr=None):

        # Compute initial edge features (distances)
        edge_feat, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            edge_feat = self.sin_embedding(edge_feat)
        if edge_attr is not None:
            edge_feat = torch.cat([edge_feat, edge_attr], dim=1)
        
        # Initial embedding
        h = self.embedding(h)
        
        # Pass through equivariant blocks
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](
                h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask,
                edge_attr=edge_feat, update_coords_mask=update_coords_mask,
                batch_mask=batch_mask)

        # Output embedding
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x

# Wrapper for compatibility with existing codebase
class CorrectedEGNNBackbone(nn.Module):
    """Wrapper to make EGNN compatible with existing Joint2D3D model"""
    
    def __init__(self, hidden_dim=256, num_layers=4, cutoff=10.0, 
                 sin_embedding=True, reflection_equiv=True):
        super().__init__()
        
        self.cutoff = cutoff
        self.egnn = EGNN(
            in_node_nf=hidden_dim,
            in_edge_nf=1,  # Will be provided by bond features
            hidden_nf=hidden_dim,
            n_layers=num_layers,
            sin_embedding=sin_embedding,
            reflection_equiv=reflection_equiv,
            attention=True,
            tanh=True,
            coords_range=10.0
        )
    
    def forward(self, h: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None,
                edge_attr: Optional[torch.Tensor] = None,
                atom_types: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # Build edge index if not provided
        if edge_index is None:
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=32)
        
        # Prepare batch mask for cross products
        batch_mask = batch if batch is not None else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        
        # Run EGNN
        h_out, pos_out = self.egnn(
            h=h, x=pos, edge_index=edge_index,
            edge_attr=edge_attr, batch_mask=batch_mask
        )
        
        return {
            'h': h_out,
            'pos': pos_out,
            'constraint_losses': {},  # No constraints for now
            'total_constraint_loss': torch.tensor(0.0, device=h.device, requires_grad=True)
        }

# Factory function
def create_corrected_egnn_backbone(hidden_dim=256, num_layers=4, cutoff=10.0, **kwargs):
    """Create corrected EGNN backbone based on DiffSBDD implementation"""
    return CorrectedEGNNBackbone(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        cutoff=cutoff,
        **kwargs
    )