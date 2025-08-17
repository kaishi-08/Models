import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import radius_graph
from typing import Dict, Optional, Tuple, Union

def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
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
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff

def coord2cross(x, edge_index, batch_mask, norm_constant=1):
    mean = unsorted_segment_sum(x, batch_mask,
                              num_segments=batch_mask.max() + 1,
                              normalization_factor=None,
                              aggregation_method='mean')
    row, col = edge_index
    cross = torch.cross(x[row] - mean[batch_mask[row]],
                       x[col] - mean[batch_mask[col]], dim=1)
    norm = torch.linalg.norm(cross, dim=1, keepdim=True)
    cross = cross / (norm + norm_constant)
    return cross

class SinusoidsEmbeddingNew(nn.Module):
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
        input_edge = input_nf * 2 # Source + target features
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:
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
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij

class ChemicalConstraints(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_valence = {6: 4, 7: 3, 8: 2, 16: 6, 9: 1, 17: 1, 35: 1, 53: 1, 15: 5, 1: 1}
        self.target_lengths = {
            (6, 6): 1.54, (6, 7): 1.47, (6, 8): 1.43, (6, 16): 1.82, (6, 9): 1.35,
            (6, 17): 1.77, (6, 1): 1.09, (7, 7): 1.45, (7, 8): 1.40, (7, 1): 1.01,
            (8, 8): 1.48, (8, 1): 0.96, (16, 16): 2.05, (15, 8): 1.63
        }
        self.vdw_radii = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98}
        self.default_length = 1.50
        self.default_radius = 1.60

    def forward(self, pos, edge_index, atom_types=None):
        if edge_index.size(1) == 0 or atom_types is None or pos.size(0) <= 1:
            return pos, torch.tensor(0.0, device=pos.device, requires_grad=True)
        
        pos_new = pos.clone()
        total_loss = 0.0
        
        # Valence constraints
        row, col = edge_index
        bond_counts = torch.zeros(pos.size(0), device=pos.device)
        bond_counts.index_add_(0, row, torch.ones(row.size(0), device=pos.device))
        
        for i, atom_type in enumerate(atom_types):
            max_val = self.max_valence.get(atom_type.item(), 4)
            if bond_counts[i] > max_val:
                violation = bond_counts[i] - max_val
                total_loss += violation ** 2
                neighbors = col[row == i]
                if len(neighbors) > 0:
                    for neighbor_idx in neighbors:
                        direction = pos_new[i] - pos_new[neighbor_idx]
                        distance = torch.norm(direction) + 1e-8
                        force = direction / distance * violation * 0.001
                        pos_new[i] += force
        
        # Bond length constraints
        bond_vectors = pos_new[row] - pos_new[col]
        current_lengths = torch.norm(bond_vectors, dim=-1, keepdim=True)
        target_lengths = torch.full_like(current_lengths, self.default_length)
        
        for i in range(edge_index.size(1)):
            atom1_type = atom_types[row[i]].item()
            atom2_type = atom_types[col[i]].item()
            bond_key = tuple(sorted([atom1_type, atom2_type]))
            target_lengths[i] = self.target_lengths.get(bond_key, self.default_length)
        
        length_errors = torch.abs(current_lengths - target_lengths)
        total_loss += torch.mean(length_errors ** 2)
        
        length_ratios = target_lengths / (current_lengths + 1e-8)
        length_ratios = torch.clamp(length_ratios, 0.98, 1.02)
        adjusted_vectors = bond_vectors * length_ratios
        adjustment = (adjusted_vectors - bond_vectors) * 0.01
        
        pos_adjustment = torch.zeros_like(pos)
        pos_adjustment.index_add_(0, row, adjustment * 0.5)
        pos_adjustment.index_add_(0, col, -adjustment * 0.5)
        pos_new += pos_adjustment
        
        # Steric constraints
        if pos.size(0) > 1:
            distances = torch.cdist(pos_new, pos_new)
            radii = torch.tensor([self.vdw_radii.get(at.item(), self.default_radius) for at in atom_types], device=pos.device)
            min_distances = (radii.unsqueeze(0) + radii.unsqueeze(1)) * 0.8
            
            mask = torch.eye(pos.size(0), device=pos.device).bool()
            distances.masked_fill_(mask, float('inf'))
            
            clashes = distances < min_distances
            total_loss += torch.sum(torch.relu(min_distances - distances) ** 2) / 2
            
            clash_indices = torch.where(clashes)
            steric_adjustment = torch.zeros_like(pos)
            for i, j in zip(clash_indices[0], clash_indices[1]):
                if i >= j:
                    continue
                direction = pos_new[i] - pos_new[j]
                distance = distances[i, j]
                min_dist = min_distances[i, j]
                if distance < min_dist and distance > 1e-8:
                    force_magnitude = (min_dist - distance) * 0.005
                    force_direction = direction / distance
                    steric_adjustment[i] += force_direction * force_magnitude * 0.5
                    steric_adjustment[j] -= force_direction * force_magnitude * 0.5
            pos_new += steric_adjustment
        
        return pos_new, total_loss * 0.1

class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0,
                 reflection_equiv=True, enable_chemical_constraints=True):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        self.reflection_equiv = reflection_equiv
        self.enable_chemical_constraints = enable_chemical_constraints
        input_edge = hidden_nf * 2 + edges_in_d
        
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)
        
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
        
        if enable_chemical_constraints:
            self.chemical_constraints = ChemicalConstraints()

    def coord_model(self, h, coord, edge_index, coord_diff, coord_cross,
                   edge_attr, edge_mask, update_coords_mask=None, atom_types=None):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        
        coord_weights = self.coord_mlp(input_tensor)
        if self.tanh:
            trans = coord_diff * torch.tanh(coord_weights) * self.coords_range
        else:
            trans = coord_diff * coord_weights

        if not self.reflection_equiv and self.cross_product_mlp is not None:
            phi_cross = self.cross_product_mlp(input_tensor)
            if self.tanh:
                phi_cross = torch.tanh(phi_cross) * self.coords_range
            trans = trans + coord_cross * phi_cross

        if edge_mask is not None:
            trans = trans * edge_mask

        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                 normalization_factor=self.normalization_factor,
                                 aggregation_method=self.aggregation_method)

        if update_coords_mask is not None:
            agg = update_coords_mask * agg

        coord = coord + agg
        
        # Apply chemical constraints
        if self.enable_chemical_constraints and hasattr(self, 'chemical_constraints') and atom_types is not None:
            coord, constraint_loss = self.chemical_constraints(coord, edge_index, atom_types)
            self.constraint_loss = constraint_loss
        else:
            self.constraint_loss = torch.tensor(0.0, device=coord.device)
        
        return coord

    def forward(self, h, coord, edge_index, coord_diff, coord_cross,
               edge_attr=None, node_mask=None, edge_mask=None,
               update_coords_mask=None, atom_types=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, coord_cross,
                               edge_attr, edge_mask, update_coords_mask, atom_types)
        if node_mask is not None:
            coord = coord * node_mask
        return coord

class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum', reflection_equiv=True,
                 enable_chemical_constraints=True):
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
        self.enable_chemical_constraints = enable_chemical_constraints

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, 
                                            edges_in_d=edge_feat_nf,
                                            act_fn=act_fn, attention=attention,
                                            normalization_factor=self.normalization_factor,
                                            aggregation_method=self.aggregation_method))
        
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, 
                                                     act_fn=nn.SiLU(), tanh=tanh,
                                                     coords_range=self.coords_range_layer,
                                                     normalization_factor=self.normalization_factor,
                                                     aggregation_method=self.aggregation_method,
                                                     reflection_equiv=self.reflection_equiv,
                                                     enable_chemical_constraints=enable_chemical_constraints))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None,
               edge_attr=None, update_coords_mask=None, batch_mask=None, atom_types=None):
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        
        if self.reflection_equiv:
            coord_cross = None
        else:
            coord_cross = coord2cross(x, edge_index, batch_mask, self.norm_constant)
        
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        
        if edge_attr is not None:
            edge_attr = torch.cat([distances, edge_attr], dim=1)
        else:
            edge_attr = distances
        
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr,
                                             node_mask=node_mask, edge_mask=edge_mask)
        
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, coord_cross, edge_attr,
                                     node_mask, edge_mask, update_coords_mask, atom_types)

        if node_mask is not None:
            h = h * node_mask
        return h, x

class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, 
                 attention=False, norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, 
                 norm_constant=1, inv_sublayers=2, sin_embedding=False, normalization_factor=100, 
                 aggregation_method='sum', reflection_equiv=True, enable_chemical_constraints=True):
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
        self.enable_chemical_constraints = enable_chemical_constraints

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim
        else:
            self.sin_embedding = None
            edge_feat_nf = 1

        edge_feat_nf = edge_feat_nf + in_edge_nf

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        
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
                                                             reflection_equiv=self.reflection_equiv,
                                                             enable_chemical_constraints=enable_chemical_constraints))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, 
               update_coords_mask=None, batch_mask=None, edge_attr=None, atom_types=None):

        edge_feat, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            edge_feat = self.sin_embedding(edge_feat)
        if edge_attr is not None:
            edge_feat = torch.cat([edge_feat, edge_attr], dim=1)
        
        h = self.embedding(h)
        
        total_constraint_loss = 0.0
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](
                h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask,
                edge_attr=edge_feat, update_coords_mask=update_coords_mask,
                batch_mask=batch_mask, atom_types=atom_types)
            
            if hasattr(self._modules["e_block_%d" % i]._modules["gcl_equiv"], 'constraint_loss'):
                total_constraint_loss += self._modules["e_block_%d" % i]._modules["gcl_equiv"].constraint_loss

        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
            
        self.total_constraint_loss = total_constraint_loss
        return h, x

class EGNNBackbone(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=4, cutoff=10.0, 
                 sin_embedding=True, reflection_equiv=True, enable_chemical_constraints=True):
        super().__init__()
        
        self.cutoff = cutoff
        self.enable_chemical_constraints = enable_chemical_constraints
        self.egnn = EGNN(
            in_node_nf=hidden_dim,
            in_edge_nf=1,
            hidden_nf=hidden_dim,
            n_layers=num_layers,
            sin_embedding=sin_embedding,
            reflection_equiv=reflection_equiv,
            attention=True,
            tanh=True,
            coords_range=10.0,
            enable_chemical_constraints=enable_chemical_constraints
        )
    
    def forward(self, h: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None,
                edge_attr: Optional[torch.Tensor] = None,
                atom_types: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        if edge_index is None:
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=32)
        
        batch_mask = batch if batch is not None else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        
        h_out, pos_out = self.egnn(
            h=h, x=pos, edge_index=edge_index,
            edge_attr=edge_attr, batch_mask=batch_mask, atom_types=atom_types
        )
        
        constraint_losses = {}
        total_constraint_loss = getattr(self.egnn, 'total_constraint_loss', torch.tensor(0.0, device=h.device))
        
        return {
            'h': h_out,
            'pos': pos_out,
            'constraint_losses': constraint_losses,
            'total_constraint_loss': total_constraint_loss
        }

def egnn_backbone(hidden_dim=256, num_layers=4, cutoff=10.0, enable_chemical_constraints=True, **kwargs):
    return EGNNBackbone(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        cutoff=cutoff,
        enable_chemical_constraints=enable_chemical_constraints,
        **kwargs
    )