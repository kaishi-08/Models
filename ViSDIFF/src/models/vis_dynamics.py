import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_scatter import scatter_mean, scatter_add
import numpy as np
from src.visnet.visnet_block import ViSNetBlock
from src.visnet.output_modules import GatedEquivariantBlock, act_class_mapping
from src.visnet.utils import rbf_class_mapping, CosineCutoff

class EquivariantVector(nn.Module):
    def __init__(self, hidden_nf, n_dims=3, activation="silu"):
        super().__init__()
        self.n_dims = n_dims
        act_class = act_class_mapping[activation]
        self.output_network = nn.ModuleList([
            GatedEquivariantBlock(hidden_nf, hidden_nf // 2, activation=activation, scalar_activation=True),
            GatedEquivariantBlock(hidden_nf // 2, hidden_nf // 4, activation=activation, scalar_activation=True),
        ])
        self.scalar_net = nn.Sequential(
            nn.Linear(hidden_nf // 4, hidden_nf // 8),
            act_class(),
            nn.Linear(hidden_nf // 8, 1),
            nn.Sigmoid()
        )
        self.l1_proj = nn.Linear(hidden_nf // 4, 1)
        self.l2_proj = nn.Sequential(
            nn.Linear(hidden_nf // 4, hidden_nf // 8),
            act_class(),
            nn.Linear(hidden_nf // 8, 1),
            nn.Tanh()
        )
        self.combination_weights = nn.Parameter(torch.tensor([0.8, 0.2], dtype=torch.float64))
        self.to(torch.float64)

    def forward(self, x, v):
        for layer in self.output_network:
            x, v = layer(x, v)
        v_l1 = v[:, :3, :]
        v_l2 = v[:, 3:8, :]
        l1_vector = self.l1_proj(v_l1).squeeze(-1)
        l2_magnitude = torch.norm(v_l2, dim=1)
        l2_modulation = self.l2_proj(l2_magnitude)
        magnitude = self.scalar_net(x)
        w1, w2 = F.softmax(self.combination_weights, dim=0)
        final_vector = (w1 + w2 * l2_modulation) * l1_vector
        final_vector = magnitude * final_vector
        return final_vector

class EquivariantFeature(nn.Module):
    def __init__(self, hidden_nf, out_nf, activation="silu"):
        super().__init__()
        self.output_network = nn.ModuleList([
            GatedEquivariantBlock(hidden_nf, hidden_nf // 2, activation=activation, scalar_activation=True),
            GatedEquivariantBlock(hidden_nf // 2, out_nf, activation=activation),
        ])
        self.to(torch.float64)

    def forward(self, x, v):
        for layer in self.output_network:
            x, v = layer(x, v)
        return x

class ViSNetDynamics(nn.Module):
    def __init__(
        self,
        atom_nf,
        residue_nf, 
        n_dims,
        hidden_nf=256,
        condition_time=True,
        update_pocket_coords=False,
        edge_cutoff_ligand=5.0,
        edge_cutoff_pocket=8.0, 
        edge_cutoff_interaction=5.0,
        lmax=2,
        vecnorm_type='none',
        trainable_vecnorm=True,
        num_heads=8,
        num_layers=6,
        num_rbf=32,
        rbf_type="expnorm",
        trainable_rbf=False,
        activation="silu",
        attn_activation="silu",
        max_num_neighbors=32,
        vertex_type="Edge",
        cutoff=5.0
    ):
        super().__init__()
        self.atom_nf = atom_nf
        self.residue_nf = residue_nf
        self.n_dims = n_dims
        self.hidden_nf = hidden_nf
        self.condition_time = condition_time
        self.update_pocket_coords = update_pocket_coords
        self.lmax = lmax
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket  
        self.edge_cutoff_i = edge_cutoff_interaction
        
        cutoff = max(edge_cutoff_ligand, edge_cutoff_pocket, edge_cutoff_interaction, cutoff)
        
        base_input_dim = hidden_nf
        if condition_time:
            base_input_dim += 1
        
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_nf, hidden_nf // 2),
            nn.LayerNorm(hidden_nf // 2),
            nn.SiLU(),
            nn.Linear(hidden_nf // 2, hidden_nf),
            nn.LayerNorm(hidden_nf)
        )
        self.residue_encoder = nn.Sequential(
            nn.Linear(residue_nf, hidden_nf // 2),
            nn.LayerNorm(hidden_nf // 2),
            nn.SiLU(),
            nn.Linear(hidden_nf // 2, hidden_nf),
            nn.LayerNorm(hidden_nf)
        )
        if condition_time:
            self.time_embedding = SinusoidalTimeEmbedding(hidden_nf)
        self.distance_expansion = rbf_class_mapping[rbf_type](cutoff, num_rbf, trainable_rbf)
        self.edge_type_embedding = nn.Embedding(3, num_rbf)
        self.visnet = ViSNetBlock(
            input_dim=base_input_dim,
            lmax=lmax,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_channels=hidden_nf,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            attn_activation=attn_activation,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            vertex_type=vertex_type
        )
        self.coordinate_head = EquivariantVector(hidden_nf, n_dims)
        self.feature_head_atoms = EquivariantFeature(hidden_nf, atom_nf)
        self.feature_head_residues = EquivariantFeature(hidden_nf, residue_nf)
        self.apply(self._init_weights)
        self.to(torch.float64)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Parameter):
            if module.dtype != torch.long:
                module.data = module.data.to(torch.float64)

    def get_edges_with_types(self, pos_atoms, pos_residues, batch_atoms, batch_residues):
        device = pos_atoms.device
        n_atoms = len(pos_atoms)
        n_residues = len(pos_residues)
        if n_atoms == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=device), torch.zeros((0,), device=device)
        edges_list = []
        types_list = []
        from torch_cluster import radius_graph
        if n_atoms > 1:
            ll_edges = radius_graph(
                pos_atoms, r=self.edge_cutoff_l, batch=batch_atoms,
                loop=False, max_num_neighbors=16
            )
            if ll_edges.size(1) > 0:
                edges_list.append(ll_edges)
                types_list.append(torch.zeros(ll_edges.size(1), dtype=torch.long, device=device))
        if n_residues > 1:
            pp_edges = radius_graph(
                pos_residues, r=self.edge_cutoff_p, batch=batch_residues,
                loop=False, max_num_neighbors=8
            )
            if pp_edges.size(1) > 0:
                pp_edges += n_atoms
                edges_list.append(pp_edges)
                types_list.append(torch.ones(pp_edges.size(1), dtype=torch.long, device=device))
        all_pos = torch.cat([pos_atoms, pos_residues], dim=0)
        all_batch = torch.cat([batch_atoms, batch_residues], dim=0)
        interaction_edges = radius_graph(
            all_pos, r=self.edge_cutoff_i, batch=all_batch,
            loop=False, max_num_neighbors=8
        )
        if interaction_edges.size(1) > 0:
            src, dst = interaction_edges[0], interaction_edges[1]
            cross_mask = ((src < n_atoms) & (dst >= n_atoms)) | ((src >= n_atoms) & (dst < n_atoms))
            if cross_mask.sum() > 0:
                interaction_edges = interaction_edges[:, cross_mask]
                edges_list.append(interaction_edges)
                types_list.append(torch.full((interaction_edges.size(1),), 2, dtype=torch.long, device=device))
        if len(edges_list) == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=device), torch.zeros((0,), device=device)
        edge_index = torch.cat(edges_list, dim=1)
        edge_types = torch.cat(types_list, dim=0)
        return edge_index, edge_types

    def forward(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues):
        if xh_atoms.size(0) == 0:
            empty_atoms = torch.zeros(0, self.n_dims + self.atom_nf, device=xh_atoms.device, dtype=torch.float64)
            empty_residues = torch.zeros(len(xh_residues), self.n_dims + self.residue_nf, device=xh_residues.device, dtype=torch.float64)
            return empty_atoms, empty_residues, {}
        x_atoms = xh_atoms[:, :self.n_dims]
        h_atoms = xh_atoms[:, self.n_dims:]
        x_residues = xh_residues[:, :self.n_dims]
        h_residues = xh_residues[:, self.n_dims:]
        h_atoms_enc = self.atom_encoder(h_atoms)
        h_residues_enc = self.residue_encoder(h_residues)
        pos = torch.cat([x_atoms, x_residues], dim=0)
        h = torch.cat([h_atoms_enc, h_residues_enc], dim=0)
        batch = torch.cat([mask_atoms, mask_residues], dim=0)
        if self.condition_time:
            # Handle both 1D and 2D t tensors
            t = t.squeeze()  # Remove extra dimensions
            t_emb = self.time_embedding(t.expand(len(h)))
            h = torch.cat([h, t_emb], dim=-1)
        edge_index, edge_types = self.get_edges_with_types(
            x_atoms, x_residues, mask_atoms, mask_residues
        )
        if edge_index.size(1) > 0:
            edge_vec_temp = pos[edge_index[0]] - pos[edge_index[1]]
            edge_weight = torch.norm(edge_vec_temp, dim=-1)
            edge_rbf = self.distance_expansion(edge_weight)
            edge_type_emb = self.edge_type_embedding(edge_types)
            edge_attr = edge_rbf + edge_type_emb
        else:
            edge_attr = torch.empty(0, self.visnet.num_rbf, device=h.device, dtype=torch.float64)
        data = Data(x=h, pos=pos, batch=batch, edge_index=edge_index, edge_attr=edge_attr)
        n_atoms = len(mask_atoms)
        try:
            h_out, vec_out = self.visnet(data)
            coord_predictions = self.coordinate_head(h_out, vec_out)
            h_out_atoms = h_out[:n_atoms]
            vec_out_atoms = vec_out[:n_atoms]
            h_out_residues = h_out[n_atoms:]
            vec_out_residues = vec_out[n_atoms:]
            feat_pred_atoms = self.feature_head_atoms(h_out_atoms, vec_out_atoms)
            feat_pred_residues = self.feature_head_residues(h_out_residues, vec_out_residues)
            coord_pred_atoms = coord_predictions[:n_atoms]
            coord_pred_residues = coord_predictions[n_atoms:]
        except Exception as e:
            print(f"ViSNet forward failed: {e}")
            n_total = len(h)
            coord_predictions = torch.zeros(n_total, 3, device=h.device, dtype=torch.float64)
            feat_pred_atoms = torch.zeros(n_atoms, self.atom_nf, device=h.device, dtype=torch.float64)
            feat_pred_residues = torch.zeros(len(mask_residues), self.residue_nf, device=h.device, dtype=torch.float64)
            coord_pred_atoms = coord_predictions[:n_atoms]
            coord_pred_residues = coord_predictions[n_atoms:]
        if not self.update_pocket_coords:
            coord_pred_residues = torch.zeros_like(coord_pred_residues)
        if coord_pred_atoms.size(0) > 0:
            combined_coords = torch.cat([coord_pred_atoms, coord_pred_residues], dim=0)
            combined_mask = torch.cat([mask_atoms, mask_residues], dim=0)
            combined_coords = self.remove_mean_batch(combined_coords, combined_mask)
            coord_pred_atoms = combined_coords[:n_atoms]
            if self.update_pocket_coords:
                coord_pred_residues = combined_coords[n_atoms:]
            else:
                coord_pred_residues = torch.zeros_like(combined_coords[n_atoms:])
        atoms_noise = torch.cat([coord_pred_atoms, feat_pred_atoms], dim=-1)
        residues_noise = torch.cat([coord_pred_residues, feat_pred_residues], dim=-1)
        analysis_info = {
            'edge_count': edge_index.size(1) if edge_index.size(1) > 0 else 0,
            'atom_count': n_atoms,
            'residue_count': len(mask_residues),
            'coord_magnitude': torch.norm(coord_pred_atoms).item() if coord_pred_atoms.size(0) > 0 else 0.0
        }
        return atoms_noise, residues_noise, analysis_info

    @staticmethod
    def remove_mean_batch(x, batch_mask):
        if x.size(0) == 0:
            return x
        mean = scatter_mean(x, batch_mask, dim=0)
        x = x - mean[batch_mask]
        return x

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1).to(torch.float64)

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float64) * -embeddings)
        embeddings = time.unsqueeze(-1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return self.linear(embeddings)