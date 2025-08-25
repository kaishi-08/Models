import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_scatter import scatter_mean, scatter_add
import numpy as np
from visnet.visnet_block import ViSNetBlock

class ViSNetDynamics(nn.Module):

    def __init__(
        self,
        atom_nf,
        residue_nf, 
        n_dims,
        hidden_nf=256,
        condition_time=True,
        update_pocket_coords=True,
        edge_cutoff_ligand=None,
        edge_cutoff_pocket=None, 
        edge_cutoff_interaction=None,
        # ViSNet specific parameters
        lmax=2,
        vecnorm_type='max_min',
        trainable_vecnorm=True,
        num_heads=8,
        num_layers=6,
        num_rbf=32,
        rbf_type="expnorm",
        trainable_rbf=False,
        activation="silu",
        attn_activation="silu",
        cutoff=5.0,
        max_num_neighbors=32,
        vertex_type="Edge"
    ):
        super().__init__()
        
        self.atom_nf = atom_nf
        self.residue_nf = residue_nf
        self.n_dims = n_dims
        self.hidden_nf = hidden_nf
        self.condition_time = condition_time
        self.update_pocket_coords = update_pocket_coords
        
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket  
        self.edge_cutoff_i = edge_cutoff_interaction
        
        total_input_dim = hidden_nf
        if condition_time:
            total_input_dim += 1
            
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf)
        )
        
        self.residue_encoder = nn.Sequential(
            nn.Linear(residue_nf, hidden_nf),
            nn.SiLU(), 
            nn.Linear(hidden_nf, hidden_nf)
        )
        
        self.visnet = ViSNetBlock(
            input_dim=total_input_dim,
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
        
        self.atom_decoder = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, atom_nf)
        )
        
        self.residue_decoder = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, residue_nf)
        )
        

        self.vel_proj = nn.Linear(hidden_nf, 1)
        
    def get_edges_with_types(self, pos_atoms, pos_residues, batch_atoms, batch_residues):
        """
        Create edges with different cutoffs for different interaction types.
        Returns edge_index and edge_types.
        """
        device = pos_atoms.device
        all_pos = torch.cat([pos_atoms, pos_residues], dim=0)
        all_batch = torch.cat([batch_atoms, batch_residues], dim=0)
        
        # Get all edges within max cutoff
        max_cutoff = max(
            self.edge_cutoff_l or 5.0,
            self.edge_cutoff_p or 5.0, 
            self.edge_cutoff_i or 5.0
        )
        
        from torch_cluster import radius_graph
        edge_index = radius_graph(
            all_pos, r=max_cutoff, batch=all_batch, 
            loop=True, max_num_neighbors=64
        )
        
        # Determine edge types and apply specific cutoffs
        n_atoms = len(pos_atoms)
        n_total = len(all_pos)
        
        src, dst = edge_index[0], edge_index[1]
        
        # Edge type classification
        # 0: ligand-pocket, 1: ligand-ligand, 2: pocket-pocket
        edge_types = torch.zeros(edge_index.size(1), dtype=torch.long, device=device)
        
        # Ligand-ligand edges (both < n_atoms)
        ll_mask = (src < n_atoms) & (dst < n_atoms)
        edge_types[ll_mask] = 1
        
        # Pocket-pocket edges (both >= n_atoms)  
        pp_mask = (src >= n_atoms) & (dst >= n_atoms)
        edge_types[pp_mask] = 2
        
        # Ligand-pocket edges (mixed)
        lp_mask = ~(ll_mask | pp_mask)
        edge_types[lp_mask] = 0
        
        # Apply distance cutoffs
        distances = torch.norm(all_pos[src] - all_pos[dst], dim=1)
        valid_mask = torch.ones_like(distances, dtype=torch.bool)
        
        if self.edge_cutoff_l is not None:
            valid_mask &= ~ll_mask | (distances <= self.edge_cutoff_l)
        if self.edge_cutoff_p is not None:
            valid_mask &= ~pp_mask | (distances <= self.edge_cutoff_p)
        if self.edge_cutoff_i is not None:
            valid_mask &= ~lp_mask | (distances <= self.edge_cutoff_i)
            
        edge_index = edge_index[:, valid_mask]
        edge_types = edge_types[valid_mask]
        
        return edge_index, edge_types
        
    def forward(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues):

        # Separate coordinates and features
        x_atoms = xh_atoms[:, :self.n_dims]  # [N_atoms, 3]
        h_atoms = xh_atoms[:, self.n_dims:]  # [N_atoms, atom_nf]
        
        x_residues = xh_residues[:, :self.n_dims]  # [N_residues, 3] 
        h_residues = xh_residues[:, self.n_dims:]  # [N_residues, residue_nf]
        
        # Encode features to common space
        h_atoms_enc = self.atom_encoder(h_atoms)      # [N_atoms, hidden_nf]
        h_residues_enc = self.residue_encoder(h_residues)  # [N_residues, hidden_nf]
        
        # Combine all nodes
        pos = torch.cat([x_atoms, x_residues], dim=0)     # [N_total, 3]
        h = torch.cat([h_atoms_enc, h_residues_enc], dim=0)  # [N_total, hidden_nf]
        batch = torch.cat([mask_atoms, mask_residues], dim=0)  # [N_total]
        
        # Add time conditioning
        if self.condition_time:
            if t.numel() == 1:
                # Same time for all nodes
                h_time = torch.full((len(h), 1), t.item(), device=h.device)
            else:
                # Different time per batch
                h_time = t[batch]  # [N_total, 1]
            h = torch.cat([h, h_time], dim=-1)  # [N_total, hidden_nf + 1]
        
        # Create PyG data object
        data = Data(x=h, pos=pos, batch=batch)
        
        # Forward through ViSNet
        h_out, vec_out = self.visnet(data)  # h_out: [N_total, hidden_nf], vec_out: [N_total, lmax_dim, hidden_nf]
        
        # Extract velocities from vector features
        # Use l=1 spherical harmonics (first 3 components) for 3D velocity
        """
            if vec_out.size(1) >= 3:
            vec_l1 = vec_out[:, :3, :]  # [N_total, 3, hidden_nf]
            # Project to scalar and use as velocity magnitude
            vel_magnitude = self.vel_proj(vec_l1).squeeze(-1)  # [N_total, 3]
        else:
            # Fallback: predict velocity from scalar features
            vel_magnitude = torch.zeros(len(h_out), 3, device=h_out.device)"""
 
        if vec_out.size(1) >=3:
            vec_l1 = vec_out
            vel_magnitude = self.vec_proj(vec_l1).squeeze(-1)

        else: 
            vel_magnitude = torch.zeros(len(h_out), 3, device=h_out.device)
    
        
        # Split outputs back to atoms and residues
        n_atoms = len(mask_atoms)
        
        h_out_atoms = h_out[:n_atoms]        # [N_atoms, hidden_nf]
        h_out_residues = h_out[n_atoms:]     # [N_residues, hidden_nf]
        
        vel_atoms = vel_magnitude[:n_atoms]      # [N_atoms, 3]
        vel_residues = vel_magnitude[n_atoms:]   # [N_residues, 3]
        
        # Decode features back to original spaces
        h_final_atoms = self.atom_decoder(h_out_atoms)        # [N_atoms, atom_nf]
        h_final_residues = self.residue_decoder(h_out_residues)  # [N_residues, residue_nf]
        
        # Handle coordinate updates
        if not self.update_pocket_coords:
            vel_residues = torch.zeros_like(vel_residues)
            
        # Check for NaN values
        if torch.any(torch.isnan(vel_atoms)) or torch.any(torch.isnan(vel_residues)):
            if self.training:
                vel_atoms[torch.isnan(vel_atoms)] = 0.0
                vel_residues[torch.isnan(vel_residues)] = 0.0
            else:
                raise ValueError("NaN detected in ViSNet output")
        
        # Remove center of mass from velocities (for translation invariance)
        if self.update_pocket_coords:
            combined_vel = torch.cat([vel_atoms, vel_residues], dim=0)
            combined_mask = torch.cat([mask_atoms, mask_residues], dim=0)
            combined_vel = self.remove_mean_batch(combined_vel, combined_mask)
            vel_atoms = combined_vel[:n_atoms]
            vel_residues = combined_vel[n_atoms:]
        
        # Concatenate velocity and features
        atoms_output = torch.cat([vel_atoms, h_final_atoms], dim=-1)
        residues_output = torch.cat([vel_residues, h_final_residues], dim=-1)
        
        return atoms_output, residues_output
    
    @staticmethod
    def remove_mean_batch(x, batch_mask):
        """Remove center of mass from coordinates/velocities"""
        mean = scatter_mean(x, batch_mask, dim=0)
        x = x - mean[batch_mask]
        return x


# ===================================================================
# Modified ViSNetBlock with edge type support
# ===================================================================

class ModifiedViSNetBlock(nn.Module):
    """
    Modified ViSNet block that can handle edge types for different molecular interactions.
    """
    def __init__(self, input_dim, edge_type_dim=3, **visnet_kwargs):
        super().__init__()
        
        # Store original ViSNet
        self.visnet = ViSNetBlock(input_dim=input_dim, **visnet_kwargs)
        
        # Edge type embedding
        self.edge_type_embedding = nn.Embedding(edge_type_dim, visnet_kwargs.get('num_rbf', 32))
        
    def forward(self, data, edge_types=None):
        """
        Forward pass with optional edge type information.
        
        Args:
            data: PyG Data object
            edge_types: [num_edges] tensor with edge type indices
        """
        if edge_types is not None:
            # Modify edge attributes to include type information
            # This would require modifying the internal ViSNet implementation
            pass
            
        return self.visnet(data)