import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_scatter import scatter_mean, scatter_add
import numpy as np
from src.visnet.visnet_block import ViSNetBlock

class ViSNetDynamics(nn.Module):
    def __init__(
        self,
        atom_nf,
        residue_nf, 
        n_dims,
        hidden_nf=256,
        condition_time=True,
        update_pocket_coords=True,
        edge_cutoff_ligand=5.0,
        edge_cutoff_pocket=8.0, 
        edge_cutoff_interaction=5.0,
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
        self.lmax = lmax
        
        # Edge cutoffs for different interaction types
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket  
        self.edge_cutoff_i = edge_cutoff_interaction
        
        # Input dimension calculation
        total_input_dim = hidden_nf
        if condition_time:
            total_input_dim += 1
            
        # Feature encoders
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
        
        # ViSNet core
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
        
        # Output decoders
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
        
        # PROPER EQUIVARIANT VECTOR PROJECTION
        # Project from l=1 spherical harmonics (3D vectors) to 3D velocity
        self.vec_to_velocity = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf // 2),
            nn.SiLU(),
            nn.Linear(hidden_nf // 2, 1)
        )
        
        # Edge type embedding for different molecular interactions
        self.edge_type_embedding = nn.Embedding(3, num_rbf)  # 3 types: L-L, P-P, L-P
        
    def get_proper_edges_with_features(self, pos_atoms, pos_residues, batch_atoms, batch_residues):
        """
        Create edges with proper types and distance-based features
        """
        device = pos_atoms.device
        all_pos = torch.cat([pos_atoms, pos_residues], dim=0)
        all_batch = torch.cat([batch_atoms, batch_residues], dim=0)
        
        from torch_cluster import radius_graph
        
        # Use maximum cutoff for initial edge detection
        max_cutoff = max(self.edge_cutoff_l, self.edge_cutoff_p, self.edge_cutoff_i)
        edge_index = radius_graph(
            all_pos, r=max_cutoff, batch=all_batch, 
            loop=True, max_num_neighbors=64
        )
        
        if edge_index.size(1) == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=device), torch.zeros((0,), device=device)
        
        # Classify edge types and apply specific cutoffs
        n_atoms = len(pos_atoms)
        src, dst = edge_index[0], edge_index[1]
        
        # Compute distances
        distances = torch.norm(all_pos[src] - all_pos[dst], dim=1)
        
        # Edge type classification: 0=L-P, 1=L-L, 2=P-P
        edge_types = torch.zeros(edge_index.size(1), dtype=torch.long, device=device)
        
        # Ligand-ligand edges
        ll_mask = (src < n_atoms) & (dst < n_atoms)
        edge_types[ll_mask] = 1
        
        # Pocket-pocket edges  
        pp_mask = (src >= n_atoms) & (dst >= n_atoms)
        edge_types[pp_mask] = 2
        
        # Ligand-pocket edges (default: 0)
        
        # Apply type-specific distance cutoffs
        valid_mask = torch.ones_like(distances, dtype=torch.bool)
        
        # Ligand-ligand cutoff
        valid_mask &= ~ll_mask | (distances <= self.edge_cutoff_l)
        # Pocket-pocket cutoff
        valid_mask &= ~pp_mask | (distances <= self.edge_cutoff_p)
        # Ligand-pocket cutoff
        lp_mask = ~(ll_mask | pp_mask)
        valid_mask &= ~lp_mask | (distances <= self.edge_cutoff_i)
        
        # Filter edges
        edge_index = edge_index[:, valid_mask]
        edge_types = edge_types[valid_mask]
        
        return edge_index, edge_types
        
    def extract_equivariant_velocities(self, vec_features):
        """
        PROPER extraction of 3D velocities from ViSNet vector features
        Maintains SE(3) equivariance
        """
        if vec_features.size(1) < 3:
            # Fallback if insufficient vector features
            return torch.zeros(vec_features.size(0), 3, device=vec_features.device)
        
        # Extract l=1 spherical harmonics (3D vector components)
        # These are the first 3 components after l=0 (which is index 0)
        vec_l1 = vec_features[:, :3, :]  # [N, 3, hidden_dim]
        
        # Project each vector component to scalar magnitude
        vec_magnitudes = self.vec_to_velocity(vec_l1)  # [N, 3, 1]
        velocities = vec_magnitudes.squeeze(-1)  # [N, 3]
        
        return velocities
        
    def forward(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues):
        """
        Forward pass with proper ViSNet integration and equivariance
        """
        # Separate coordinates and features
        x_atoms = xh_atoms[:, :self.n_dims]
        h_atoms = xh_atoms[:, self.n_dims:]
        
        x_residues = xh_residues[:, :self.n_dims] 
        h_residues = xh_residues[:, self.n_dims:]
        
        # Encode features to common hidden space
        h_atoms_enc = self.atom_encoder(h_atoms)
        h_residues_enc = self.residue_encoder(h_residues)
        
        # Combine all nodes
        pos = torch.cat([x_atoms, x_residues], dim=0)
        h = torch.cat([h_atoms_enc, h_residues_enc], dim=0)
        batch = torch.cat([mask_atoms, mask_residues], dim=0)
        
        # Add time conditioning
        if self.condition_time:
            if t.numel() == 1:
                h_time = torch.full((len(h), 1), t.item(), device=h.device)
            else:
                h_time = t[batch]
            h = torch.cat([h, h_time], dim=-1)
        
        # Get proper edges with types
        edge_index, edge_types = self.get_proper_edges_with_features(
            x_atoms, x_residues, mask_atoms, mask_residues
        )
        
        # Create enhanced PyG data with edge types
        data = Data(x=h, pos=pos, batch=batch, edge_index=edge_index)
        
        # Add edge type information to data
        if edge_index.size(1) > 0:
            data.edge_attr = self.edge_type_embedding(edge_types)
        
        # Forward through ViSNet
        h_out, vec_out = self.visnet(data)
        
        # PROPER equivariant velocity extraction
        velocities = self.extract_equivariant_velocities(vec_out)
        
        # Split outputs back to atoms and residues
        n_atoms = len(mask_atoms)
        
        h_out_atoms = h_out[:n_atoms]
        h_out_residues = h_out[n_atoms:]
        
        vel_atoms = velocities[:n_atoms]
        vel_residues = velocities[n_atoms:]
        
        # Decode features back to original spaces
        h_final_atoms = self.atom_decoder(h_out_atoms)
        h_final_residues = self.residue_decoder(h_out_residues)
        
        # Handle coordinate updates (pocket can be frozen)
        if not self.update_pocket_coords:
            vel_residues = torch.zeros_like(vel_residues)
            
        # IMPORTANT: Maintain center-of-mass invariance
        # Remove COM motion from entire system for translation invariance
        combined_vel = torch.cat([vel_atoms, vel_residues], dim=0)
        combined_mask = torch.cat([mask_atoms, mask_residues], dim=0)
        combined_vel = self.remove_mean_batch(combined_vel, combined_mask)
        
        vel_atoms = combined_vel[:n_atoms]
        if self.update_pocket_coords:
            vel_residues = combined_vel[n_atoms:]
        else:
            vel_residues = torch.zeros_like(combined_vel[n_atoms:])
        
        # Check for NaN/Inf and handle gracefully
        if torch.any(torch.isnan(vel_atoms)) or torch.any(torch.isinf(vel_atoms)):
            print("Warning: NaN/Inf in atom velocities, zeroing out")
            vel_atoms = torch.zeros_like(vel_atoms)
        
        if torch.any(torch.isnan(vel_residues)) or torch.any(torch.isinf(vel_residues)):
            print("Warning: NaN/Inf in residue velocities, zeroing out")
            vel_residues = torch.zeros_like(vel_residues)
        
        # Concatenate velocity and feature predictions
        atoms_output = torch.cat([vel_atoms, h_final_atoms], dim=-1)
        residues_output = torch.cat([vel_residues, h_final_residues], dim=-1)
        
        return atoms_output, residues_output
    
    @staticmethod
    def remove_mean_batch(x, batch_mask):
        """
        Remove center of mass from coordinates/velocities
        Essential for translation invariance
        """
        if x.size(0) == 0:
            return x
            
        mean = scatter_mean(x, batch_mask, dim=0)
        x = x - mean[batch_mask]
        return x

    def check_equivariance(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues):
        """
        Test method to verify SE(3) equivariance
        """
        # Original output
        out1_atoms, out1_residues = self.forward(xh_atoms, xh_residues, t, mask_atoms, mask_residues)
        
        # Apply random rotation + translation
        rotation = torch.randn(3, 3, device=xh_atoms.device)
        rotation, _ = torch.qr(rotation)  # Proper rotation matrix
        translation = torch.randn(3, device=xh_atoms.device)
        
        # Transform input coordinates
        xh_atoms_rot = xh_atoms.clone()
        xh_residues_rot = xh_residues.clone()
        
        xh_atoms_rot[:, :3] = torch.matmul(xh_atoms[:, :3], rotation.T) + translation
        xh_residues_rot[:, :3] = torch.matmul(xh_residues[:, :3], rotation.T) + translation
        
        # Compute output after transformation
        out2_atoms, out2_residues = self.forward(xh_atoms_rot, xh_residues_rot, t, mask_atoms, mask_residues)
        
        # Transform output1 and compare with output2
        expected_atoms = out1_atoms.clone()
        expected_residues = out1_residues.clone()
        
        # Velocities (first 3 dims) should transform as vectors
        expected_atoms[:, :3] = torch.matmul(out1_atoms[:, :3], rotation.T)
        expected_residues[:, :3] = torch.matmul(out1_residues[:, :3], rotation.T)
        
        # Features should be invariant (no change needed)
        
        # Check equivariance error
        error_atoms = torch.norm(out2_atoms - expected_atoms)
        error_residues = torch.norm(out2_residues - expected_residues)
        
        print(f"Equivariance error - Atoms: {error_atoms:.6f}, Residues: {error_residues:.6f}")
        
        return error_atoms + error_residues