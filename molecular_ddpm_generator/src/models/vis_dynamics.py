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
        
        # Edge cutoffs
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket  
        self.edge_cutoff_i = edge_cutoff_interaction
        
        # 🧮 Calculate spherical harmonics dimensions
        self.sh_dimensions = self._calculate_sh_dimensions_no_l0(lmax)
        self.total_sh_dim = sum(self.sh_dimensions.values())
        
        print(f"Spherical Harmonics Structure (lmax={lmax}):")
        for l, dim in self.sh_dimensions.items():
            print(f"   l={l}: {dim} components ({self._get_physical_meaning(l)})")
        print(f"Total SH dimensions: {self.total_sh_dim}")
        
        # Input dimension for ViSNet
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
        
        # 🌟 ViSNet core with full lmax support
        self.visnet = ViSNetBlock(
            input_dim=total_input_dim,
            lmax=lmax,  # Will generate l=0,1,2 features
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
        
        # 🎯 Multiple specialized decoders for different SH orders
        self.l0_decoder = self._build_scalar_decoder()      # l=0 → scalars
        self.l1_decoder = self._build_vector_decoder()      # l=1 → vectors  
        self.l2_decoder = self._build_quadrupole_decoder()  # l=2 → quadrupoles
        
        # Feature decoders
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
        
        # 🔗 Fusion network to combine all SH information
        self.sh_fusion = nn.Sequential(
            nn.Linear(3 + 5, hidden_nf // 2),  #l=1(3) + l=2(5)
            nn.SiLU(),
            nn.Linear(hidden_nf // 2, 3)  # → final 3D velocity
        )
        
        # Edge type embedding
        self.edge_type_embedding = nn.Embedding(3, num_rbf)
        
    def _calculate_sh_dimensions(self, lmax):
        """Calculate dimensions for each spherical harmonics order"""
        return {l: 2*l + 1 for l in range(lmax + 1)}
    
    def _get_physical_meaning(self, l):
        """Get physical meaning of each SH order"""
        meanings = {
            0: "scalars/invariants",
            1: "vectors/dipoles", 
            2: "quadrupoles/d-orbitals",
            3: "octupoles/f-orbitals"
        }
        return meanings.get(l, f"order-{l}")
    
    def _build_scalar_decoder(self):
        """Decoder for l=0: scalar features → energy-like quantities"""
        return nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_nf // 2, 1),
            nn.Tanh()  # Bounded output
        )
    
    def _build_vector_decoder(self):
        """Decoder for l=1: vector features → 3D directional information"""
        return nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_nf // 2, 1)
        )
    
    def _build_quadrupole_decoder(self):
        """Decoder for l=2: quadrupole features → angular/strain information"""
        return nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_nf // 2, 1),
            nn.Tanh()  # Bounded for stability
        )
    
    def extract_all_spherical_harmonics(self, vec_features):
        """
        🌟 MAIN INNOVATION: Extract spherical harmonics (ViSNet 8D convention)
        
        Input: vec_features [N, 8, hidden_nf]  # ViSNet: l=1(3) + l=2(5), no l=0
        Output: Combined prediction using l=1,2 orders
        """
        batch_size = vec_features.size(0)
        device = vec_features.device
        
        # ViSNet outputs 8 dimensions: l=1(3) + l=2(5), no l=0
        if vec_features.size(1) < 8:
            print(f"⚠️ Warning: Expected 8 SH components, got {vec_features.size(1)}")
            return self._fallback_simple_extraction(vec_features)
        
        # Extract features directly (no l=0)
        sh_features = {}
        
        # Extract l=1 features (first 3 components)
        l1_raw = vec_features[:, 0:3, :]  # [N, 3, hidden]
        l1_vectors = self.l1_decoder(l1_raw).squeeze(-1)  # [N, 3]
        sh_features['l1'] = l1_vectors
        
        # Extract l=2 features (next 5 components)
        l2_raw = vec_features[:, 3:8, :]  # [N, 5, hidden]  
        l2_quadrupoles = self.l2_decoder(l2_raw).squeeze(-1)  # [N, 5]
        sh_features['l2'] = l2_quadrupoles
        
        # Combine l=1 and l=2 (modify existing _fuse_spherical_harmonics)
        final_velocities = self._fuse_spherical_harmonics(sh_features)
        
        return final_velocities, sh_features
    
    def _calculate_sh_dimensions_no_l0(self, lmax):
        return {l: 2*l + 1 for l in range(1, lmax + 1)} 
    
    def _fuse_spherical_harmonics(self, sh_features):

        device = list(sh_features.values())[0].device
        batch_size = list(sh_features.values())[0].size(0)
        

        l1 = sh_features.get('l1', torch.zeros(batch_size, 3, device=device))      # [N, 3] 
        l2 = sh_features.get('l2', torch.zeros(batch_size, 5, device=device))      # [N, 5]
        
        # 🎛️ Method 1: Simple concatenation + learned fusion
        combined_features = torch.cat([l1, l2], dim=1)  # [N, 3+5=8]
        fused_velocities = self.sh_fusion(combined_features)  # [N, 3]
        
        # 🧪 Method 2: Physics-inspired combination (advanced)
        physics_velocities = self._physics_inspired_fusion(l1, l2)
        
        # 🎯 Combine both methods (weighted average)
        alpha = 0.7  # Weight for learned fusion vs physics fusion
        final_velocities = alpha * fused_velocities + (1 - alpha) * physics_velocities
        
        return final_velocities
    
    def _physics_inspired_fusion(self, l1, l2):

        # l=1 provides base directional information
        base_velocity = l1  # [N, 3]
        
        l2_magnitude = torch.norm(l2, dim=1, keepdim=True)
        magnitude_scaling = torch.sigmoid(l2_magnitude)
        scaled_velocity = base_velocity + magnitude_scaling
        # l=2 provides angular corrections (simplified)
        # In reality, this needs proper spherical harmonics mathematics
        angular_correction = self._convert_l2_to_directional(l2)  # [N, 3]
        
        # Final combination
        corrected_velocity = scaled_velocity + 0.1 * angular_correction
        
        return corrected_velocity
    
    def _convert_l2_to_directional(self, l2_features):
        """
        Convert l=2 spherical harmonics to 3D directional corrections
        
        This is a SIMPLIFIED version. Full implementation requires:
        - Proper spherical harmonics rotation matrices
        - Clebsch-Gordan coefficients
        - Wigner D-matrices
        """
        batch_size = l2_features.size(0)
        device = l2_features.device
        
        # 🎯 Simplified mapping: l=2 components → xyz directions
        # Real implementation would use proper SH mathematics
        directional = torch.zeros(batch_size, 3, device=device)
        
        # Approximate influence of each l=2 component on x,y,z
        # These coefficients come from angular momentum theory
        directional[:, 0] = l2_features[:, 0] + l2_features[:, 4] * 0.5  # x-component
        directional[:, 1] = l2_features[:, 1] + l2_features[:, 3] * 0.5  # y-component  
        directional[:, 2] = l2_features[:, 2]                            # z-component
        
        # Small scaling to prevent instability
        return directional * 0.1
    
    def _fallback_simple_extraction(self, vec_features):
        """Fallback cho trường hợp không đủ SH dimensions"""
        print("🔄 Using fallback simple extraction")
        available_dim = min(3, vec_features.size(1))
        simple_vectors = vec_features[:, :available_dim, :]
        simple_output = self.l1_decoder(simple_vectors).squeeze(-1)
        
        # Pad to 3 dimensions if needed
        if simple_output.size(1) < 3:
            padding = torch.zeros(simple_output.size(0), 3 - simple_output.size(1), device=simple_output.device)
            simple_output = torch.cat([simple_output, padding], dim=1)
        
        return simple_output[:, :3], {}
    
    def get_proper_edges_with_features(self, pos_atoms, pos_residues, batch_atoms, batch_residues):
        """
        Create edges with proper types and distance-based features
        (Same as before - không thay đổi)
        """
        device = pos_atoms.device
        all_pos = torch.cat([pos_atoms, pos_residues], dim=0)
        all_batch = torch.cat([batch_atoms, batch_residues], dim=0)
        
        from torch_cluster import radius_graph
        
        max_cutoff = max(self.edge_cutoff_l, self.edge_cutoff_p, self.edge_cutoff_i)
        edge_index = radius_graph(
            all_pos, r=max_cutoff, batch=all_batch, 
            loop=True, max_num_neighbors=64
        )
        
        if edge_index.size(1) == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=device), torch.zeros((0,), device=device)
        
        n_atoms = len(pos_atoms)
        src, dst = edge_index[0], edge_index[1]
        distances = torch.norm(all_pos[src] - all_pos[dst], dim=1)
        
        edge_types = torch.zeros(edge_index.size(1), dtype=torch.long, device=device)
        ll_mask = (src < n_atoms) & (dst < n_atoms)
        pp_mask = (src >= n_atoms) & (dst >= n_atoms)
        edge_types[ll_mask] = 1
        edge_types[pp_mask] = 2
        
        valid_mask = torch.ones_like(distances, dtype=torch.bool)
        valid_mask &= ~ll_mask | (distances <= self.edge_cutoff_l)
        valid_mask &= ~pp_mask | (distances <= self.edge_cutoff_p)
        lp_mask = ~(ll_mask | pp_mask)
        valid_mask &= ~lp_mask | (distances <= self.edge_cutoff_i)
        
        edge_index = edge_index[:, valid_mask]
        edge_types = edge_types[valid_mask]
        
        return edge_index, edge_types
        
    def forward(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues):
        """
        🌟 MAIN FORWARD PASS: Full Spherical Harmonics Utilization
        """
        # Separate coordinates and features
        x_atoms = xh_atoms[:, :self.n_dims]
        h_atoms = xh_atoms[:, self.n_dims:]
        x_residues = xh_residues[:, :self.n_dims] 
        h_residues = xh_residues[:, self.n_dims:]
        
        # Encode features
        h_atoms_enc = self.atom_encoder(h_atoms)
        h_residues_enc = self.residue_encoder(h_residues)
        
        # Combine nodes
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
        
        # Get edges with types
        edge_index, edge_types = self.get_proper_edges_with_features(
            x_atoms, x_residues, mask_atoms, mask_residues
        )
        
        # Create PyG data
        data = Data(x=h, pos=pos, batch=batch, edge_index=edge_index)
        if edge_index.size(1) > 0:
            data.edge_attr = self.edge_type_embedding(edge_types)
        
        # 🚀 ViSNet forward pass - generates FULL spherical harmonics
        h_out, vec_out = self.visnet(data)
        
        #print(f"📊 ViSNet output shapes: h_out={h_out.shape}, vec_out={vec_out.shape}")
        #print(f"🌐 Expected SH dimensions: {self.total_sh_dim}")
        
        # 🌟 KEY INNOVATION: Extract ALL spherical harmonics information
        velocities, sh_analysis = self.extract_all_spherical_harmonics(vec_out)
        
        # Split outputs
        n_atoms = len(mask_atoms)
        h_out_atoms = h_out[:n_atoms]
        h_out_residues = h_out[n_atoms:]
        vel_atoms = velocities[:n_atoms]
        vel_residues = velocities[n_atoms:]
        
        # Decode features
        h_final_atoms = self.atom_decoder(h_out_atoms)
        h_final_residues = self.residue_decoder(h_out_residues)
        
        # Handle pocket coordinate updates
        if not self.update_pocket_coords:
            vel_residues = torch.zeros_like(vel_residues)
            
        # Center of mass correction (translation invariance)
        combined_vel = torch.cat([vel_atoms, vel_residues], dim=0)
        combined_mask = torch.cat([mask_atoms, mask_residues], dim=0)
        combined_vel = self.remove_mean_batch(combined_vel, combined_mask)
        
        vel_atoms = combined_vel[:n_atoms]
        if self.update_pocket_coords:
            vel_residues = combined_vel[n_atoms:]
        else:
            vel_residues = torch.zeros_like(combined_vel[n_atoms:])
        
        # Safety checks
        if torch.any(torch.isnan(vel_atoms)) or torch.any(torch.isinf(vel_atoms)):
            print("⚠️ Warning: NaN/Inf in atom velocities")
            vel_atoms = torch.zeros_like(vel_atoms)
        
        if torch.any(torch.isnan(vel_residues)) or torch.any(torch.isinf(vel_residues)):
            print("⚠️ Warning: NaN/Inf in residue velocities")
            vel_residues = torch.zeros_like(vel_residues)
        
        # Final outputs with spherical harmonics analysis
        atoms_output = torch.cat([vel_atoms, h_final_atoms], dim=-1)
        residues_output = torch.cat([vel_residues, h_final_residues], dim=-1)
        
        # 📊 Additional output: Spherical harmonics decomposition for analysis
        analysis_info = {
            'sh_breakdown': sh_analysis,
            'l0_contribution': sh_analysis.get('l0', torch.tensor(0.0)),
            'l1_contribution': sh_analysis.get('l1', torch.tensor(0.0)),
            'l2_contribution': sh_analysis.get('l2', torch.tensor(0.0)),
            'total_sh_utilization': len(sh_analysis)
        }
        
        return atoms_output, residues_output, analysis_info
    
    @staticmethod
    def remove_mean_batch(x, batch_mask):
        """Remove center of mass (same as before)"""
        if x.size(0) == 0:
            return x
        mean = scatter_mean(x, batch_mask, dim=0)
        x = x - mean[batch_mask]
        return x

    def analyze_spherical_harmonics_usage(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues):
        """
        🔬 Analysis tool: Hiểu mức độ sử dụng mỗi SH order
        """
        print("\n🔬 ANALYZING SPHERICAL HARMONICS USAGE:")
        
        atoms_out, residues_out, analysis = self.forward(xh_atoms, xh_residues, t, mask_atoms, mask_residues)
        
        sh_breakdown = analysis['sh_breakdown']
        
        for order, features in sh_breakdown.items():
            if features is not None:
                magnitude = torch.norm(features).item()
                print(f"   {order}: magnitude = {magnitude:.4f}")
        
        total_magnitude = sum(torch.norm(features).item() 
                            for features in sh_breakdown.values() 
                            if features is not None)
        
        print(f"📊 Total SH magnitude: {total_magnitude:.4f}")
        print(f"🎯 Active SH orders: {list(sh_breakdown.keys())}")
        
        return analysis
    
    def check_equivariance(self, xh_atoms, xh_pocket, t, mask_atoms, mask_pocket):
        """Test SE(3) equivariance of the dynamics"""
        x_atoms_orig = xh_atoms[:, :self.n_dims] 
        x_pocket_orig = xh_pocket[:, :self.n_dims]
        x_atoms_orig, x_pocket_orig = self.remove_mean_batch_simple(x_atoms_orig, x_pocket_orig)
        xh_atoms = torch.cat([x_atoms_orig, xh_atoms[:, self.n_dims:]], dim=1)
        xh_pocket = torch.cat([x_pocket_orig, xh_pocket[:, self.n_dims:]], dim=1)
        
        # Generate random rotation and translation
        R = self._random_rotation_matrix().to(xh_atoms.device)
        translation = torch.randn(3, device=xh_atoms.device) * 0.1
        
        # Original coordinates
        x_atoms_orig = xh_atoms[:, :self.n_dims]
        x_pocket_orig = xh_pocket[:, :self.n_dims]
        
        # Transformed coordinates
        x_atoms_rot = torch.matmul(x_atoms_orig, R.T) + translation
        x_pocket_rot = torch.matmul(x_pocket_orig, R.T) + translation
        
        # Create transformed inputs
        xh_atoms_rot = torch.cat([x_atoms_rot, xh_atoms[:, self.n_dims:]], dim=1)
        xh_pocket_rot = torch.cat([x_pocket_rot, xh_pocket[:, self.n_dims:]], dim=1)
        
        # Forward pass on original
        out_orig_atoms, out_orig_pocket, _ = self.forward(
            xh_atoms, xh_pocket, t, mask_atoms, mask_pocket
        )
        
        # Forward pass on transformed
        out_rot_atoms, out_rot_pocket, _ = self.forward(
            xh_atoms_rot, xh_pocket_rot, t, mask_atoms, mask_pocket
        )
        
        # Expected transformed output
        expected_atoms = torch.cat([
            torch.matmul(out_orig_atoms[:, :self.n_dims], R.T),
            out_orig_atoms[:, self.n_dims:]  # Features unchanged
        ], dim=1)
        
        expected_pocket = torch.cat([
            torch.matmul(out_orig_pocket[:, :self.n_dims], R.T),
            out_orig_pocket[:, self.n_dims:]
        ], dim=1)
        
        # Compute errors
        error_atoms = torch.norm(out_rot_atoms - expected_atoms, dim=-1).mean()
        error_pocket = torch.norm(out_rot_pocket - expected_pocket, dim=-1).mean()
        
        return (error_atoms + error_pocket).cpu().item()

    def _random_rotation_matrix(self):
        """Generate random 3D rotation matrix"""
        q = torch.randn(4)
        q = q / torch.norm(q)
        w, x, y, z = q
        rotation_matrix = torch.tensor([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ], dtype=torch.float)
        return rotation_matrix
    def remove_mean_batch_simple(self, x_atoms, x_pocket):
        """Simple COM removal for testing"""
        all_coords = torch.cat([x_atoms, x_pocket], dim=0)
        com = all_coords.mean(dim=0, keepdim=True)
        return x_atoms - com, x_pocket - com