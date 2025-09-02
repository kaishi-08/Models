import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_add, scatter_mean

from .vis_dynamics import ViSNetDynamics
from ..utils import molecular_utils

class ConditionalDDPMViSNet(nn.Module):
    def __init__(
            self,
            atom_nf: int,
            residue_nf: int,
            n_dims: int = 3,
            #visnet_parameter
            hidden_nf: int = 256,
            num_layers: int = 6,
            num_heads: int = 8,
            lmax: int =2, 
            vecnorm_type: str = 'max_min',
            trainable_vecnorm: bool = True,
            edge_cutoff_ligand: float = 5.0,
            edge_cutoff_pocket: float = 8.0,
            edge_cutoff_interaction: float = 5.0,
            activation: str = 'silu',
            cutoff: float = 5.0,
            update_pocket_coords: bool = False,
            size_histogram: dict = None,
            timesteps: int = 1000,
            parametrization: str = 'eps',
            noise_schedule: str = 'learned',
            noise_precision: float = 1e-4,
            loss_type: str = 'vlb',
            norm_values: tuple = (1., 1.),
            norm_biases: tuple = (None, 0.),
            vnode_idx: int = None
    ):
        super().__init__()
        
        assert loss_type in {'vlb', 'l2'}
        assert parametrization == 'eps'
        
        self.dynamics = ViSNetDynamics(
            atom_nf=atom_nf,
            residue_nf=residue_nf,
            n_dims=n_dims,
            hidden_nf=hidden_nf,
            num_layers=num_layers,
            num_heads=num_heads,
            lmax=lmax,  
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
            edge_cutoff_ligand=edge_cutoff_ligand,
            edge_cutoff_pocket=edge_cutoff_pocket,
            edge_cutoff_interaction=edge_cutoff_interaction,
            activation=activation,
            cutoff=cutoff,
            update_pocket_coords=update_pocket_coords
    )
                # Ensure dynamics doesn't update pocket coords (fixed condition)
        assert not self.dynamics.update_pocket_coords, \
            "Pocket coordinates should not be updated in conditional model"
        
        self.atom_nf = atom_nf
        self.residue_nf = residue_nf
        self.n_dims = n_dims
        self.T = timesteps
        self.parametrization = parametrization
        self.loss_type = loss_type
        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.vnode_idx = vnode_idx
        
        # Noise schedule
        if noise_schedule == 'learned':
            assert loss_type == 'vlb', 'Learned schedule requires VLB objective'
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(
                noise_schedule, timesteps=timesteps, precision=noise_precision)
        
        # Size distribution for ligands given pockets
        if size_histogram is not None:
            self.size_distribution = DistributionNodes(size_histogram)
        else:
            self.size_distribution = None
        
        self.register_buffer('buffer', torch.zeros(1))
        
        if noise_schedule != 'learned':
            self.check_issues_norm_values()

    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()
        
        norm_value = self.norm_values[1]
        if sigma_0 * num_stdevs > 1. / norm_value:
            raise ValueError(
                f'Normalization value {norm_value} too large with sigma_0 {sigma_0:.5f}')
    
    def sigma_and_alpha_t_given_s(self, gamma_t, gamma_s, target_tensor):
        """Compute sigma and alpha for transition from s to t."""
        sigma2_t_given_s = self.inflate_batch_array(
            -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t)), target_tensor)
        
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s
        
        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(alpha_t_given_s, target_tensor)
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
        
        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def sigma(self, gamma, target_tensor):
        """Compute sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Compute alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    @staticmethod
    def SNR(gamma):
        """Compute signal-to-noise ratio."""
        return torch.exp(-gamma)

    @staticmethod
    def inflate_batch_array(array, target):
        """Inflate batch array to match target shape."""
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def normalize(self, ligand=None, pocket=None):
        """Normalize ligand and pocket data."""
        if ligand is not None:
            ligand_norm = {
                'x': ligand['x'] / self.norm_values[0],
                'one_hot': (ligand['one_hot'].float() - self.norm_biases[1]) / self.norm_values[1],
                'mask': ligand['mask'],
                'size': ligand['size']
            }
        else:
            ligand_norm = None

        if pocket is not None:
            pocket_norm = {
                'x': pocket['x'] / self.norm_values[0],
                'one_hot': (pocket['one_hot'].float() - self.norm_biases[1]) / self.norm_values[1],
                'mask': pocket['mask'],
                'size': pocket['size']
            }
        else:
            pocket_norm = None

        return ligand_norm, pocket_norm

    def unnormalize(self, x, h_cat):
        """Unnormalize data."""
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        return x, h_cat

    def unnormalize_z(self, z_lig, z_pocket):
        """Unnormalize concatenated z tensor."""
        x_lig, h_lig = z_lig[:, :self.n_dims], z_lig[:, self.n_dims:]
        x_pocket, h_pocket = z_pocket[:, :self.n_dims], z_pocket[:, self.n_dims:]
        
        x_lig, h_lig = self.unnormalize(x_lig, h_lig)
        x_pocket, h_pocket = self.unnormalize(x_pocket, h_pocket)
        
        return torch.cat([x_lig, h_lig], dim=1), torch.cat([x_pocket, h_pocket], dim=1)

    @classmethod
    
    def remove_mean_batch(cls, x_lig, x_pocket, lig_indices, pocket_indices):
        # Compute ligand center of mass
        mean = scatter_mean(x_lig, lig_indices, dim=0)
        
        # Subtract ligand COM from both ligand and pocket
        # This creates a ligand-centered coordinate system
        x_lig = x_lig - mean[lig_indices]
        x_pocket = x_pocket - mean[pocket_indices]
        
        return x_lig, x_pocket

    def test_equivariance(self, ligand, pocket):
        """
        Test method to verify the model maintains SE(3) equivariance
        """
        print("Testing SE(3) equivariance...")
        
        # Prepare normalized inputs
        ligand_norm, pocket_norm = self.normalize(ligand, pocket)
        
        # Sample a random timestep
        t = torch.rand(1, 1, device=ligand['x'].device)
        
        # Prepare concatenated inputs for dynamics
        xh_lig = torch.cat([ligand_norm['x'], ligand_norm['one_hot']], dim=1)
        xh_pocket = torch.cat([pocket_norm['x'], pocket_norm['one_hot']], dim=1)
        
        # Test equivariance of the dynamics
        error = self.dynamics.check_equivariance(
            xh_lig, xh_pocket, t, ligand['mask'], pocket['mask']
        )
        
        print(f"Total equivariance error: {error:.8f}")
        
        if error < 1e-5:
            print("✅ Model is properly equivariant!")
        else:
            print("❌ Model has equivariance issues!")
        
        return error

    def forward(self, ligand, pocket, return_info=False):
        # Test equivariance periodically during training
        if self.training and torch.rand(1).item() < 0.001:  # 0.1% chance
            with torch.no_grad():
                self.test_equivariance(ligand, pocket)
        
        # Normalize data
        ligand, pocket = self.normalize(ligand, pocket)
        
        # Volume change (ligand only)
        delta_log_px = self.delta_log_px(ligand['size'])
        
        # Sample timestep
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, 
                             size=(ligand['size'].size(0), 1),
                             device=ligand['x'].device).float()
        s_int = t_int - 1
        
        t_is_zero = (t_int == 0).float()
        t_is_not_zero = 1 - t_is_zero
        
        s = s_int / self.T
        t = t_int / self.T
        
        # Noise schedule
        gamma_s = self.inflate_batch_array(self.gamma(s), ligand['x'])
        gamma_t = self.inflate_batch_array(self.gamma(t), ligand['x'])
        
        # Prepare input
        xh0_lig = torch.cat([ligand['x'], ligand['one_hot']], dim=1)
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)
        
        # Center system (ligand COM reference) - PROPER equivariant centering
        xh0_lig[:, :self.n_dims], xh0_pocket[:, :self.n_dims] = \
            self.remove_mean_batch(xh0_lig[:, :self.n_dims],
                                   xh0_pocket[:, :self.n_dims],
                                   ligand['mask'], pocket['mask'])
        
        # Add noise to ligand only
        z_t_lig, xh_pocket, eps_t_lig = \
            self.noised_representation(xh0_lig, xh0_pocket, 
                                       ligand['mask'], pocket['mask'], gamma_t)
        
        # Neural network prediction using IMPROVED ViSNet
        net_out_lig, _ , _= self.dynamics(z_t_lig, xh_pocket, t, ligand['mask'], pocket['mask'])
        
        # Compute L2 error
        squared_error = (eps_t_lig - net_out_lig) ** 2
        if self.vnode_idx is not None:
            squared_error[ligand['one_hot'][:, self.vnode_idx].bool(), :self.n_dims] = 0
        error_t_lig = self.sum_except_batch(squared_error, ligand['mask'])
        
        # SNR weighting
        SNR_weight = (1 - self.SNR(gamma_s - gamma_t)).squeeze(1)
        
        # Constants and KL prior
        neg_log_constants = -self.log_constants_p_x_given_z0(ligand['size'], error_t_lig.device)
        kl_prior = self.kl_prior(xh0_lig, ligand['mask'], ligand['size'])
        
        # L0 term computation
        if self.training:
            log_p_x_given_z0_without_constants_ligand, log_ph_given_z0 = \
                self.log_pxh_given_z0_without_constants(ligand, z_t_lig, eps_t_lig, net_out_lig, gamma_t)
            loss_0_x_ligand = -log_p_x_given_z0_without_constants_ligand * t_is_zero.squeeze()
            loss_0_h = -log_ph_given_z0 * t_is_zero.squeeze()
            error_t_lig = error_t_lig * t_is_not_zero.squeeze()
        else:
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), ligand['x'])
            z_0_lig, xh_pocket, eps_0_lig = \
                self.noised_representation(xh0_lig, xh0_pocket, ligand['mask'], pocket['mask'], gamma_0)
            net_out_0_lig, _, _ = self.dynamics(z_0_lig, xh_pocket, t_zeros, ligand['mask'], pocket['mask'])
            log_p_x_given_z0_without_constants_ligand, log_ph_given_z0 = \
                self.log_pxh_given_z0_without_constants(ligand, z_0_lig, eps_0_lig, net_out_0_lig, gamma_0)
            loss_0_x_ligand = -log_p_x_given_z0_without_constants_ligand
            loss_0_h = -log_ph_given_z0
        
        # Size prior
        log_pN = self.log_pN(ligand['size'], pocket['size'])
        
        # For potential additional losses
        xh_lig_hat = self.xh_given_zt_and_epsilon(z_t_lig, net_out_lig, gamma_t, ligand['mask'])
        
        info = {
            'eps_hat_lig_x': scatter_mean(net_out_lig[:, :self.n_dims].abs().mean(1), ligand['mask'], dim=0).mean(),
            'eps_hat_lig_h': scatter_mean(net_out_lig[:, self.n_dims:].abs().mean(1), ligand['mask'], dim=0).mean(),
        }
        
        loss_terms = (delta_log_px, error_t_lig, torch.tensor(0.0), SNR_weight,
                      loss_0_x_ligand, torch.tensor(0.0), loss_0_h,
                      neg_log_constants, kl_prior, log_pN,
                      t_int.squeeze(), xh_lig_hat)
        
        return (*loss_terms, info) if return_info else loss_terms
    
    def noised_representation(self, xh_lig, xh0_pocket, lig_mask, pocket_mask, gamma_t):
        """Create noised representation for ligand only (pocket stays clean)."""
        # Compute alpha_t and sigma_t from gamma
        alpha_t = self.alpha(gamma_t, xh_lig)
        sigma_t = self.sigma(gamma_t, xh_lig)

        # Sample noise for ligand only
        eps_lig = self.sample_gaussian(
            size=(len(lig_mask), self.n_dims + self.atom_nf),
            device=lig_mask.device)

        # Apply noise to ligand
        z_t_lig = alpha_t[lig_mask] * xh_lig + sigma_t[lig_mask] * eps_lig

        # Pocket stays clean, but apply COM centering
        xh_pocket = xh0_pocket.detach().clone()
        z_t_lig[:, :self.n_dims], xh_pocket[:, :self.n_dims] = \
            self.remove_mean_batch(z_t_lig[:, :self.n_dims],
                                   xh_pocket[:, :self.n_dims],
                                   lig_mask, pocket_mask)

        return z_t_lig, xh_pocket, eps_lig

    def sample_normal_zero_com(self, mu_lig, xh0_pocket, sigma, lig_mask, pocket_mask, fix_noise=False):
        """Sample from normal distribution with COM constraint."""
        if fix_noise:
            raise NotImplementedError("fix_noise option not implemented")

        eps_lig = self.sample_gaussian(
            size=(len(lig_mask), self.n_dims + self.atom_nf),
            device=lig_mask.device)

        out_lig = mu_lig + sigma[lig_mask] * eps_lig

        # Apply COM constraint
        xh_pocket = xh0_pocket.detach().clone()
        out_lig[:, :self.n_dims], xh_pocket[:, :self.n_dims] = \
            self.remove_mean_batch(out_lig[:, :self.n_dims],
                                   xh0_pocket[:, :self.n_dims],
                                   lig_mask, pocket_mask)

        return out_lig, xh_pocket

    # ==================== LOSS COMPUTATION ====================

    def kl_prior(self, xh_lig, mask_lig, num_nodes):
        """KL divergence between q(z1|x_ligand) and p(z1) = N(0,1)."""
        batch_size = len(num_nodes)
        
        ones = torch.ones((batch_size, 1), device=xh_lig.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh_lig)
        
        mu_T_lig = alpha_T[mask_lig] * xh_lig
        mu_T_lig_x, mu_T_lig_h = mu_T_lig[:, :self.n_dims], mu_T_lig[:, self.n_dims:]
        
        sigma_T_x = self.sigma(gamma_T, mu_T_lig_x).squeeze()
        sigma_T_h = self.sigma(gamma_T, mu_T_lig_h).squeeze()
        
        # KL for features
        zeros = torch.zeros_like(mu_T_lig_h)
        ones = torch.ones_like(sigma_T_h)
        mu_norm2 = self.sum_except_batch((mu_T_lig_h - zeros) ** 2, mask_lig)
        kl_distance_h = self.gaussian_KL(mu_norm2, sigma_T_h, ones, d=1)
        
        # KL for coordinates
        zeros = torch.zeros_like(mu_T_lig_x)
        ones = torch.ones_like(sigma_T_x)
        mu_norm2 = self.sum_except_batch((mu_T_lig_x - zeros) ** 2, mask_lig)
        subspace_d = self.subspace_dimensionality(num_nodes)
        kl_distance_x = self.gaussian_KL(mu_norm2, sigma_T_x, ones, subspace_d)
        
        return kl_distance_x + kl_distance_h

    def log_pxh_given_z0_without_constants(self, ligand, z_0_lig, eps_lig, net_out_lig, gamma_0, epsilon=1e-10):
        """Compute log probability for ligand given z0."""
        z_h_lig = z_0_lig[:, self.n_dims:]
        eps_lig_x = eps_lig[:, :self.n_dims]
        net_lig_x = net_out_lig[:, :self.n_dims]
        
        sigma_0 = self.sigma(gamma_0, target_tensor=z_0_lig)
        sigma_0_cat = sigma_0 * self.norm_values[1]
        
        # Coordinate error
        squared_error = (eps_lig_x - net_lig_x) ** 2
        if self.vnode_idx is not None:
            squared_error[ligand['one_hot'][:, self.vnode_idx].bool(), :self.n_dims] = 0
            
        log_p_x_given_z0_without_constants_ligand = -0.5 * (
            self.sum_except_batch(squared_error, ligand['mask']))
        
        # Feature probability
        ligand_onehot = ligand['one_hot'] * self.norm_values[1] + self.norm_biases[1]
        estimated_ligand_onehot = z_h_lig * self.norm_values[1] + self.norm_biases[1]
        centered_ligand_onehot = estimated_ligand_onehot - 1
        
        log_ph_cat_proportional_ligand = torch.log(
            self.cdf_standard_gaussian((centered_ligand_onehot + 0.5) / sigma_0_cat[ligand['mask']])
            - self.cdf_standard_gaussian((centered_ligand_onehot - 0.5) / sigma_0_cat[ligand['mask']])
            + epsilon)
        
        log_Z = torch.logsumexp(log_ph_cat_proportional_ligand, dim=1, keepdim=True)
        log_probabilities_ligand = log_ph_cat_proportional_ligand - log_Z
        log_ph_given_z0_ligand = self.sum_except_batch(
            log_probabilities_ligand * ligand_onehot, ligand['mask'])
        
        return log_p_x_given_z0_without_constants_ligand, log_ph_given_z0_ligand

    def log_constants_p_x_given_z0(self, n_nodes, device):
        """Constants for p(x|z0)."""
        batch_size = len(n_nodes)
        degrees_of_freedom_x = self.subspace_dimensionality(n_nodes)
        
        zeros = torch.zeros((batch_size, 1), device=device)
        gamma_0 = self.gamma(zeros)
        log_sigma_x = 0.5 * gamma_0.view(batch_size)
        
        return degrees_of_freedom_x * (-log_sigma_x - 0.5 * np.log(2 * np.pi))

    def log_pN(self, N_lig, N_pocket):
        """Prior on sample size."""
        if self.size_distribution is not None:
            return self.size_distribution.log_prob_n1_given_n2(N_lig, N_pocket)
        else:
            return torch.zeros_like(N_lig, dtype=torch.float)

    def delta_log_px(self, num_nodes):
        return -self.subspace_dimensionality(num_nodes) * np.log(self.norm_values[0])

    # ==================== SAMPLING ====================

    def compute_x_pred(self, net_out, zt, gamma_t, batch_mask):
        """Compute x prediction from noise prediction."""
        sigma_t = self.sigma(gamma_t, target_tensor=net_out)
        alpha_t = self.alpha(gamma_t, target_tensor=net_out)
        eps_t = net_out
        x_pred = 1. / alpha_t[batch_mask] * (zt - sigma_t[batch_mask] * eps_t)
        return x_pred

    def sample_p_xh_given_z0(self, z0_lig, xh0_pocket, lig_mask, pocket_mask, batch_size, fix_noise=False):
        """Sample ligand from p(x,h|z0) conditioned on pocket."""
        t_zeros = torch.zeros(size=(batch_size, 1), device=z0_lig.device)
        gamma_0 = self.gamma(t_zeros)
        sigma_x = self.SNR(-0.5 * gamma_0)
        
        net_out_lig, _, _= self.dynamics(z0_lig, xh0_pocket, t_zeros, lig_mask, pocket_mask)
        mu_x_lig = self.compute_x_pred(net_out_lig, z0_lig, gamma_0, lig_mask)
        
        xh_lig, xh0_pocket = self.sample_normal_zero_com(
            mu_x_lig, xh0_pocket, sigma_x, lig_mask, pocket_mask, fix_noise)
        
        x_lig, h_lig = self.unnormalize(xh_lig[:, :self.n_dims], z0_lig[:, self.n_dims:])
        x_pocket, h_pocket = self.unnormalize(xh0_pocket[:, :self.n_dims], xh0_pocket[:, self.n_dims:])
        
        h_lig = F.one_hot(torch.argmax(h_lig, dim=1), self.atom_nf)
        
        return x_lig, h_lig, x_pocket, h_pocket

    def sample_p_zs_given_zt(self, s, t, zt_lig, xh0_pocket, ligand_mask, pocket_mask, fix_noise=False):
        """Sample zs ~ p(zs | zt) during reverse diffusion."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)
        
        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt_lig)
        
        sigma_s = self.sigma(gamma_s, target_tensor=zt_lig)
        sigma_t = self.sigma(gamma_t, target_tensor=zt_lig)
        
        eps_t_lig, _, _= self.dynamics(zt_lig, xh0_pocket, t, ligand_mask, pocket_mask)
        
        mu_lig = zt_lig / alpha_t_given_s[ligand_mask] - \
                 (sigma2_t_given_s / alpha_t_given_s / sigma_t)[ligand_mask] * eps_t_lig
        
        sigma = sigma_t_given_s * sigma_s / sigma_t
        
        zs_lig, xh0_pocket = self.sample_normal_zero_com(
            mu_lig, xh0_pocket, sigma, ligand_mask, pocket_mask, fix_noise)
        
        return zs_lig, xh0_pocket

    @torch.no_grad()
    def sample_given_pocket(self, pocket, num_nodes_lig, return_frames=1, timesteps=None):
        """Generate ligands conditioned on given pocket."""
        timesteps = self.T if timesteps is None else timesteps
        assert 0 < return_frames <= timesteps
        assert timesteps % return_frames == 0
        
        n_samples = len(pocket['size'])
        device = pocket['x'].device
        
        _, pocket = self.normalize(pocket=pocket)
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)
        
        lig_mask = molecular_utils.num_nodes_to_batch_mask(n_samples, num_nodes_lig, device)
        
        # Initialize ligand at pocket center
        mu_lig_x = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        mu_lig_h = torch.zeros((n_samples, self.atom_nf), device=device)
        mu_lig = torch.cat((mu_lig_x, mu_lig_h), dim=1)[lig_mask]
        sigma = torch.ones_like(pocket['size']).unsqueeze(1)
        
        z_lig, xh_pocket = self.sample_normal_zero_com(
            mu_lig, xh0_pocket, sigma, lig_mask, pocket['mask'])
        
        out_lig = torch.zeros((return_frames,) + z_lig.size(), device=z_lig.device)
        out_pocket = torch.zeros((return_frames,) + xh_pocket.size(), device=device)
        
        # Reverse diffusion
        for s in reversed(range(0, timesteps)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z_lig.device)
            t_array = s_array + 1
            s_array = s_array / timesteps
            t_array = t_array / timesteps
            
            z_lig, xh_pocket = self.sample_p_zs_given_zt(
                s_array, t_array, z_lig, xh_pocket, lig_mask, pocket['mask'])
            
            if (s * return_frames) % timesteps == 0:
                idx = (s * return_frames) // timesteps
                out_lig[idx], out_pocket[idx] = self.unnormalize_z(z_lig, xh_pocket)
        
        # Final sampling
        x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(
            z_lig, xh_pocket, lig_mask, pocket['mask'], n_samples)
        
        # CoM drift correction
        if return_frames == 1:
            max_cog = scatter_add(x_lig, lig_mask, dim=0).abs().max().item()
            if max_cog > 5e-2:
                print(f'Warning: CoM drift {max_cog:.3f}, correcting...')
                x_lig, x_pocket = self.remove_mean_batch(x_lig, x_pocket, lig_mask, pocket['mask'])
        
        out_lig[0] = torch.cat([x_lig, h_lig], dim=1)
        out_pocket[0] = torch.cat([x_pocket, h_pocket], dim=1)
        
        return out_lig.squeeze(0), out_pocket.squeeze(0), lig_mask, pocket['mask']

    # ==================== UTILITY FUNCTIONS ====================

    def xh_given_zt_and_epsilon(self, z_t, epsilon, gamma_t, batch_mask):
        """Equation (7) in EDM paper."""
        alpha_t = self.alpha(gamma_t, z_t)
        sigma_t = self.sigma(gamma_t, z_t)
        xh = z_t / alpha_t[batch_mask] - epsilon * sigma_t[batch_mask] / alpha_t[batch_mask]
        return xh

    def subspace_dimensionality(self, input_size):
        """Dimensionality on translation-invariant subspace."""
        return (input_size - 1) * self.n_dims

    @staticmethod
    def gaussian_KL(q_mu_minus_p_mu_squared, q_sigma, p_sigma, d):
        """KL divergence between two Gaussians."""
        return d * torch.log(p_sigma / q_sigma) + \
               0.5 * (d * q_sigma ** 2 + q_mu_minus_p_mu_squared) / (p_sigma ** 2) - 0.5 * d

    @staticmethod
    def sum_except_batch(x, indices):
        return scatter_add(x.sum(-1), indices, dim=0)

    @staticmethod
    def cdf_standard_gaussian(x):
        return 0.5 * (1. + torch.erf(x / math.sqrt(2)))

    @staticmethod
    def sample_gaussian(size, device):
        return torch.randn(size, device=device)


# ==================== SUPPORTING CLASSES ====================

class DistributionNodes:
    """Distribution for number of ligand nodes given pocket."""
    def __init__(self, histogram):
        histogram = torch.tensor(histogram).float() + 1e-3
        prob = histogram / histogram.sum()
        
        self.idx_to_n_nodes = torch.tensor(
            [[(i, j) for j in range(prob.shape[1])] for i in range(prob.shape[0])]
        ).view(-1, 2)
        
        self.n_nodes_to_idx = {tuple(x.tolist()): i for i, x in enumerate(self.idx_to_n_nodes)}
        self.prob = prob
        self.m = torch.distributions.Categorical(prob.view(-1))
        
        self.n1_given_n2 = [
            torch.distributions.Categorical(prob[:, j]) for j in range(prob.shape[1])
        ]

    def log_prob_n1_given_n2(self, n1, n2):
        log_probs = torch.stack([self.n1_given_n2[c].log_prob(i.cpu()) 
                                for i, c in zip(n1, n2)])
        return log_probs.to(n1.device)


class PositiveLinear(nn.Module):
    """Linear layer with positive weights."""
    def __init__(self, in_features, out_features, bias=True, weight_init_offset=-2):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, F.softplus(self.weight), self.bias)


class GammaNetwork(nn.Module):
    """Learnable monotonic noise schedule."""
    def __init__(self):
        super().__init__()
        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)
        self.gamma_0 = nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = nn.Parameter(torch.tensor([10.]))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)
        
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (gamma_tilde_1 - gamma_tilde_0)
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma
        return gamma


class PredefinedNoiseSchedule(nn.Module):
    """Predefined noise schedules."""
    def __init__(self, noise_schedule, timesteps, precision):
        super().__init__()
        self.timesteps = timesteps
        
        if noise_schedule == 'cosine':
            alphas2 = self._cosine_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            power = float(noise_schedule.split('_')[1])
            alphas2 = self._polynomial_schedule(timesteps, precision, power)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule}")
        
        sigmas2 = 1 - alphas2
        log_alphas2_to_sigmas2 = np.log(alphas2) - np.log(sigmas2)
        
        self.gamma = nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def _cosine_schedule(self, timesteps, s=0.008):
        steps = timesteps + 2
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, a_min=0, a_max=0.999)
        return np.cumprod(1. - betas, axis=0)

    def _polynomial_schedule(self, timesteps, s=1e-4, power=3.):
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas2 = (1 - np.power(x / steps, power))**2
        precision = 1 - 2 * s
        return precision * alphas2 + s

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]