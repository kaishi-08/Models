import math
from typing import Dict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean

import utils


class Diffusion(nn.Module):
    def __init__(
            self,
            dynamics: nn.Module, 
            atom_nf: int, 
            residue_nf: int,
            n_dims: int, 
            size_histogram: Dict,
            timesteps: int = 1000, 
            parametrization='eps',
            noise_schedule='learned', 
            noise_precision=1e-4,
            loss_type='vlb', 
            norm_values=(1., 1.), 
            norm_biases=(None, 0.)):
        super().__init__()

        assert loss_type in {'vlb', 'l2'}
        self.loss_type = loss_type
        if noise_schedule == 'learned':
            assert loss_type == 'vlb', 'A noise schedule can only be learned with a vlb objective.'

        # Only supported parametrization.
        assert parametrization == 'eps'

        if noise_schedule == 'learned':
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule,
                                                 timesteps=timesteps,
                                                 precision=noise_precision)

        self.dynamics = dynamics #VisNet

        self.atom_nf = atom_nf
        self.residue_nf = residue_nf
        self.n_dims = n_dims
        self.num_classes = self.atom_nf

        self.T = timesteps
        self.parametrization = parametrization

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer('buffer', torch.zeros(1))

        self.size_distribution = DistributionNodes(size_histogram)

        if noise_schedule != 'learned':
            self.check_issues_norm_values()

    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        norm_value = self.norm_values[1]
        if sigma_0 * num_stdevs > 1. / norm_value:
            raise ValueError(
                f'Value for normalization value {norm_value} probably too '
                f'large with sigma_0 {sigma_0:.5f} and '
                f'1 / norm_value = {1. / norm_value}')

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor,
                                  gamma_s: torch.Tensor,
                                  target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t)), target_tensor
        )

        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def kl_prior_ligand(self, xh_lig, mask_lig, num_nodes):
        """
        Computes the KL between q(z1 | x_ligand) and the prior p(z1) = Normal(0, 1).
        Chỉ tính cho ligand vì chỉ ligand được noise.
        """
        batch_size = len(num_nodes)

        # Compute the last alpha value, alpha_T.
        ones = torch.ones((batch_size, 1), device=xh_lig.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh_lig)

        # Compute means for ligand
        mu_T_lig = alpha_T[mask_lig] * xh_lig
        mu_T_lig_x, mu_T_lig_h = mu_T_lig[:, :self.n_dims], mu_T_lig[:, self.n_dims:]

        # Compute standard deviations
        sigma_T_x = self.sigma(gamma_T, mu_T_lig_x).squeeze()
        sigma_T_h = self.sigma(gamma_T, mu_T_lig_h).squeeze()

        # Compute KL for h-part (ligand features)
        zeros_lig = torch.zeros_like(mu_T_lig_h)
        ones = torch.ones_like(sigma_T_h)
        mu_norm2_h = self.sum_except_batch((mu_T_lig_h - zeros_lig) ** 2, mask_lig)
        kl_distance_h = self.gaussian_KL(mu_norm2_h, sigma_T_h, ones, d=1)

        # Compute KL for x-part (ligand positions)
        zeros_lig_x = torch.zeros_like(mu_T_lig_x)
        ones_x = torch.ones_like(sigma_T_x)
        mu_norm2_x = self.sum_except_batch((mu_T_lig_x - zeros_lig_x) ** 2, mask_lig)
        subspace_d = self.subspace_dimensionality(num_nodes)
        kl_distance_x = self.gaussian_KL(mu_norm2_x, sigma_T_x, ones_x, subspace_d)

        return kl_distance_x + kl_distance_h

    def compute_x_pred(self, net_out, zt, gamma_t, batch_mask):
        """Computes x_pred, i.e. the most likely prediction of x."""
        if self.parametrization == 'eps':
            sigma_t = self.sigma(gamma_t, target_tensor=net_out)
            alpha_t = self.alpha(gamma_t, target_tensor=net_out)
            eps_t = net_out
            x_pred = 1. / alpha_t[batch_mask] * (zt - sigma_t[batch_mask] * eps_t)
        else:
            raise ValueError(self.parametrization)
        return x_pred

    def log_constants_p_x_given_z0(self, n_nodes, device):
        """Computes p(x|z0) constants."""
        batch_size = len(n_nodes)
        degrees_of_freedom_x = self.subspace_dimensionality(n_nodes)

        zeros = torch.zeros((batch_size, 1), device=device)
        gamma_0 = self.gamma(zeros)

        log_sigma_x = 0.5 * gamma_0.view(batch_size)
        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))

    def log_pxh_given_z0_without_constants(self, ligand, z_0_lig, eps_lig, 
                                           net_out_lig, gamma_0, epsilon=1e-10):
        """
        Tính log probability cho ligand given z0, protein không được noise.
        """
        # Discrete properties are predicted directly from z_t.
        z_h_lig = z_0_lig[:, self.n_dims:]

        # Take only part over x.
        eps_lig_x = eps_lig[:, :self.n_dims]
        net_lig_x = net_out_lig[:, :self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data.
        sigma_0 = self.sigma(gamma_0, target_tensor=z_0_lig)
        sigma_0_cat = sigma_0 * self.norm_values[1]

        # Computes the error for the distribution N(x | predicted_x, sigma_0)
        log_p_x_given_z0_without_constants_ligand = -0.5 * (
            self.sum_except_batch((eps_lig_x - net_lig_x) ** 2, ligand['mask'])
        )

        # Compute delta indicator masks cho categorical features
        ligand_onehot = ligand['one_hot'] * self.norm_values[1] + self.norm_biases[1]
        estimated_ligand_onehot = z_h_lig * self.norm_values[1] + self.norm_biases[1]

        # Centered h_cat around 1, since onehot encoded.
        centered_ligand_onehot = estimated_ligand_onehot - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        log_ph_cat_proportional_ligand = torch.log(
            self.cdf_standard_gaussian((centered_ligand_onehot + 0.5) / sigma_0_cat[ligand['mask']])
            - self.cdf_standard_gaussian((centered_ligand_onehot - 0.5) / sigma_0_cat[ligand['mask']])
            + epsilon
        )

        # Normalize the distribution over the categories.
        log_Z = torch.logsumexp(log_ph_cat_proportional_ligand, dim=1, keepdim=True)
        log_probabilities_ligand = log_ph_cat_proportional_ligand - log_Z

        # Select the log_prob of the current category using the onehot representation.
        log_ph_given_z0_ligand = self.sum_except_batch(
            log_probabilities_ligand * ligand_onehot, ligand['mask'])

        return log_p_x_given_z0_without_constants_ligand, log_ph_given_z0_ligand

    def sample_p_xh_given_z0(self, z0_lig, lig_mask, batch_size):
        """Samples ligand x ~ p(x|z0) với protein làm điều kiện."""
        t_zeros = torch.zeros(size=(batch_size, 1), device=z0_lig.device)
        gamma_0 = self.gamma(t_zeros)
        sigma_x = self.SNR(-0.5 * gamma_0)
        
        # Neural network prediction - chỉ predict cho ligand
        net_out_lig = self.dynamics(z0_lig, t_zeros, lig_mask)

        # Compute mu for p(x | z0)
        mu_x_lig = self.compute_x_pred(net_out_lig, z0_lig, gamma_0, lig_mask)
        
        # Sample from normal distribution
        xh_lig = self.sample_normal_ligand(mu_x_lig, sigma_x, lig_mask)

        x_lig, h_lig = self.unnormalize(
            xh_lig[:, :self.n_dims], xh_lig[:, self.n_dims:])

        h_lig = F.one_hot(torch.argmax(h_lig, dim=1), self.atom_nf)

        return x_lig, h_lig

    def sample_normal_ligand(self, mu_lig, sigma, lig_mask):
        """Samples from a Normal distribution cho ligand."""
        eps_lig = self.sample_ligand_noise(lig_mask)
        return mu_lig + sigma[lig_mask] * eps_lig

    def noised_representation_ligand(self, xh_lig, lig_mask, gamma_t):
        """
        Tạo noised representation chỉ cho ligand.
        """
        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, xh_lig)
        sigma_t = self.sigma(gamma_t, xh_lig)

        # Sample noise chỉ cho ligand
        eps_lig = self.sample_ligand_noise(lig_mask)

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t_lig = alpha_t[lig_mask] * xh_lig + sigma_t[lig_mask] * eps_lig

        return z_t_lig, eps_lig

    def log_pN(self, N_lig):
        """
        Prior on the ligand sample size.
        """
        log_pN = self.size_distribution.log_prob_ligand(N_lig)
        return log_pN

    def delta_log_px(self, num_nodes):
        """Volume change due to normalization"""
        return -self.subspace_dimensionality(num_nodes) * np.log(self.norm_values[0])

    def forward(self, ligand, pocket, return_info=False):
        """
        Computes the loss - chỉ cho ligand, protein làm điều kiện.
        """
        # Normalize ligand data
        ligand_norm, pocket_norm = self.normalize(ligand, pocket)

        # Likelihood change due to normalization (chỉ cho ligand)
        delta_log_px = self.delta_log_px(ligand['size'])

        # Sample timestep t cho mỗi example trong batch
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(ligand['size'].size(0), 1),
            device=ligand['x'].device).float()
        s_int = t_int - 1

        # Masks
        t_is_zero = (t_int == 0).float()
        t_is_not_zero = 1 - t_is_zero

        # Normalize t to [0, 1]
        s = s_int / self.T
        t = t_int / self.T

        # Compute gamma_s and gamma_t
        gamma_s = self.inflate_batch_array(self.gamma(s), ligand['x'])
        gamma_t = self.inflate_batch_array(self.gamma(t), ligand['x'])

        # Concatenate x và h cho ligand
        xh_lig = torch.cat([ligand_norm['x'], ligand_norm['one_hot']], dim=1)

        # Find noised representation - chỉ cho ligand
        z_t_lig, eps_t_lig = self.noised_representation_ligand(
            xh_lig, ligand['mask'], gamma_t)

        # Neural net prediction - pass cả ligand (noised) và protein (clean) làm điều kiện
        net_out_lig = self.dynamics(
            z_t_lig, pocket_norm, t, ligand['mask'], pocket['mask'])

        # Compute L2 error cho ligand
        error_t_lig = self.sum_except_batch((eps_t_lig - net_out_lig) ** 2,
                                            ligand['mask'])

        # Compute weighting with SNR
        SNR_weight = (1 - self.SNR(gamma_s - gamma_t)).squeeze(1)

        # Constants depending on sigma_0
        neg_log_constants = -self.log_constants_p_x_given_z0(
            n_nodes=ligand['size'], device=error_t_lig.device)

        # KL between q(zT | x_ligand) and p(zT) = Normal(0, 1)
        kl_prior = self.kl_prior_ligand(xh_lig, ligand['mask'], ligand['size'])

        if self.training:
            # Computes the L_0 term
            log_p_x_given_z0_without_constants_ligand, log_ph_given_z0 = \
                self.log_pxh_given_z0_without_constants(
                    ligand_norm, z_t_lig, eps_t_lig, net_out_lig, gamma_t)

            loss_0_x_ligand = -log_p_x_given_z0_without_constants_ligand * t_is_zero.squeeze()
            loss_0_h = -log_ph_given_z0 * t_is_zero.squeeze()

            # apply t_is_zero mask
            error_t_lig = error_t_lig * t_is_not_zero.squeeze()

        else:
            # Compute noise values for t = 0
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), ligand['x'])

            z_0_lig, eps_0_lig = self.noised_representation_ligand(
                xh_lig, ligand['mask'], gamma_0)

            net_out_0_lig = self.dynamics(
                z_0_lig, pocket_norm, t_zeros, ligand['mask'], pocket['mask'])

            log_p_x_given_z0_without_constants_ligand, log_ph_given_z0 = \
                self.log_pxh_given_z0_without_constants(
                    ligand_norm, z_0_lig, eps_0_lig, net_out_0_lig, gamma_0)
            
            loss_0_x_ligand = -log_p_x_given_z0_without_constants_ligand
            loss_0_h = -log_ph_given_z0

        # sample size prior cho ligand
        log_pN = self.log_pN(ligand['size'])

        info = {
            'eps_hat_lig_x': scatter_mean(
                net_out_lig[:, :self.n_dims].abs().mean(1), ligand['mask'], dim=0).mean(),
            'eps_hat_lig_h': scatter_mean(
                net_out_lig[:, self.n_dims:].abs().mean(1), ligand['mask'], dim=0).mean(),
        }
        
        loss_terms = (delta_log_px, error_t_lig, SNR_weight,
                      loss_0_x_ligand, loss_0_h, neg_log_constants, 
                      kl_prior, log_pN, t_int.squeeze())
        
        return (*loss_terms, info) if return_info else loss_terms

    def sample_p_zs_given_zt(self, s, t, zt_lig, ligand_mask, pocket, pocket_mask):
        """
        Samples from zs ~ p(zs | zt) cho ligand. 
        Protein được pass như điều kiện không đổi.
        """
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt_lig)

        sigma_s = self.sigma(gamma_s, target_tensor=zt_lig)
        sigma_t = self.sigma(gamma_t, target_tensor=zt_lig)

        # Neural net prediction với protein làm điều kiện
        eps_t_lig = self.dynamics(zt_lig, pocket, t, ligand_mask, pocket_mask)

        # Assert mean zero cho ligand positions
        self.assert_mean_zero_with_mask(zt_lig[:, :self.n_dims], ligand_mask)
        self.assert_mean_zero_with_mask(eps_t_lig[:, :self.n_dims], ligand_mask)

        # Compute mu for p(zs | zt)
        mu_lig = zt_lig / alpha_t_given_s[ligand_mask] - \
                 (sigma2_t_given_s / alpha_t_given_s / sigma_t)[ligand_mask] * eps_t_lig

        # Compute sigma for p(zs | zt)
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs
        zs_lig = self.sample_normal_ligand(mu_lig, sigma, ligand_mask)

        # Project down to avoid numerical runaway of center of gravity
        zs_lig_x = self.remove_mean_batch(zs_lig[:, :self.n_dims], ligand_mask)
        zs_lig = torch.cat((zs_lig_x, zs_lig[:, self.n_dims:]), dim=1)

        return zs_lig

    def sample_ligand_noise(self, lig_indices):
        """
        Samples mean-centered normal noise cho ligand positions và standard normal cho features.
        """
        z_x = self.sample_center_gravity_zero_gaussian(
            size=(len(lig_indices), self.n_dims), indices=lig_indices)
        z_h_lig = self.sample_gaussian(
            size=(len(lig_indices), self.atom_nf), device=lig_indices.device)
        z_lig = torch.cat([z_x, z_h_lig], dim=1)
        return z_lig

    @torch.no_grad()
    def sample(self, n_samples, num_nodes_lig, pocket, pocket_mask,
               return_frames=1, timesteps=None, device='cpu'):
        """
        Draw samples from the generative model cho ligand với protein làm điều kiện.
        """
        timesteps = self.T if timesteps is None else timesteps
        assert 0 < return_frames <= timesteps
        assert timesteps % return_frames == 0

        lig_mask = utils.num_nodes_to_batch_mask(n_samples, num_nodes_lig, device)

        # Sample initial noise cho ligand
        z_lig = self.sample_ligand_noise(lig_mask)

        self.assert_mean_zero_with_mask(z_lig[:, :self.n_dims], lig_mask)

        out_lig = torch.zeros((return_frames,) + z_lig.size(), device=z_lig.device)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1
        for s in reversed(range(0, timesteps)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z_lig.device)
            t_array = s_array + 1
            s_array = s_array / timesteps
            t_array = t_array / timesteps

            z_lig = self.sample_p_zs_given_zt(
                s_array, t_array, z_lig, lig_mask, pocket, pocket_mask)

            # save frame
            if (s * return_frames) % timesteps == 0:
                idx = (s * return_frames) // timesteps
                out_lig[idx] = self.unnormalize_z_ligand(z_lig)

        # Finally sample p(x, h | z_0) cho ligand
        x_lig, h_lig = self.sample_p_xh_given_z0(z_lig, lig_mask, n_samples)

        self.assert_mean_zero_with_mask(x_lig, lig_mask)

        # Correct CoM drift
        if return_frames == 1:
            max_cog = scatter_add(x_lig, lig_mask, dim=0).abs().max().item()
            if max_cog > 5e-2:
                print(f'Warning CoG drift with error {max_cog:.3f}. Projecting positions down.')
                x_lig = self.remove_mean_batch(x_lig, lig_mask)

        # Overwrite last frame
        out_lig[0] = torch.cat([x_lig, h_lig], dim=1)

        return out_lig.squeeze(0), lig_mask

    @torch.no_grad()
    def inpaint_ligand(self, ligand, pocket, lig_fixed, resamplings=1,
                       jump_length=1, return_frames=1, timesteps=None):
        """
        Inpainting cho ligand với protein làm điều kiện cố định.
        """
        timesteps = self.T if timesteps is None else timesteps
        assert 0 < return_frames <= timesteps
        assert timesteps % return_frames == 0

        if len(lig_fixed.size()) == 1:
            lig_fixed = lig_fixed.unsqueeze(1)

        ligand_norm, pocket_norm = self.normalize(ligand, pocket)

        n_samples = len(ligand['size'])
        xh0_lig = torch.cat([ligand_norm['x'], ligand_norm['one_hot']], dim=1)

        # Sample initial noise cho ligand
        z_lig = self.sample_ligand_noise(ligand['mask'])

        out_lig = torch.zeros((return_frames,) + z_lig.size(), device=z_lig.device)

        # Iteratively sample theo pre-defined schedule
        schedule = self.get_repaint_schedule(resamplings, jump_length, timesteps)
        s = timesteps - 1
        
        for i, n_denoise_steps in enumerate(schedule):
            for j in range(n_denoise_steps):
                s_array = torch.full((n_samples, 1), fill_value=s, device=z_lig.device)
                t_array = s_array + 1
                s_array = s_array / timesteps
                t_array = t_array / timesteps

                # sample known nodes from input
                gamma_s = self.inflate_batch_array(self.gamma(s_array), ligand['x'])
                z_lig_known, _ = self.noised_representation_ligand(
                    xh0_lig, ligand['mask'], gamma_s)

                # sample unknown part
                z_lig_unknown = self.sample_p_zs_given_zt(
                    s_array, t_array, z_lig, ligand['mask'], pocket_norm, pocket['mask'])

                # combine known and unknown parts
                z_lig = z_lig_known * lig_fixed + z_lig_unknown * (1 - lig_fixed)

                self.assert_mean_zero_with_mask(z_lig[:, :self.n_dims], ligand['mask'])

                # save frame
                if n_denoise_steps > jump_length or i == len(schedule) - 1:
                    if (s * return_frames) % timesteps == 0:
                        idx = (s * return_frames) // timesteps
                        out_lig[idx] = self.unnormalize_z_ligand(z_lig)

                # Jump back nếu cần
                if j == n_denoise_steps - 1 and i < len(schedule) - 1:
                    t = s + jump_length
                    t_array = torch.full((n_samples, 1), fill_value=t, device=z_lig.device)
                    t_array = t_array / timesteps

                    gamma_s = self.inflate_batch_array(self.gamma(s_array), ligand['x'])
                    gamma_t = self.inflate_batch_array(self.gamma(t_array), ligand['x'])

                    z_lig = self.sample_p_zt_given_zs_ligand(
                        z_lig, ligand['mask'], gamma_t, gamma_s)
                    s = t

                s -= 1

        # Finally sample p(x, h | z_0)
        x_lig, h_lig = self.sample_p_xh_given_z0(z_lig, ligand['mask'], n_samples)

        self.assert_mean_zero_with_mask(x_lig, ligand['mask'])

        # Correct CoM drift
        if return_frames == 1:
            max_cog = scatter_add(x_lig, ligand['mask'], dim=0).abs().max().item()
            if max_cog > 5e-2:
                print(f'Warning CoG drift with error {max_cog:.3f}. Projecting positions down.')
                x_lig = self.remove_mean_batch(x_lig, ligand['mask'])

        out_lig[0] = torch.cat([x_lig, h_lig], dim=1)

        return out_lig.squeeze(0), ligand['mask']

    def sample_p_zt_given_zs_ligand(self, zs_lig, ligand_mask, gamma_t, gamma_s):
        """Sample từ forward process cho ligand."""
        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zs_lig)

        mu_lig = alpha_t_given_s[ligand_mask] * zs_lig
        zt_lig = self.sample_normal_ligand(mu_lig, sigma_t_given_s, ligand_mask)

        # Remove center of mass
        zt_x = self.remove_mean_batch(zt_lig[:, :self.n_dims], ligand_mask)
        zt_lig = torch.cat((zt_x, zt_lig[:, self.n_dims:]), dim=1)

        return zt_lig

    def get_repaint_schedule(self, resamplings, jump_length, timesteps):
        """Tạo repaint schedule."""
        repaint_schedule = []
        curr_t = 0
        while curr_t < timesteps:
            if curr_t + jump_length < timesteps:
                if len(repaint_schedule) > 0:
                    repaint_schedule[-1] += jump_length
                    repaint_schedule.extend([jump_length] * (resamplings - 1))
                else:
                    repaint_schedule.extend([jump_length] * resamplings)
                curr_t += jump_length
            else:
                residual = (timesteps - curr_t)
                if len(repaint_schedule) > 0:
                    repaint_schedule[-1] += residual
                else:
                    repaint_schedule.append(residual)
                curr_t += residual

        return list(reversed(repaint_schedule))

    # Utility methods
    @staticmethod
    def gaussian_KL(q_mu_minus_p_mu_squared, q_sigma, p_sigma, d):
        """Computes KL distance between two normal distributions."""
        return d * torch.log(p_sigma / q_sigma) + \
               0.5 * (d * q_sigma ** 2 + q_mu_minus_p_mu_squared) / (p_sigma ** 2) - 0.5 * d

    @staticmethod
    def inflate_batch_array(array, target):
        """Inflates batch array to match target shape."""
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    @staticmethod
    def SNR(gamma):
        """Computes signal to noise ratio given gamma."""
        return torch.exp(-gamma)

    def normalize(self, ligand=None, pocket=None):
        """Normalize ligand và pocket data."""
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

    def unnormalize_z_ligand(self, z_lig):
        """Unnormalize ligand z."""
        x_lig, h_lig = z_lig[:, :self.n_dims], z_lig[:, self.n_dims:]
        x_lig, h_lig = self.unnormalize(x_lig, h_lig)
        return torch.cat([x_lig, h_lig], dim=1)

    def subspace_dimensionality(self, input_size):
        """Compute dimensionality on translation-invariant linear subspace."""
        return (input_size - 1) * self.n_dims

    @staticmethod
    def remove_mean_batch(x, indices):
        mean = scatter_mean(x, indices, dim=0)
        x = x - mean[indices]
        return x

    @staticmethod
    def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
        largest_value = x.abs().max().item()
        error = scatter_add(x, node_mask, dim=0).abs().max().item()
        rel_error = error / (largest_value + eps)
        assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'

    @staticmethod
    def sample_center_gravity_zero_gaussian(size, indices):
        assert len(size) == 2
        x = torch.randn(size, device=indices.device)
        x_projected = Diffusion.remove_mean_batch(x, indices)
        return x_projected

    @staticmethod
    def sum_except_batch(x, indices):
        return scatter_add(x.sum(-1), indices, dim=0)

    @staticmethod
    def cdf_standard_gaussian(x):
        return 0.5 * (1. + torch.erf(x / math.sqrt(2)))

    @staticmethod
    def sample_gaussian(size, device):
        return torch.randn(size, device=device)


class DistributionNodes:
    """Distribution cho số lượng nodes của ligand."""
    def __init__(self, histogram):
        histogram = torch.tensor(histogram).float()
        histogram = histogram + 1e-3  # numerical stability

        prob = histogram / histogram.sum()
        
        self.idx_to_n_nodes = torch.arange(len(histogram))
        self.n_nodes_to_idx = {i: i for i in range(len(histogram))}

        self.prob = prob
        self.m = torch.distributions.Categorical(self.prob, validate_args=True)

        entropy = self.m.entropy()
        print("Entropy of ligand n_nodes: H[N]", entropy.item())

    def sample(self, n_samples=1):
        """Sample số lượng nodes cho ligand."""
        idx = self.m.sample((n_samples,))
        return self.idx_to_n_nodes[idx]

    def log_prob_ligand(self, batch_n_nodes):
        """Log probability của số lượng nodes ligand."""
        assert len(batch_n_nodes.size()) == 1
        log_probs = self.m.log_prob(batch_n_nodes)
        return log_probs.to(batch_n_nodes.device)


class PositiveLinear(torch.nn.Module):
    """Linear layer với weights luôn dương."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = F.softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


class GammaNetwork(torch.nn.Module):
    """Gamma network models monotonic increasing function."""
    def __init__(self):
        super().__init__()
        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)
        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma
        return gamma


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """Cosine schedule như trong DiffSBDD."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def clip_noise_schedule(alphas2, clip_value=0.001):
    """Clip noise schedule for stability."""
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
    alphas_step = (alphas2[1:] / alphas2[:-1])
    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)
    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """Polynomial noise schedule."""
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2
    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)
    precision = 1 - 2 * s
    alphas2 = precision * alphas2 + s
    return alphas2


class PredefinedNoiseSchedule(torch.nn.Module):
    """Predefined noise schedule cho non-learned schedules."""
    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        sigmas2 = 1 - alphas2
        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)
        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]