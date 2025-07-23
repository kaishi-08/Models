# src/models/multimodal_sde.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Any
from joint_2d_3d_model import Joint2D3DMolecularModel

class MultiModalSDE:
    """
    Separate diffusion processes for different feature types
    - Continuous diffusion for 3D positions
    - Discrete diffusion for atom/bond types
    """
    
    def __init__(self, 
                 # Position diffusion (VESDE)
                 pos_sigma_min: float = 0.01,
                 pos_sigma_max: float = 50.0,
                 
                 # Discrete diffusion parameters
                 atom_types: int = 100,
                 bond_types: int = 5,
                 discrete_noise_schedule: str = "cosine",
                 
                 num_steps: int = 1000):
        
        self.pos_sigma_min = pos_sigma_min
        self.pos_sigma_max = pos_sigma_max
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.num_steps = num_steps
        
        # Discrete noise schedule (cho atom/bond types)
        if discrete_noise_schedule == "cosine":
            self.discrete_betas = self._cosine_beta_schedule(num_steps)
        else:
            self.discrete_betas = torch.linspace(0.0001, 0.02, num_steps)
        
        self.discrete_alphas = 1 - self.discrete_betas
        self.discrete_alphas_cumprod = torch.cumprod(self.discrete_alphas, dim=0)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine noise schedule cho discrete diffusion"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def position_diffusion(self, pos: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """VESDE diffusion cho 3D positions"""
        sigma = self.pos_sigma_min * (self.pos_sigma_max / self.pos_sigma_min) ** t
        drift = torch.zeros_like(pos)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * np.log(self.pos_sigma_max / self.pos_sigma_min)))
        return drift, diffusion
    
    def position_marginal_prob(self, pos: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Marginal probability cho positions"""
        std = self.pos_sigma_min * (self.pos_sigma_max / self.pos_sigma_min) ** t
        mean = pos
        return mean, std
    
    def discrete_forward_process(self, x_discrete: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward process cho discrete features (atom/bond types)
        Áp dụng categorical diffusion
        """
        batch_size = x_discrete.size(0)
        
        # Convert discrete indices to one-hot
        if x_discrete.dim() == 1:
            x_onehot = F.one_hot(x_discrete, num_classes=self.atom_types).float()
        else:
            x_onehot = x_discrete
        
        # Sample noise timestep
        t_discrete = t.long().clamp(0, self.num_steps - 1)
        
        # Get noise level
        alpha_cumprod = self.discrete_alphas_cumprod[t_discrete]
        
        # Apply categorical noise
        # q(x_t | x_0) = alpha_t * x_0 + (1 - alpha_t) * uniform
        uniform_noise = torch.ones_like(x_onehot) / x_onehot.size(-1)
        
        # Reshape alpha for broadcasting
        alpha_cumprod = alpha_cumprod.view(-1, 1)
        
        # Noisy distribution
        x_noisy = alpha_cumprod * x_onehot + (1 - alpha_cumprod) * uniform_noise
        
        return x_noisy
    
    def sample_combined(self, shape_pos: tuple, shape_atoms: tuple, shape_bonds: tuple, 
                       device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """Sample từ combined prior"""
        
        # Prior cho positions (Gaussian)
        pos_prior = torch.randn(shape_pos, device=device) * self.pos_sigma_max
        
        # Prior cho atom types (uniform categorical)
        atom_prior = torch.ones(shape_atoms + (self.atom_types,), device=device) / self.atom_types
        
        # Prior cho bond types (uniform categorical)  
        bond_prior = torch.ones(shape_bonds + (self.bond_types,), device=device) / self.bond_types
        
        return {
            'positions': pos_prior,
            'atom_types': atom_prior,
            'bond_types': bond_prior
        }

class MultiModalMolecularModel(nn.Module):
    """Model handle multiple diffusion processes"""
    
    def __init__(self, base_model, multimodal_sde: MultiModalSDE):
        super().__init__()
        self.base_model = base_model
        self.sde = multimodal_sde
        
        # Additional heads cho discrete diffusion
        self.atom_denoiser = nn.Sequential(
            nn.Linear(base_model.hidden_dim, base_model.hidden_dim),
            nn.ReLU(),
            nn.Linear(base_model.hidden_dim, multimodal_sde.atom_types)
        )
        
        self.bond_denoiser = nn.Sequential(
            nn.Linear(base_model.hidden_dim * 2, base_model.hidden_dim),
            nn.ReLU(), 
            nn.Linear(base_model.hidden_dim, multimodal_sde.bond_types)
        )
    
    def forward(self, x, pos, edge_index, edge_attr, batch, t_pos, t_discrete, **kwargs):
        """
        Forward pass với multiple timesteps
        
        Args:
            t_pos: timesteps cho position diffusion
            t_discrete: timesteps cho discrete diffusion
        """
        
        # Base model forward
        outputs = self.base_model(x, pos, edge_index, edge_attr, batch, **kwargs)
        
        # Position denoising (existing)
        pos_pred = outputs['pos_pred']
        
        # Discrete denoising
        # Embed discrete timesteps
        t_discrete_emb = self._embed_timestep(t_discrete, self.base_model.hidden_dim)
        
        # Add timestep to node features
        node_features_with_time = outputs['node_features'] + t_discrete_emb[batch]
        
        # Atom type denoising
        atom_logits = self.atom_denoiser(node_features_with_time)
        
        # Bond type denoising
        if edge_index.size(1) > 0:
            row, col = edge_index
            edge_features_with_time = torch.cat([
                node_features_with_time[row], 
                node_features_with_time[col]
            ], dim=-1)
            bond_logits = self.bond_denoiser(edge_features_with_time)
        else:
            bond_logits = torch.zeros((0, self.sde.bond_types))
        
        return {
            'pos_pred': pos_pred,
            'atom_logits': atom_logits,  # For discrete diffusion
            'bond_logits': bond_logits,  # For discrete diffusion
            'node_features': outputs['node_features']
        }
    
    def _embed_timestep(self, t, dim):
        """Sinusoidal timestep embedding"""
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(t.device)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

class MultiModalLoss(nn.Module):
    """Loss function cho multi-modal diffusion"""
    
    def __init__(self, pos_weight=1.0, atom_weight=0.2, bond_weight=0.2):
        super().__init__()
        self.pos_weight = pos_weight
        self.atom_weight = atom_weight
        self.bond_weight = bond_weight
    
    def forward(self, predictions, targets, noise_pos, noise_discrete):
        """
        Args:
            predictions: model outputs
            targets: ground truth
            noise_pos: position noise
            noise_discrete: discrete noise info
        """
        losses = {}
        
        # Position loss (score matching)
        if 'pos_pred' in predictions:
            pos_target = -noise_pos / noise_discrete['pos_std'][:, None]
            pos_loss = F.mse_loss(predictions['pos_pred'], pos_target)
            losses['pos_loss'] = pos_loss
        
        # Discrete diffusion losses
        if 'atom_logits' in predictions and 'atom_noisy' in targets:
            # Predict original distribution from noisy
            atom_loss = F.cross_entropy(
                predictions['atom_logits'].view(-1, predictions['atom_logits'].size(-1)),
                targets['atom_original'].view(-1)
            )
            losses['atom_loss'] = atom_loss
        
        if 'bond_logits' in predictions and 'bond_noisy' in targets:
            bond_loss = F.cross_entropy(
                predictions['bond_logits'].view(-1, predictions['bond_logits'].size(-1)),
                targets['bond_original'].view(-1)
            )
            losses['bond_loss'] = bond_loss
        
        # Total loss
        total_loss = (self.pos_weight * losses.get('pos_loss', 0) + 
                     self.atom_weight * losses.get('atom_loss', 0) + 
                     self.bond_weight * losses.get('bond_loss', 0))
        
        losses['total_loss'] = total_loss
        return losses

# Usage example
def create_multimodal_trainer():
    """Create trainer với multi-modal diffusion"""
    
    # Create multi-modal SDE
    multimodal_sde = MultiModalSDE(
        pos_sigma_min=0.01,
        pos_sigma_max=50.0,
        atom_types=100,
        bond_types=5,
        num_steps=1000
    )
    
    # Wrap existing model
    base_model = Joint2D3DMolecularModel(...)  # Your existing model
    model = MultiModalMolecularModel(base_model, multimodal_sde)
    
    # Use specialized loss
    loss_fn = MultiModalLoss(pos_weight=1.0, atom_weight=0.2, bond_weight=0.2)
    
    return model, multimodal_sde, loss_fn