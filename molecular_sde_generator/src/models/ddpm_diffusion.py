# src/models/ddpm_diffusion.py - Replace SDE with DDPM
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class MolecularDDPM(nn.Module):
    """DDPM for molecular generation - More stable than SDE"""
    
    def __init__(self, 
                 num_timesteps: int = 1000,
                 beta_schedule: str = "cosine",
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        
        # Create beta schedule (much simpler than SDE)
        if beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-compute constants (key advantage of DDPM)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # For sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        # For reverse process
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule - works better for molecular data"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward_process(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Add noise to data (much simpler than SDE)"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        # Extract coefficients for batch
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.view(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.view(-1, 1)
        
        # Forward process: q(x_t | x_0)
        x_t = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        
        return x_t, noise
    
    def reverse_process(self, model, x_t: torch.Tensor, t: torch.Tensor, **model_kwargs):
        """Single reverse step (much cleaner than SDE solver)"""
        model.eval()
        with torch.no_grad():
            # Predict noise
            noise_pred = model(x_t, t, **model_kwargs)
            
            # Compute x_{t-1}
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            # Reshape for broadcasting
            alpha_t = alpha_t.view(-1, 1)
            alpha_cumprod_t = alpha_cumprod_t.view(-1, 1)
            beta_t = beta_t.view(-1, 1)
            
            # Mean of reverse process
            x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
            x_prev_mean = (x_t - beta_t / torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_t)
            
            # Add noise (except for t=0)
            if t[0] > 0:
                posterior_variance_t = self.posterior_variance[t].view(-1, 1)
                noise = torch.randn_like(x_t)
                x_prev = x_prev_mean + torch.sqrt(posterior_variance_t) * noise
            else:
                x_prev = x_prev_mean
            
            return x_prev, x_0_pred
    
    def sample(self, model, shape: Tuple[int, ...], device: str = 'cuda', **model_kwargs):
        """Full sampling process (much simpler than SDE)"""
        # Start from pure noise
        x_t = torch.randn(shape, device=device)
        
        # Reverse process
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x_t, _ = self.reverse_process(model, x_t, t_batch, **model_kwargs)
        
        return x_t
    
    def compute_loss(self, model, x0: torch.Tensor, **model_kwargs):
        """Simple loss computation (key advantage)"""
        batch_size = x0.shape[0]
        device = x0.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(x0)
        
        # Forward process
        x_t, _ = self.forward_process(x0, t, noise)
        
        # Predict noise
        noise_pred = model(x_t, t, **model_kwargs)
        
        # Simple MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss, {'noise_loss': loss.item()}

class MolecularDDPMModel(nn.Module):
    """Wrapper for molecular model with DDPM"""
    
    def __init__(self, base_model, ddpm: MolecularDDPM):
        super().__init__()
        self.base_model = base_model
        self.ddpm = ddpm
        
        # Time embedding for DDPM
        self.time_embedding = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, base_model.hidden_dim)
        )
    
    def forward(self, x, t, pos=None, edge_index=None, edge_attr=None, batch=None, **kwargs):
        """Forward with time conditioning"""
        
        # Embed timestep
        t_emb = self._embed_timestep(t)
        time_features = self.time_embedding(t_emb)
        
        # Add time conditioning to node features
        if x.dim() == 2:
            # Broadcast time features to match batch
            if batch is not None:
                time_features_broadcast = time_features[batch]  # [N, hidden_dim]
            else:
                time_features_broadcast = time_features[0:1].expand(x.size(0), -1)
            
            # Simple addition conditioning
            if x.size(1) == time_features_broadcast.size(1):
                x_conditioned = x + time_features_broadcast
            else:
                # Project x to match dimensions
                x_proj = F.linear(x, torch.eye(time_features_broadcast.size(1), x.size(1), device=x.device))
                x_conditioned = x_proj + time_features_broadcast
        else:
            x_conditioned = x
        
        # Use base model
        outputs = self.base_model(
            x=x_conditioned, pos=pos, edge_index=edge_index, 
            edge_attr=edge_attr, batch=batch, **kwargs
        )
        
        # Return position prediction (main DDPM target)
        return outputs['pos_pred']
    
    def _embed_timestep(self, timesteps, dim=128):
        """Sinusoidal timestep embedding"""
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb