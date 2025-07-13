# src/models/sde_diffusion.py
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Tuple
from .e3_egnn import E3EquivariantGNN


class SDE(ABC):
    """Abstract base class for stochastic differential equations"""
    
    def __init__(self, N: int):
        self.N = N  # Number of discretization steps
        
    @abstractmethod
    def sde(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns drift and diffusion coefficients"""
        pass
    
    @abstractmethod
    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns mean and std of marginal probability"""
        pass
    
    @abstractmethod
    def prior_sampling(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Sample from prior distribution"""
        pass

class VESDE(SDE):
    """Variance Exploding SDE for molecular generation"""
    
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0, N: int = 1000):
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(
            np.log(self.sigma_min), np.log(self.sigma_max), N))
        
    def sde(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * np.log(self.sigma_max / self.sigma_min)))
        return drift, diffusion
    
    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std
    
    def prior_sampling(self, shape: Tuple[int, ...]) -> torch.Tensor:
        return torch.randn(*shape) * self.sigma_max

class ScoreNet(nn.Module):
    """Score network for SDE diffusion"""
    
    def __init__(self, e3_gnn: E3EquivariantGNN, marginal_prob_std):
        super().__init__()
        self.e3_gnn = e3_gnn
        self.marginal_prob_std = marginal_prob_std
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # Conditional features (protein pocket)
        self.condition_embed = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor,
                t: torch.Tensor, condition: torch.Tensor = None, batch=None):
        # Time embedding
        time_emb = self.positional_encoding(t)
        time_emb = self.time_embed(time_emb)
        
        # Conditional embedding
        if condition is not None:
            cond_emb = self.condition_embed(condition)
            # Broadcast to match batch size
            if batch is not None:
                cond_emb = cond_emb[batch]
            time_emb = time_emb + cond_emb
        
        # Add time embedding to node features
        x = x + time_emb.unsqueeze(-1)
        
        # Apply E(3) equivariant network
        score = self.e3_gnn(x, pos, edge_index, batch)
        
        # Scale by marginal probability std
        std = self.marginal_prob_std(t)
        score = score / std[:, None, None]
        
        return score
    
    def positional_encoding(self, t: torch.Tensor, dim: int = 128):
        """Sinusoidal positional encoding for time"""
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(t.device)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

class EulerMaruyamaSDESolver:
    """Euler-Maruyama solver for SDE sampling"""
    
    def __init__(self, sde: SDE, score_fn, probability_flow: bool = False):
        self.sde = sde
        self.score_fn = score_fn
        self.probability_flow = probability_flow
        
    def step(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor,
             t: torch.Tensor, condition: torch.Tensor = None, batch=None):
        """Single step of Euler-Maruyama solver"""
        dt = -1.0 / self.sde.N
        
        # Get SDE coefficients
        drift, diffusion = self.sde.sde(x, t)
        
        # Get score
        score = self.score_fn(x, pos, edge_index, t, condition, batch)
        
        # Compute drift including score
        drift = drift - (diffusion ** 2)[:, None, None] * score
        
        # Euler-Maruyama step
        x_mean = x + drift * dt
        
        if not self.probability_flow:
            noise = torch.randn_like(x)
            x = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * noise
        else:
            x = x_mean
            
        return x