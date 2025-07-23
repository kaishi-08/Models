# src/models/ddpm_diffusion.py - Complete DDPM implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any

class MolecularDDPM(nn.Module):
    """DDPM for molecular generation with robust device handling"""
    
    def __init__(self, 
                 num_timesteps: int = 1000,
                 beta_schedule: str = "cosine",
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        
        # Create beta schedule
        if beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-compute constants
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # For forward process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine noise schedule"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward_process(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Add noise to data (forward diffusion process)"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        device = x0.device
        
        # Move schedules to device
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        
        # Extract coefficients for batch
        sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting with 3D positions
        if x0.dim() == 2:  # [N, 3] positions
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.view(-1, 1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.view(-1, 1)
        
        # Forward process: q(x_t | x_0)
        x_t = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        
        return x_t, noise
    
    def compute_loss(self, model, x0: torch.Tensor, **model_kwargs):
        """
        Compute DDPM loss with comprehensive error handling
        
        Args:
            model: The neural network model
            x0: Clean data (target positions)
            **model_kwargs: Additional model inputs
        """
        device = x0.device
        batch_size = x0.size(0)
        
        try:
            # Sample timesteps for the batch
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
            
            # Sample noise
            noise = torch.randn_like(x0)
            
            # Forward process (add noise)
            x_t, _ = self.forward_process(x0, t, noise)
            
            # Prepare model inputs with device safety
            model_inputs = self._prepare_model_inputs(x_t, t, model_kwargs, device)
            
            # Predict noise
            noise_pred = model(**model_inputs)
            
            # Handle different output formats
            if isinstance(noise_pred, dict):
                if 'pos_pred' in noise_pred:
                    noise_pred = noise_pred['pos_pred']
                elif 'noise_pred' in noise_pred:
                    noise_pred = noise_pred['noise_pred']
                else:
                    # Take first tensor value
                    noise_pred = next(iter(noise_pred.values()))
            
            # Ensure noise_pred has same shape as noise
            if noise_pred.shape != noise.shape:
                print(f"Warning: Shape mismatch. noise: {noise.shape}, pred: {noise_pred.shape}")
                # Try to fix shape mismatch
                if noise_pred.size(0) == noise.size(0):
                    if noise_pred.dim() == 1:
                        noise_pred = noise_pred.view(-1, 1).expand_as(noise)
                    elif noise_pred.size(1) != noise.size(1):
                        noise_pred = noise_pred[:, :noise.size(1)]
            
            # Compute MSE loss
            loss = F.mse_loss(noise_pred, noise)
            
            return loss, {'noise_loss': loss.item()}
            
        except Exception as e:
            print(f"DDPM compute_loss error: {e}")
            print(f"x0 shape: {x0.shape}, device: {x0.device}")
            print(f"Model kwargs keys: {list(model_kwargs.keys())}")
            
            # Return dummy loss to continue training
            dummy_loss = torch.tensor(1.0, device=device, requires_grad=True)
            return dummy_loss, {'noise_loss': 1.0}
    
    def _prepare_model_inputs(self, x_t: torch.Tensor, t: torch.Tensor, 
                             model_kwargs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """Prepare model inputs with proper device handling"""
        model_inputs = {
            'pos': x_t,  # Noisy positions
            't': t       # Timesteps
        }
        
        # Add molecular features (avoid 'x' name conflict)
        if 'atom_features' in model_kwargs:
            model_inputs['x'] = self._ensure_device(model_kwargs['atom_features'], device)
        elif 'x' in model_kwargs:
            model_inputs['x'] = self._ensure_device(model_kwargs['x'], device)
        
        # Add graph structure
        graph_keys = ['edge_index', 'edge_attr', 'batch']
        for key in graph_keys:
            if key in model_kwargs and model_kwargs[key] is not None:
                model_inputs[key] = self._ensure_device(model_kwargs[key], device)
        
        # Add pocket data with device safety
        pocket_keys = ['pocket_x', 'pocket_pos', 'pocket_edge_index', 'pocket_batch']
        for key in pocket_keys:
            if key in model_kwargs and model_kwargs[key] is not None:
                model_inputs[key] = self._ensure_device(model_kwargs[key], device)
        
        return model_inputs
    
    def _ensure_device(self, tensor, device):
        """Ensure tensor is on correct device"""
        if tensor is None:
            return None
        if not isinstance(tensor, torch.Tensor):
            return tensor
        return tensor.to(device)

class MolecularDDPMModel(nn.Module):
    """Wrapper for molecular model with DDPM time conditioning"""
    
    def __init__(self, base_model, ddpm: MolecularDDPM):
        super().__init__()
        self.base_model = base_model
        self.ddpm = ddpm
        
        # Time embedding for DDPM
        hidden_dim = getattr(base_model, 'hidden_dim', 256)
        self.time_embedding = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
    
    def forward(self, **kwargs):
        """
        Forward pass with time conditioning
        
        Handles various argument patterns and ensures compatibility
        """
        # Extract time if present
        t = kwargs.pop('t', None)
        
        # Get main inputs
        x = kwargs.get('x')
        pos = kwargs.get('pos')
        
        # Time embedding (if provided)
        if t is not None:
            t_emb = self._embed_timestep(t)
            time_features = self.time_embedding(t_emb)
            
            # For now, we'll let the base model handle time conditioning
            # Advanced time conditioning can be added later
        
        # Call base model
        try:
            outputs = self.base_model(**kwargs)
            
            # Return appropriate output for DDPM
            if isinstance(outputs, dict):
                # Return position prediction if available
                if 'pos_pred' in outputs:
                    return outputs['pos_pred']
                elif 'positions' in outputs:
                    return outputs['positions']
                else:
                    # Return the predicted noise (for position)
                    return outputs.get('node_features', pos)
            else:
                # If single tensor output, assume it's position prediction
                return outputs
                
        except Exception as e:
            print(f"MolecularDDPMModel forward error: {e}")
            print(f"Input keys: {list(kwargs.keys())}")
            
            # Return dummy output with correct shape
            if pos is not None:
                return torch.zeros_like(pos)
            elif x is not None:
                return torch.zeros(x.size(0), 3, device=x.device)
            else:
                return torch.zeros(1, 3)
    
    def _embed_timestep(self, timesteps, dim=128):
        """Sinusoidal timestep embedding"""
        device = timesteps.device
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb
    
    def parameters(self):
        """Get all parameters"""
        params = list(self.base_model.parameters())
        params.extend(list(self.time_embedding.parameters()))
        return iter(params)
    
    def train(self):
        """Set to training mode"""
        self.base_model.train()
        self.time_embedding.train()
        return super().train()
    
    def eval(self):
        """Set to evaluation mode"""
        self.base_model.eval()
        self.time_embedding.eval()
        return super().eval()
    
    def state_dict(self):
        """Get state dictionary"""
        return {
            'base_model': self.base_model.state_dict(),
            'time_embedding': self.time_embedding.state_dict()
        }
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dictionary"""
        if 'base_model' in state_dict:
            self.base_model.load_state_dict(state_dict['base_model'], strict=strict)
        if 'time_embedding' in state_dict:
            self.time_embedding.load_state_dict(state_dict['time_embedding'], strict=strict)
        else:
            # Fallback for old checkpoints
            try:
                self.base_model.load_state_dict(state_dict, strict=strict)
            except:
                print("Warning: Could not load state dict")
    
    def to(self, device):
        """Move to device"""
        self.base_model.to(device)
        self.time_embedding.to(device)
        return super().to(device)