import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional
from ..models.sde_diffusion import VESDE, EulerMaruyamaSDESolver

class SDESampler:
    """SDE sampling for molecular generation"""
    
    def __init__(self, model, sde: VESDE, device: str = 'cuda'):
        self.model = model
        self.sde = sde
        self.device = device
        self.solver = EulerMaruyamaSDESolver(sde, self._score_fn)
    
    def _score_fn(self, x, pos, edge_index, t, condition=None, batch=None):
        """Score function wrapper"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                x=x, pos=pos, edge_index=edge_index,
                edge_attr=torch.zeros((edge_index.size(1), 1)),
                batch=batch, **condition if condition else {}
            )
            return outputs['pos_pred']
    
    def sample(self, shape, condition: Optional[Dict] = None, num_steps: Optional[int] = None):
        """Sample from the SDE"""
        num_steps = num_steps or self.sde.N
        
        # Initialize from prior
        x_init = self.sde.prior_sampling(shape).to(self.device)
        
        # Time steps
        time_steps = torch.linspace(1., 0., num_steps + 1).to(self.device)
        
        x = x_init
        for i in range(num_steps):
            t = time_steps[i].expand(x.size(0))
            x = self.solver.step(x, None, None, t, condition)
        
        return x