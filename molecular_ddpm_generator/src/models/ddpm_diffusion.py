# Enhanced DDPM that also noises graph structure
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

class MolecularDDPM(nn.Module):    
    def __init__(self, 
                 num_timesteps: int = 1000,
                 beta_schedule: str = "cosine",
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 noise_graph_structure: bool = True,
                 atom_noise_scale: float = 0.1,
                 bond_noise_scale: float = 0.2):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.noise_graph_structure = noise_graph_structure
        self.atom_noise_scale = atom_noise_scale
        self.bond_noise_scale = bond_noise_scale
        
        # Original position diffusion parameters
        if beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        if noise_graph_structure:
            self.graph_betas = self._cosine_beta_schedule(num_timesteps, s=0.01)  # Slower for discrete
            self.graph_alphas = 1 - self.graph_betas
            self.graph_alphas_cumprod = torch.cumprod(self.graph_alphas, dim=0)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule for diffusion"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward_process_positions(self, pos: torch.Tensor, t: torch.Tensor, 
                                noise: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion for positions (continuous)"""
        if noise is None:
            noise = torch.randn_like(pos)
        
        device = pos.device
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        
        sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
        
        # Handle single vs batch timesteps
        if t.numel() == 1:
            alpha_coeff = sqrt_alpha_cumprod_t.item()
            noise_coeff = sqrt_one_minus_alpha_cumprod_t.item()
        else:
            alpha_coeff = sqrt_alpha_cumprod_t.view(-1, 1)
            noise_coeff = sqrt_one_minus_alpha_cumprod_t.view(-1, 1)
        
        pos_t = alpha_coeff * pos + noise_coeff * noise
        return pos_t, noise
    
    def forward_process_atoms(self, atom_features: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward diffusion for atom features (add noise to continuous features)"""
        if not self.noise_graph_structure or atom_features is None:
            return atom_features
        
        device = atom_features.device
        graph_alphas_cumprod = self.graph_alphas_cumprod.to(device)
        
        # Get noise level for this timestep
        alpha_t = graph_alphas_cumprod[t]
        if t.numel() == 1:
            alpha_coeff = alpha_t.item()
        else:
            alpha_coeff = alpha_t.view(-1, 1)
        
        # Add noise to atom features (continuous features like formal charge, etc.)
        noise = torch.randn_like(atom_features) * self.atom_noise_scale
        noisy_atoms = alpha_coeff * atom_features + (1 - alpha_coeff) * noise
        
        # For discrete features (atom type), use different approach
        if atom_features.size(1) > 0:  # First feature is typically atom type
            atom_types = atom_features[:, 0].long()
            
            # Discrete diffusion: randomly flip atom types based on noise level
            flip_prob = (1 - alpha_coeff) * 0.1  # Small probability to change atom type
            random_mask = torch.rand(atom_types.size(0), device=device) < flip_prob
            
            if random_mask.any():
                # Change to random atom type (C, N, O, S mainly)
                new_atom_types = torch.randint(0, 4, (random_mask.sum(),), device=device)
                noisy_atoms[random_mask, 0] = new_atom_types.float()
        
        return noisy_atoms
    
    def forward_process_bonds(self, edge_attr: torch.Tensor, edge_index: torch.Tensor, 
                            t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion for bond features"""
        if not self.noise_graph_structure or edge_attr is None or edge_index.size(1) == 0:
            return edge_attr, edge_index
        
        device = edge_attr.device
        graph_alphas_cumprod = self.graph_alphas_cumprod.to(device)
        
        alpha_t = graph_alphas_cumprod[t]
        if t.numel() == 1:
            alpha_coeff = alpha_t.item()
        else:
            alpha_coeff = alpha_t.item()  # Simplified for now
        
        # Add noise to bond features
        noise = torch.randn_like(edge_attr) * self.bond_noise_scale
        noisy_bonds = alpha_coeff * edge_attr + (1 - alpha_coeff) * noise
        
        # Bond connectivity changes (remove/add bonds with small probability)
        edge_remove_prob = (1 - alpha_coeff) * 0.05  # Small prob to remove bonds
        edge_mask = torch.rand(edge_index.size(1), device=device) > edge_remove_prob
        
        if edge_mask.sum() < edge_index.size(1):
            # Remove some edges
            edge_index = edge_index[:, edge_mask]
            noisy_bonds = noisy_bonds[edge_mask]
        
        return noisy_bonds, edge_index
    
    def compute_enhanced_loss(self, model, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Enhanced loss that includes both position and graph structure"""
        device = batch['pos'].device
        t = torch.randint(0, self.num_timesteps, (1,), device=device)
        
        losses = {}
        total_loss = 0.0
        
        # 1. Position diffusion loss (main)
        pos_noise = torch.randn_like(batch['pos'])
        pos_t, _ = self.forward_process_positions(batch['pos'], t, pos_noise)
        
        # 2. Graph structure diffusion (if enabled)
        noisy_x = batch['x']  # Default: no noise
        noisy_edge_attr = batch['edge_attr']
        noisy_edge_index = batch['edge_index']
        
        if self.noise_graph_structure:
            noisy_x = self.forward_process_atoms(batch['x'], t)
            noisy_edge_attr, noisy_edge_index = self.forward_process_bonds(
                batch['edge_attr'], batch['edge_index'], t
            )
        
        # Prepare model inputs
        model_inputs = {
            'pos': pos_t,  # Noisy positions
            'x': noisy_x,  # Noisy atom features (if enabled)
            'edge_index': noisy_edge_index,  # Potentially modified connectivity
            'edge_attr': noisy_edge_attr,  # Noisy bond features (if enabled)
            'batch': batch['batch'],
            't': t
        }
        
        # Add pocket data if available
        if 'pocket_x' in batch:
            model_inputs.update({
                'pocket_x': batch['pocket_x'],
                'pocket_pos': batch['pocket_pos'],
                'pocket_edge_index': batch.get('pocket_edge_index'),
                'pocket_batch': batch.get('pocket_batch')
            })
        
        try:
            # Model forward pass
            model_output = model(**model_inputs)
            
            # 1. Position loss (main DDPM loss)
            if isinstance(model_output, dict) and 'pos_pred' in model_output:
                pos_pred = model_output['pos_pred']
            else:
                pos_pred = model_output
            
            pos_loss = F.mse_loss(pos_pred, pos_noise)
            losses['pos_loss'] = pos_loss.item()
            total_loss += pos_loss
            
            # 2. Graph structure losses (if enabled)
            if self.noise_graph_structure:
                # Atom type reconstruction loss
                if 'atom_pred' in model_output:
                    atom_loss = F.mse_loss(model_output['atom_pred'], batch['x'])
                    losses['atom_loss'] = atom_loss.item()
                    total_loss += 0.1 * atom_loss
                
                # Bond type reconstruction loss  
                if 'bond_pred' in model_output and batch['edge_attr'] is not None:
                    if model_output['bond_pred'].size(0) == batch['edge_attr'].size(0):
                        bond_loss = F.mse_loss(model_output['bond_pred'], batch['edge_attr'])
                        losses['bond_loss'] = bond_loss.item()
                        total_loss += 0.1 * bond_loss
            
            # 3. Constraint losses
            if isinstance(model_output, dict) and 'total_constraint_loss' in model_output:
                constraint_loss = model_output['total_constraint_loss']
                if constraint_loss is not None and constraint_loss.requires_grad:
                    losses['constraint_loss'] = constraint_loss.item()
                    total_loss += 0.1 * constraint_loss
            
            losses['total_loss'] = total_loss.item()
            return total_loss, losses
            
        except Exception as e:
            print(f"Enhanced DDPM loss error: {e}")
            dummy_loss = torch.tensor(1.0, device=device, requires_grad=True)
            return dummy_loss, {'pos_loss': 1.0, 'total_loss': 1.0}


# Usage example
def ddpm(noise_graph_structure=True):
    """Create enhanced DDPM with optional graph structure noise"""
    return MolecularDDPM(
        num_timesteps=1000,
        beta_schedule="cosine",
        noise_graph_structure=noise_graph_structure,
        atom_noise_scale=0.1,
        bond_noise_scale=0.2
    )

class DDPM_trainer:
    def __init__(self, base_model, enchanced_ddpm, optimizer, device = "cuda" ):
        self.base_model = base_model
        self.enchanced_ddpm = enchanced_ddpm
        self.optimizer = optimizer

        self.device = device

    def train_step(self, batch):
        self.optimizer.zero_grad()

        loss, loss_dict = self.ddpm.compute_enhanced_loss(self.model, batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss_dict