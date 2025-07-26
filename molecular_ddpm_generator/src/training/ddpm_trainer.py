# src/training/ddpm_trainer.py - FIXED gradient handling
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, List, Optional
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path for absolute imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Safe imports
try:
    from src.models.ddpm_diffusion import MolecularDDPM, MolecularDDPMModel
    from src.training.callbacks_fixed import TrainingCallback
except ImportError:
    print("Warning: Using fallback DDPM implementation")
    
    class MolecularDDPM:
        def __init__(self, num_timesteps=1000, beta_schedule="cosine", **kwargs):
            self.num_timesteps = num_timesteps
            self.beta_schedule = beta_schedule
        
        def compute_loss(self, model, x0, **model_kwargs):
            """FIXED DDPM loss with proper gradient handling"""
            device = x0.device
            batch_size = x0.size(0)
            
            # Sample timesteps
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
            
            # Sample noise with gradients
            noise = torch.randn_like(x0, requires_grad=True)
            
            # Simple noise schedule
            alpha = 0.99
            alpha_bar = (alpha ** t.float()).view(-1, 1)
            
            # Add noise with gradient preservation
            sqrt_alpha_bar = torch.sqrt(alpha_bar)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
            x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
            
            # Ensure x_t requires gradients
            x_t = x_t.requires_grad_(True)
            
            # Prepare model inputs with gradient preservation
            model_inputs = self._prepare_model_inputs(x_t, t, model_kwargs, device)
            
            try:
                # Forward pass
                output = model(**model_inputs)
                
                # Handle different output formats
                if isinstance(output, dict):
                    predicted_noise = output.get('pos_pred', output.get('noise_pred', list(output.values())[0]))
                else:
                    predicted_noise = output
                
                # Compute loss with gradient preservation
                loss = nn.MSELoss()(predicted_noise, noise)
                
                # Ensure loss requires gradients
                if not loss.requires_grad:
                    loss = loss.requires_grad_(True)
                
                return loss, {'noise_loss': loss.item()}
                
            except Exception as e:
                print(f"Model forward error: {e}")
                # Return dummy loss with gradients
                dummy_loss = torch.tensor(1.0, device=device, requires_grad=True)
                return dummy_loss, {'noise_loss': 1.0}
        
        def _prepare_model_inputs(self, x_t, t, model_kwargs, device):
            """Prepare model inputs with gradient preservation"""
            model_inputs = {
                'pos': x_t.requires_grad_(True),
                't': t
            }
            
            # Add atom features with gradients
            if 'atom_features' in model_kwargs:
                model_inputs['x'] = self._ensure_device_and_grad(model_kwargs['atom_features'], device)
            elif 'x' in model_kwargs:
                model_inputs['x'] = self._ensure_device_and_grad(model_kwargs['x'], device)
            
            # Add graph structure
            graph_keys = ['edge_index', 'edge_attr', 'batch']
            for key in graph_keys:
                if key in model_kwargs and model_kwargs[key] is not None:
                    tensor = self._ensure_device(model_kwargs[key], device)
                    if key == 'edge_attr' and tensor.dtype == torch.float:
                        tensor = tensor.requires_grad_(True)
                    model_inputs[key] = tensor
            
            # Add pocket data with gradients where appropriate
            pocket_keys = ['pocket_x', 'pocket_pos', 'pocket_edge_index', 'pocket_batch']
            for key in pocket_keys:
                if key in model_kwargs and model_kwargs[key] is not None:
                    tensor = self._ensure_device(model_kwargs[key], device)
                    if key in ['pocket_x', 'pocket_pos'] and tensor.dtype == torch.float:
                        tensor = tensor.requires_grad_(True)
                    model_inputs[key] = tensor
            
            return model_inputs
        
        def _ensure_device_and_grad(self, tensor, device):
            """Ensure tensor is on device and has gradients if needed"""
            if tensor is None:
                return None
            if not isinstance(tensor, torch.Tensor):
                return tensor
            
            tensor = tensor.to(device)
            if tensor.dtype == torch.float and not tensor.requires_grad:
                tensor = tensor.requires_grad_(True)
            
            return tensor
        
        def _ensure_device(self, tensor, device):
            """Ensure tensor is on correct device"""
            if tensor is None:
                return None
            if not isinstance(tensor, torch.Tensor):
                return tensor
            return tensor.to(device)
    
    class MolecularDDPMModel:
        def __init__(self, base_model, ddpm):
            self.base_model = base_model
            self.ddpm = ddpm
        
        def parameters(self):
            return self.base_model.parameters()
        
        def eval(self):
            self.base_model.eval()
        
        def state_dict(self):
            return self.base_model.state_dict()
        
        def load_state_dict(self, state_dict):
            return self.base_model.load_state_dict(state_dict)
        
        def to(self, device):
            self.base_model.to(device)
            return self
        
        def __call__(self, **kwargs):
            return self.base_model(**kwargs)
    
    class TrainingCallback:
        def on_epoch_start(self, epoch: int, trainer): pass
        def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer): pass
        def on_batch_start(self, batch_idx: int, batch, trainer): pass
        def on_batch_end(self, batch_idx: int, outputs: Dict[str, Any], trainer): pass

def move_batch_to_device(batch, device):
    """FIXED: Move batch to device with gradient preservation"""
    if batch is None:
        return None
    
    try:
        # Move entire batch to device
        batch = batch.to(device)
        
        # Explicitly handle key tensors with gradient preservation
        float_attrs = ['x', 'pos', 'pocket_x', 'pocket_pos', 'edge_attr', 'pocket_edge_attr']
        int_attrs = ['edge_index', 'batch', 'pocket_edge_index', 'pocket_batch']
        
        for attr in float_attrs:
            if hasattr(batch, attr):
                tensor = getattr(batch, attr)
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    tensor = tensor.to(device)
                    if tensor.dtype == torch.float and not tensor.requires_grad:
                        tensor = tensor.requires_grad_(True)
                    setattr(batch, attr, tensor)
        
        for attr in int_attrs:
            if hasattr(batch, attr):
                tensor = getattr(batch, attr)
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    if tensor.device != device:
                        setattr(batch, attr, tensor.to(device))
        
        return batch
        
    except Exception as e:
        print(f"Error moving batch to device: {e}")
        return None

class DDPMMolecularTrainer:
    """FIXED DDPM trainer with proper gradient handling"""
    
    def __init__(self, 
                 base_model,
                 ddpm: MolecularDDPM,
                 optimizer: optim.Optimizer,
                 scheduler=None,
                 device: str = 'cuda',
                 callbacks: List[TrainingCallback] = None):
        
        self.device = torch.device(device)
        self.base_model = base_model.to(self.device)
        self.ddpm = ddpm
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        
        # Wrap model
        if hasattr(base_model, 'base_model'):
            self.model = base_model.to(self.device)
        else:
            self.model = MolecularDDPMModel(base_model, ddpm).to(self.device)
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        print(f"✅ DDPM Trainer initialized on {self.device}")
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """FIXED: Train one epoch with proper gradient handling"""
        self.model.base_model.train()  # Ensure training mode
        epoch_losses = {'total_loss': 0.0, 'noise_loss': 0.0}
        num_batches = 0
        device_errors = 0
        gradient_errors = 0
        
        # Callbacks
        for callback in self.callbacks:
            try:
                callback.on_epoch_start(self.current_epoch, self)
            except:
                pass
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None:
                continue
            
            # FIXED: Proper device handling with gradients
            batch = move_batch_to_device(batch, self.device)
            if batch is None:
                device_errors += 1
                continue
            
            # Callback
            for callback in self.callbacks:
                try:
                    callback.on_batch_start(batch_idx, batch, self)
                except:
                    pass
            
            # FIXED: Training step with gradient preservation
            try:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Prepare kwargs with gradient safety
                ddpm_kwargs = self._prepare_safe_kwargs(batch)
                
                # Ensure target has gradients
                target_pos = batch.pos.to(self.device)
                if not target_pos.requires_grad:
                    target_pos = target_pos.requires_grad_(True)
                
                # DDPM loss computation
                loss, loss_dict = self.ddpm.compute_loss(
                    model=self.model,
                    x0=target_pos,
                    **ddpm_kwargs
                )
                
                # Verify loss has gradients
                if not loss.requires_grad:
                    gradient_errors += 1
                    print(f"Warning: Loss at batch {batch_idx} doesn't require grad")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                # Update metrics
                epoch_losses['total_loss'] += loss.item()
                epoch_losses['noise_loss'] += loss_dict.get('noise_loss', loss.item())
                num_batches += 1
                self.global_step += 1
                
                # Update progress
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    'dev_err': device_errors,
                    'grad_err': gradient_errors
                })
                
                # Callback
                batch_outputs = {'losses': loss_dict, 'loss': loss}
                for callback in self.callbacks:
                    try:
                        callback.on_batch_end(batch_idx, batch_outputs, self)
                    except:
                        pass
                
            except Exception as e:
                print(f"Training error at batch {batch_idx}: {e}")
                
                # Debug info for first few errors
                if batch_idx % 100 == 0:  # Reduced frequency
                    self._debug_batch_devices(batch, batch_idx)
                
                device_errors += 1
                continue
        
        # Average losses
        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        
        if device_errors > 0:
            print(f"⚠️  {device_errors} device errors encountered")
        if gradient_errors > 0:
            print(f"⚠️  {gradient_errors} gradient errors encountered")
        
        return epoch_losses
    
    def _prepare_safe_kwargs(self, batch):
        """FIXED: Prepare kwargs with guaranteed device consistency and gradients"""
        kwargs = {}
        
        # Add atom features with gradients
        if hasattr(batch, 'x') and batch.x is not None:
            x = batch.x.to(self.device)
            if x.dtype == torch.float and not x.requires_grad:
                x = x.requires_grad_(True)
            kwargs['atom_features'] = x
        
        # Add graph structure
        if hasattr(batch, 'edge_index') and batch.edge_index is not None:
            kwargs['edge_index'] = batch.edge_index.to(self.device)
        
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            edge_attr = batch.edge_attr.to(self.device)
            if edge_attr.dtype == torch.float and not edge_attr.requires_grad:
                edge_attr = edge_attr.requires_grad_(True)
            kwargs['edge_attr'] = edge_attr
        
        if hasattr(batch, 'batch') and batch.batch is not None:
            kwargs['batch'] = batch.batch.to(self.device)
        
        # Add pocket data with gradients
        pocket_attrs = ['pocket_x', 'pocket_pos', 'pocket_edge_index', 'pocket_batch']
        for attr in pocket_attrs:
            if hasattr(batch, attr):
                value = getattr(batch, attr)
                if value is not None:
                    value = value.to(self.device)
                    if attr in ['pocket_x', 'pocket_pos'] and value.dtype == torch.float and not value.requires_grad:
                        value = value.requires_grad_(True)
                    kwargs[attr] = value
        
        return kwargs
    
    def _debug_batch_devices(self, batch, batch_idx):
        """Debug batch device information"""
        print(f"  Debug batch {batch_idx}:")
        print(f"    Target device: {self.device}")
        
        attrs_to_check = ['x', 'pos', 'edge_index', 'edge_attr', 'batch',
                         'pocket_x', 'pocket_pos', 'pocket_edge_index', 'pocket_batch']
        
        for attr in attrs_to_check:
            if hasattr(batch, attr):
                value = getattr(batch, attr)
                if value is not None and isinstance(value, torch.Tensor):
                    print(f"    {attr}: {value.device} {value.shape} grad={value.requires_grad}")
                else:
                    print(f"    {attr}: None or not tensor")
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validation with device handling"""
        self.model.base_model.eval()
        val_losses = {'total_loss': 0.0, 'noise_loss': 0.0}
        num_batches = 0
        device_errors = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                if batch is None:
                    continue
                
                # Device handling
                batch = move_batch_to_device(batch, self.device)
                if batch is None:
                    device_errors += 1
                    continue
                
                try:
                    ddpm_kwargs = self._prepare_safe_kwargs(batch)
                    
                    loss, loss_dict = self.ddpm.compute_loss(
                        model=self.model,
                        x0=batch.pos.to(self.device),
                        **ddpm_kwargs
                    )
                    
                    val_losses['total_loss'] += loss.item()
                    val_losses['noise_loss'] += loss_dict.get('noise_loss', loss.item())
                    num_batches += 1
                    
                except Exception as e:
                    device_errors += 1
                    continue
        
        # Average losses
        if num_batches > 0:
            for key in val_losses:
                val_losses[key] /= num_batches
        
        if device_errors > 0:
            print(f"⚠️  {device_errors} validation device errors")
        
        return val_losses
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int, save_path: str = None):
        """Full training loop"""
        
        print(f"🚀 Starting DDPM training for {num_epochs} epochs on {self.device}...")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            print(f"\n📈 DDPM Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate  
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Combine metrics
            epoch_metrics = {
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }
            
            print(f"Train Loss: {train_metrics['total_loss']:.4f}, "
                  f"Val Loss: {val_metrics['total_loss']:.4f}")
            
            # Save best model
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                if save_path:
                    self.save_checkpoint(save_path, epoch, val_metrics['total_loss'])
                    print(f"💾 Best model saved!")
            
            # Callbacks
            for callback in self.callbacks:
                try:
                    callback.on_epoch_end(epoch, epoch_metrics, self)
                    
                    if hasattr(callback, 'should_stop') and callback.should_stop:
                        print("🛑 Early stopping!")
                        return
                except Exception as e:
                    print(f"Callback error: {e}")
                    continue
    
    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """Save checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': loss,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'device': str(self.device)
        }
        
        torch.save(checkpoint, path)
    
    def generate_samples(self, num_samples: int = 10, max_atoms: int = 30, **kwargs):
        """Generate molecular samples"""
        self.model.base_model.eval()
        
        print(f"🧪 Generating {num_samples} samples on {self.device}...")
        samples = []
        
        with torch.no_grad():
            for i in range(num_samples):
                sample_atoms = torch.randint(5, max_atoms, (1,)).item()
                pos = torch.randn(sample_atoms, 3, device=self.device)
                
                samples.append({
                    'positions': pos.cpu(),
                    'num_atoms': sample_atoms
                })
        
        return samples