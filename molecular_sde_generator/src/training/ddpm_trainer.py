# src/training/ddpm_trainer.py - Much simpler than SDE trainer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, List
import numpy as np

from ..models.ddpm_diffusion import MolecularDDPM, MolecularDDPMModel
from .callbacks_fixed import TrainingCallback

class DDPMMolecularTrainer:
    """DDPM trainer - Much cleaner than SDE version"""
    
    def __init__(self, 
                 base_model,
                 ddpm: MolecularDDPM,
                 optimizer: optim.Optimizer,
                 scheduler=None,
                 device: str = 'cuda',
                 callbacks: List[TrainingCallback] = None):
        
        # Wrap base model with DDPM
        self.model = MolecularDDPMModel(base_model, ddpm)
        self.ddpm = ddpm
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.callbacks = callbacks or []
        
        # Move schedules to device
        self.ddpm.betas = self.ddpm.betas.to(device)
        self.ddpm.alphas = self.ddpm.alphas.to(device)
        self.ddpm.alphas_cumprod = self.ddpm.alphas_cumprod.to(device)
        self.ddpm.sqrt_alphas_cumprod = self.ddpm.sqrt_alphas_cumprod.to(device)
        self.ddpm.sqrt_one_minus_alphas_cumprod = self.ddpm.sqrt_one_minus_alphas_cumprod.to(device)
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train one epoch - Much simpler than SDE"""
        self.model.train()
        epoch_losses = {'total_loss': 0.0, 'noise_loss': 0.0}
        num_batches = 0
        
        # Callbacks
        for callback in self.callbacks:
            callback.on_epoch_start(self.current_epoch, self)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None:
                continue
            
            batch = batch.to(self.device)
            
            # Callback: batch start
            for callback in self.callbacks:
                callback.on_batch_start(batch_idx, batch, self)
            
            # DDPM loss computation (MUCH simpler than SDE)
            try:
                loss, loss_dict = self.ddpm.compute_loss(
                    model=self.model,
                    x0=batch.pos,  # Target positions
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch,
                    # Pocket conditioning
                    pocket_x=getattr(batch, 'pocket_x', None),
                    pocket_pos=getattr(batch, 'pocket_pos', None),
                    pocket_edge_index=getattr(batch, 'pocket_edge_index', None),
                    pocket_batch=getattr(batch, 'pocket_batch', None)
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                # Update metrics
                epoch_losses['total_loss'] += loss.item()
                epoch_losses['noise_loss'] += loss_dict['noise_loss']
                num_batches += 1
                self.global_step += 1
                
                # Update progress
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # Callback: batch end
                batch_outputs = {'losses': loss_dict, 'loss': loss}
                for callback in self.callbacks:
                    callback.on_batch_end(batch_idx, batch_outputs, self)
                
            except Exception as e:
                print(f"Training error at batch {batch_idx}: {e}")
                continue
        
        # Average losses
        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validation - Also much simpler"""
        self.model.eval()
        val_losses = {'total_loss': 0.0, 'noise_loss': 0.0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                if batch is None:
                    continue
                
                batch = batch.to(self.device)
                
                try:
                    loss, loss_dict = self.ddpm.compute_loss(
                        model=self.model,
                        x0=batch.pos,
                        x=batch.x,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        batch=batch.batch,
                        pocket_x=getattr(batch, 'pocket_x', None),
                        pocket_pos=getattr(batch, 'pocket_pos', None),
                        pocket_edge_index=getattr(batch, 'pocket_edge_index', None),
                        pocket_batch=getattr(batch, 'pocket_batch', None)
                    )
                    
                    val_losses['total_loss'] += loss.item()
                    val_losses['noise_loss'] += loss_dict['noise_loss']
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Validation error: {e}")
                    continue
        
        # Average losses
        if num_batches > 0:
            for key in val_losses:
                val_losses[key] /= num_batches
        
        return val_losses
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int, save_path: str = None):
        """Full training loop"""
        
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
                    print(f"💾 Best DDPM model saved!")
            
            # Callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, epoch_metrics, self)
                
                if hasattr(callback, 'should_stop') and callback.should_stop:
                    print("🛑 Early stopping!")
                    break
            else:
                continue
            break
    
    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """Save checkpoint"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'ddpm_config': {
                'num_timesteps': self.ddpm.num_timesteps,
                'betas': self.ddpm.betas.cpu(),
            },
            'loss': loss,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, path)
    
    def generate_samples(self, num_samples: int = 10, max_atoms: int = 30, **kwargs):
        """Generate molecular samples"""
        self.model.eval()
        
        # Sample shapes (simplified)
        shape = (num_samples, max_atoms, 3)  # positions
        
        with torch.no_grad():
            samples = self.ddpm.sample(
                model=self.model,
                shape=shape,
                device=self.device,
                **kwargs
            )
        
        return samples

# Usage comparison
def create_ddpm_trainer(base_model, device='cuda'):
    """Much simpler setup than SDE"""
    
    # Create DDPM (much simpler than VESDE)
    ddpm = MolecularDDPM(
        num_timesteps=1000,
        beta_schedule="cosine"  # Just this!
    )
    
    # Create optimizer
    optimizer = optim.AdamW(base_model.parameters(), lr=1e-4)
    
    # Create trainer (much cleaner)
    trainer = DDPMMolecularTrainer(
        base_model=base_model,
        ddpm=ddpm,
        optimizer=optimizer,
        device=device
    )
    
    return trainer