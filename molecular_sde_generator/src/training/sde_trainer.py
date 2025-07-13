# src/training/sde_trainer.py (Enhanced version)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from typing import Dict, Any, List, Optional
import numpy as np
import os
from pathlib import Path

from ..models.joint_2d_3d_model import Joint2D3DMolecularModel
from ..models.sde_diffusion import VESDE
from .losses import MolecularLoss, ScoreMatchingLoss
from .callbacks import TrainingCallback

class SDEMolecularTrainer:
    """Enhanced trainer for SDE-based molecular generation"""
    
    def __init__(self, 
                 model: Joint2D3DMolecularModel, 
                 sde: VESDE,
                 optimizer: optim.Optimizer, 
                 scheduler=None,
                 device: str = 'cuda', 
                 log_wandb: bool = True,
                 callbacks: List[TrainingCallback] = None):
        
        self.model = model
        self.sde = sde
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_wandb = log_wandb
        self.callbacks = callbacks or []
        
        # Loss functions
        self.score_loss = ScoreMatchingLoss()
        self.molecular_loss = MolecularLoss()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            'total_loss': 0.0,
            'score_loss': 0.0,
            'atom_loss': 0.0,
            'bond_loss': 0.0
        }
        num_batches = 0
        
        # Callback: epoch start
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
            
            # Sample random times for each molecule in batch
            batch_size = batch.batch.max().item() + 1
            t = torch.rand(batch_size, device=self.device)
            t_expanded = t[batch.batch]
            
            # Add noise according to SDE
            mean, std = self.sde.marginal_prob(batch.pos, t_expanded)
            noise = torch.randn_like(batch.pos)
            perturbed_pos = mean + std[:, None] * noise
            
            # Forward pass
            try:
                outputs = self.model(
                    x=batch.x,
                    pos=perturbed_pos,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch,
                    pocket_x=getattr(batch, 'pocket_x', None),
                    pocket_pos=getattr(batch, 'pocket_pos', None),
                    pocket_edge_index=getattr(batch, 'pocket_edge_index', None),
                    pocket_batch=getattr(batch, 'pocket_batch', None)
                )
            except Exception as e:
                print(f"Forward pass error: {e}")
                continue
            
            # Compute losses
            losses = self._compute_losses(outputs, batch, noise, std, t_expanded)
            total_loss = losses['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Update running losses
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    epoch_losses[key] += value.item()
                else:
                    epoch_losses[key] += value
            
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log to wandb
            if self.log_wandb and self.global_step % 50 == 0:
                log_dict = {f'train/{k}': v.item() if isinstance(v, torch.Tensor) else v 
                           for k, v in losses.items()}
                log_dict['train/lr'] = self.optimizer.param_groups[0]['lr']
                log_dict['global_step'] = self.global_step
                wandb.log(log_dict)
            
            # Callback: batch end
            batch_outputs = {'losses': losses, 'outputs': outputs}
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, batch_outputs, self)
        
        # Average losses
        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def _compute_losses(self, outputs: Dict[str, torch.Tensor], 
                       batch, noise: torch.Tensor, std: torch.Tensor,
                       t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all losses"""
        losses = {}
        
        # Score matching loss (main SDE loss)
        score_target = -noise / std[:, None]
        score_pred = outputs['pos_pred']
        score_loss = self.score_loss(score_pred, score_target, batch.pos, t)
        losses['score_loss'] = score_loss
        
        # Atom type loss
        atom_loss = torch.tensor(0.0, device=self.device)
        if 'atom_logits' in outputs and hasattr(batch, 'x'):
            atom_targets = batch.x.squeeze(-1).long()
            atom_loss = nn.CrossEntropyLoss()(outputs['atom_logits'], atom_targets)
            losses['atom_loss'] = atom_loss
        
        # Bond type loss
        bond_loss = torch.tensor(0.0, device=self.device)
        if 'bond_logits' in outputs and hasattr(batch, 'edge_attr'):
            bond_targets = batch.edge_attr.squeeze(-1).long()
            bond_loss = nn.CrossEntropyLoss()(outputs['bond_logits'], bond_targets)
            losses['bond_loss'] = bond_loss
        
        # Total loss
        total_loss = score_loss + 0.1 * atom_loss + 0.1 * bond_loss
        losses['total_loss'] = total_loss
        
        return losses
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        val_losses = {
            'total_loss': 0.0,
            'score_loss': 0.0,
            'atom_loss': 0.0,
            'bond_loss': 0.0
        }
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                if batch is None:
                    continue
                    
                batch = batch.to(self.device)
                
                # Sample random times
                batch_size = batch.batch.max().item() + 1
                t = torch.rand(batch_size, device=self.device)
                t_expanded = t[batch.batch]
                
                # Add noise according to SDE
                mean, std = self.sde.marginal_prob(batch.pos, t_expanded)
                noise = torch.randn_like(batch.pos)
                perturbed_pos = mean + std[:, None] * noise
                
                try:
                    # Forward pass
                    outputs = self.model(
                        x=batch.x,
                        pos=perturbed_pos,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        batch=batch.batch,
                        pocket_x=getattr(batch, 'pocket_x', None),
                        pocket_pos=getattr(batch, 'pocket_pos', None),
                        pocket_edge_index=getattr(batch, 'pocket_edge_index', None),
                        pocket_batch=getattr(batch, 'pocket_batch', None)
                    )
                    
                    # Compute losses
                    losses = self._compute_losses(outputs, batch, noise, std, t_expanded)
                    
                    # Update running losses
                    for key, value in losses.items():
                        if isinstance(value, torch.Tensor):
                            val_losses[key] += value.item()
                        else:
                            val_losses[key] += value
                    
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
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
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
            
            # Log metrics
            if self.log_wandb:
                log_dict = {**epoch_metrics, 'epoch': epoch}
                wandb.log(log_dict)
            
            print(f"Train Loss: {train_metrics['total_loss']:.4f}, "
                  f"Val Loss: {val_metrics['total_loss']:.4f}")
            
            # Save best model
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                if save_path:
                    self.save_checkpoint(save_path, epoch, val_metrics['total_loss'])
                    print(f"Best model saved to {save_path}")
            
            # Callback: epoch end
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, epoch_metrics, self)
                
                # Check early stopping
                if hasattr(callback, 'should_stop') and callback.should_stop:
                    print("Early stopping triggered!")
                    break
            else:
                continue  # Only executed if break was not called
            break  # Early stopping
    
    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': loss,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Checkpoint loaded from epoch {self.current_epoch}")
        return checkpoint