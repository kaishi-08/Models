# src/training/sde_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from typing import Dict, Any

class SDEMolecularTrainer:
    """Trainer for SDE-based molecular generation"""
    
    def __init__(self, model: Joint2D3DMolecularModel, sde: VESDE,
                 optimizer: optim.Optimizer, scheduler=None,
                 device: str = 'cuda', log_wandb: bool = True):
        self.model = model
        self.sde = sde
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_wandb = log_wandb
        
        # Loss function
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            batch = batch.to(self.device)
            
            # Sample random times
            t = torch.rand(batch.batch.max().item() + 1, device=self.device)
            t = t[batch.batch]
            
            # Add noise according to SDE
            mean, std = self.sde.marginal_prob(batch.pos, t)
            noise = torch.randn_like(batch.pos)
            perturbed_pos = mean + std[:, None] * noise
            
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
            
            # Compute score target
            score_target = -noise / std[:, None]
            
            # Compute loss
            score_pred = outputs['pos_pred']
            loss = self.criterion(score_pred, score_target)
            
            # Add additional losses
            if 'atom_logits' in outputs:
                atom_loss = nn.CrossEntropyLoss()(outputs['atom_logits'], batch.x.squeeze())
                loss += atom_loss
            
            if 'bond_logits' in outputs:
                bond_loss = nn.CrossEntropyLoss()(outputs['bond_logits'], batch.edge_attr.squeeze())
                loss += bond_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if self.log_wandb:
                wandb.log({'batch_loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        
        if self.scheduler:
            self.scheduler.step()
            
        return {'train_loss': avg_loss}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                batch = batch.to(self.device)
                
                # Sample random times
                t = torch.rand(batch.batch.max().item() + 1, device=self.device)
                t = t[batch.batch]
                
                # Add noise according to SDE
                mean, std = self.sde.marginal_prob(batch.pos, t)
                noise = torch.randn_like(batch.pos)
                perturbed_pos = mean + std[:, None] * noise
                
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
                
                # Compute score target
                score_target = -noise / std[:, None]
                
                # Compute loss
                score_pred = outputs['pos_pred']
                loss = self.criterion(score_pred, score_target)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int, save_path: str = None):
        """Full training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log metrics
            if self.log_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['train_loss'],
                    'val_loss': val_metrics['val_loss']
                })
            
            print(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Val Loss: {val_metrics['val_loss']:.4f}")
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': best_val_loss,
                    }, save_path)
                    print(f"Best model saved to {save_path}")