# src/training/callbacks_fixed.py - Fixed callbacks
import torch
import numpy as np
from typing import Dict, Any, List, Optional
import os
from pathlib import Path

class TrainingCallback:
    """Base class for training callbacks"""
    
    def on_epoch_start(self, epoch: int, trainer):
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        pass
    
    def on_batch_start(self, batch_idx: int, batch, trainer):
        pass
    
    def on_batch_end(self, batch_idx: int, outputs: Dict[str, Any], trainer):
        pass

class WandBLogger(TrainingCallback):
    """Weights & Biases logging callback"""
    
    def __init__(self, project_name: str, log_frequency: int = 100):
        self.project_name = project_name
        self.log_frequency = log_frequency
        self.step = 0
    
    def on_batch_end(self, batch_idx: int, outputs: Dict[str, Any], trainer):
        if batch_idx % self.log_frequency == 0:
            try:
                import wandb
                # Log training metrics
                log_dict = {}
                if 'losses' in outputs:
                    for key, value in outputs['losses'].items():
                        if isinstance(value, torch.Tensor):
                            log_dict[f"train/{key}"] = value.item()
                        else:
                            log_dict[f"train/{key}"] = value
                
                log_dict['step'] = self.step
                wandb.log(log_dict)
            except ImportError:
                pass  # wandb not available
        
        self.step += 1
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        try:
            import wandb
            # Log epoch metrics
            epoch_log = {f"epoch/{key}": value for key, value in metrics.items()}
            epoch_log['epoch'] = epoch
            wandb.log(epoch_log)
        except ImportError:
            pass

class EarlyStopping(TrainingCallback):
    """Early stopping callback"""
    
    def __init__(self, monitor: str = 'val_loss', patience: int = 10,
                 min_delta: float = 0.001, mode: str = 'min'):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.wait = 0
        self.best_value = None
        self.should_stop = False
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        current_value = metrics.get(self.monitor)
        
        if current_value is None:
            return
        
        if self.best_value is None:
            self.best_value = current_value
        elif self.monitor_op(current_value, self.best_value + self.min_delta):
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            print(f"Early stopping triggered after {epoch} epochs")
            self.should_stop = True

class ModelCheckpoint(TrainingCallback):
    """Model checkpointing callback"""
    
    def __init__(self, save_path: str, monitor: str = 'val_loss',
                 save_best_only: bool = True, mode: str = 'min'):
        self.save_path = save_path
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        
        self.best_value = None
        
        if mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        current_value = metrics.get(self.monitor)
        
        if current_value is None:
            return
        
        should_save = False
        
        if not self.save_best_only:
            should_save = True
        elif self.best_value is None or self.monitor_op(current_value, self.best_value):
            self.best_value = current_value
            should_save = True
        
        if should_save:
            # Create directory if not exists
            os.makedirs(self.save_path, exist_ok=True)
            checkpoint_path = f"{self.save_path}/best_model_epoch_{epoch}.pth"
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict() if trainer.scheduler else None,
                'loss': current_value,
                'global_step': getattr(trainer, 'global_step', 0),
                'best_val_loss': getattr(trainer, 'best_val_loss', float('inf'))
            }
            
            torch.save(checkpoint, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

class MolecularVisualizationCallback(TrainingCallback):
    """Simplified molecular visualization callback"""
    
    def __init__(self, visualization_frequency: int = 10, num_samples: int = 4):
        self.visualization_frequency = visualization_frequency
        self.num_samples = num_samples
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        if epoch % self.visualization_frequency == 0:
            print(f"🧪 Generating sample molecules at epoch {epoch}...")
            # This is a placeholder for molecular generation
            # You can implement this later when inference is ready
            pass