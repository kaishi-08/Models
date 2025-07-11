# src/training/callbacks.py
import torch
import wandb
import numpy as np
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
import io
import base64

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
        
        self.step += 1
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        # Log epoch metrics
        epoch_log = {f"epoch/{key}": value for key, value in metrics.items()}
        epoch_log['epoch'] = epoch
        wandb.log(epoch_log)

class MolecularVisualizationCallback(TrainingCallback):
    """Callback for visualizing generated molecules"""
    
    def __init__(self, visualization_frequency: int = 10, num_samples: int = 4):
        self.visualization_frequency = visualization_frequency
        self.num_samples = num_samples
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        if epoch % self.visualization_frequency == 0:
            # Generate sample molecules
            trainer.model.eval()
            with torch.no_grad():
                samples = self._generate_samples(trainer)
                if samples:
                    self._log_molecular_visualizations(samples, epoch)
            trainer.model.train()
    
    def _generate_samples(self, trainer) -> List[str]:
        """Generate sample molecules for visualization"""
        try:
            # This would use the conditional generator
            from ..inference.conditional_generator import ConditionalMolecularGenerator
            
            generator = ConditionalMolecularGenerator(
                model=trainer.model,
                sde=trainer.sde,
                device=trainer.device
            )
            
            # Generate without pocket conditioning for simplicity
            samples = generator.generate_molecules(
                pocket_data={},
                num_molecules=self.num_samples,
                max_atoms=30
            )
            
            return generator.molecules_to_smiles(samples['molecules'])
        
        except Exception as e:
            print(f"Error generating samples: {e}")
            return []
    
    def _log_molecular_visualizations(self, smiles_list: List[str], epoch: int):
        """Log molecular visualizations to wandb"""
        valid_molecules = []
        
        for smiles in smiles_list:
            if smiles:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        valid_molecules.append(mol)
                except:
                    continue
        
        if valid_molecules:
            # Create molecular images
            imgs = []
            for mol in valid_molecules[:4]:  # Limit to 4 molecules
                img = Draw.MolToImage(mol, size=(300, 300))
                
                # Convert to base64 for wandb
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                imgs.append(wandb.Image(img, caption=Chem.MolToSmiles(mol)))
            
            wandb.log({f"generated_molecules_epoch_{epoch}": imgs})

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
            checkpoint_path = f"{self.save_path}/best_model_epoch_{epoch}.pth"
            trainer.model.save_checkpoint(
                path=checkpoint_path,
                optimizer_state=trainer.optimizer.state_dict(),
                scheduler_state=trainer.scheduler.state_dict() if trainer.scheduler else None,
                epoch=epoch,
                loss=current_value
            )
            print(f"Model saved to {checkpoint_path}")