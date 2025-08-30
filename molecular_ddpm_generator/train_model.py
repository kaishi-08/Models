import os
import sys
import pickle
import yaml
import argparse
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime
import logging

# Add src to Python path
sys.path.append('src')

# Import model components
from src.models.model import ConditionalDDPMViSNet
from src.models.vis_dynamics import ViSNetDynamics
from src.data.data_loaders import CrossDockDataLoader
from src.utils.molecular_utils import MolecularMetrics
from src.utils.evaluation_utils import MolecularEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DDPMLoss:
    """Loss computation for DDPM training"""
    
    def __init__(self, config):
        self.loss_type = config.get('loss_type', 'vlb')
        self.weighted_loss = config.get('weighted_loss', True)
        
    def compute_loss(self, loss_terms, info=None):
        """Compute total loss from DDPM loss terms"""
        (delta_log_px, error_t_lig, error_t_pocket, SNR_weight,
         loss_0_x_ligand, loss_0_x_pocket, loss_0_h,
         neg_log_constants, kl_prior, log_pN, t_int, xh_lig_hat) = loss_terms
        
        # Weighted L2 loss
        if self.weighted_loss and SNR_weight is not None:
            loss_t = (error_t_lig + error_t_pocket) * SNR_weight
        else:
            loss_t = error_t_lig + error_t_pocket
        
        # Reconstruction loss at t=0
        loss_0 = loss_0_x_ligand + loss_0_x_pocket + loss_0_h
        
        # Total VLB loss
        if self.loss_type == 'vlb':
            vlb_loss = loss_t.mean() + loss_0.mean() + kl_prior.mean() + neg_log_constants.mean() - log_pN.mean() - delta_log_px.mean()
        else:
            vlb_loss = loss_t.mean() + loss_0.mean()
        
        # Additional losses
        reg_loss = torch.tensor(0.0, device=vlb_loss.device)
        
        # Position regularization
        if xh_lig_hat is not None:
            pos_reg = torch.norm(xh_lig_hat[:, :3], dim=-1).mean()
            reg_loss += 0.01 * pos_reg
        
        total_loss = vlb_loss + reg_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'vlb_loss': vlb_loss.item(),
            'error_t': loss_t.mean().item(),
            'loss_0': loss_0.mean().item(),
            'kl_prior': kl_prior.mean().item(),
            'reg_loss': reg_loss.item(),
        }
        
        if info:
            loss_dict.update({
                'eps_hat_lig_x': info.get('eps_hat_lig_x', 0.0),
                'eps_hat_lig_h': info.get('eps_hat_lig_h', 0.0),
            })
        
        return total_loss, loss_dict

class DDPMTrainer:
    """Main trainer class for DDPM molecular generation"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        logger.info(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize dataset info
        self.dataset_info = self._load_dataset_info()
        
        # Initialize model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,}")
        
        # Initialize optimizer
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Loss computation
        self.loss_fn = DDPMLoss(config.get('loss', {}))
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_val_loss = float('inf')
        
        # Initialize W&B if configured
        if config.get('use_wandb', False):
            self._init_wandb()
    
    def _load_dataset_info(self):
        """Load dataset information"""
        dataset_info_path = Path(self.config['data']['processed_dir']) / 'dataset_info.pkl'
        
        if dataset_info_path.exists():
            with open(dataset_info_path, 'rb') as f:
                dataset_info = pickle.load(f)
            logger.info("Loaded dataset info")
            return dataset_info
        else:
            logger.warning("Dataset info not found - using defaults")
            return {
                'atom_nf': 11,
                'residue_nf': 21,
                'size_histogram': None
            }
    
    def _build_model(self):
        """Build DDPM model with ViSNet dynamics"""
        model_config = self.config['model']
        
        # Build dynamics network (ViSNet)
        dynamics = ViSNetDynamics(
            atom_nf=self.dataset_info['atom_nf'],
            residue_nf=self.dataset_info['residue_nf'],
            n_dims=3,
            hidden_nf=model_config.get('hidden_dim', 256),
            condition_time=True,
            update_pocket_coords=model_config.get('update_pocket_coords', False),
            edge_cutoff_ligand=model_config.get('edge_cutoff_ligand', 5.0),
            edge_cutoff_pocket=model_config.get('edge_cutoff_pocket', 5.0),
            edge_cutoff_interaction=model_config.get('edge_cutoff_interaction', 5.0),
            num_layers=model_config.get('num_layers', 6),
            num_heads=model_config.get('num_heads', 8),
            cutoff=model_config.get('cutoff', 5.0),
            activation=model_config.get('activation', 'silu')
        )
        
        # Build DDPM model
        model = ConditionalDDPMViSNet(
            dynamics=dynamics,
            atom_nf=self.dataset_info['atom_nf'],
            residue_nf=self.dataset_info['residue_nf'],
            n_dims=3,
            size_histogram=self.dataset_info.get('size_histogram'),
            timesteps=model_config.get('timesteps', 1000),
            parametrization=model_config.get('parametrization', 'eps'),
            noise_schedule=model_config.get('noise_schedule', 'cosine'),
            noise_precision=model_config.get('noise_precision', 1e-4),
            loss_type=model_config.get('loss_type', 'vlb'),
            norm_values=tuple(model_config.get('norm_values', [1.0, 1.0])),
            norm_biases=tuple(model_config.get('norm_biases', [None, 0.0]))
        )
        
        return model
    
    def _build_optimizer(self):
        """Build optimizer"""
        opt_config = self.config.get('optimizer', {})
        
        if opt_config.get('type', 'adamw').lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_config.get('lr', 1e-4),
                weight_decay=opt_config.get('weight_decay', 1e-5),
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=opt_config.get('lr', 1e-4),
                weight_decay=opt_config.get('weight_decay', 1e-5)
            )
        
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        sched_config = self.config.get('scheduler', {})
        
        if sched_config.get('type') == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=sched_config.get('min_lr', 1e-6)
            )
        elif sched_config.get('type') == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 50),
                gamma=sched_config.get('gamma', 0.5)
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _init_wandb(self):
        """Initialize Weights & Biases"""
        wandb_config = self.config.get('wandb', {})
        
        wandb.init(
            project=wandb_config.get('project', 'molecular-ddpm'),
            entity=wandb_config.get('entity', None),
            name=wandb_config.get('run_name', f"ddpm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
            config=self.config,
            tags=wandb_config.get('tags', ['molecular', 'ddpm', 'generation'])
        )
        
        wandb.watch(self.model, log_freq=100)
    
    def prepare_batch(self, batch):
        """Prepare batch data for model"""
        if batch is None:
            return None
        
        # Extract ligand data
        ligand = {
            'x': batch.pos,
            'one_hot': batch.x,
            'mask': batch.batch,
            'size': torch.bincount(batch.batch)
        }
        
        # Extract pocket data if available
        pocket = None
        if hasattr(batch, 'pocket_x') and batch.pocket_x is not None:
            pocket = {
                'x': batch.pocket_pos,
                'one_hot': batch.pocket_x,
                'mask': batch.pocket_batch,
                'size': torch.bincount(batch.pocket_batch) if batch.pocket_batch is not None else None
            }
        
        # Move to device
        ligand = {k: v.to(self.device) if v is not None else None for k, v in ligand.items()}
        if pocket:
            pocket = {k: v.to(self.device) if v is not None else None for k, v in pocket.items()}
        
        return ligand, pocket
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Prepare batch
                prepared_data = self.prepare_batch(batch)
                if prepared_data is None:
                    continue
                
                ligand, pocket = prepared_data
                
                # Forward pass
                self.optimizer.zero_grad()
                
                if pocket is not None:
                    model_output = self.model(ligand, pocket, return_info=True)
                else:
                    dummy_pocket = {
                        'x': torch.zeros((1, 3), device=self.device),
                        'one_hot': torch.zeros((1, self.dataset_info['residue_nf']), device=self.device),
                        'mask': torch.zeros(1, dtype=torch.long, device=self.device),
                        'size': torch.tensor([1], device=self.device)
                    }
                    model_output = self.model(ligand, dummy_pocket, return_info=True)
                loss_terms = model_output[:-1]
                info = model_output[-1]
                # Compute loss
                loss, loss_dict = self.loss_fn.compute_loss(loss_terms, info)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                clip_grad = self.config.get('training', {}).get('clip_grad', 1.0)
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                
                self.optimizer.step()
                
                # Update metrics
                batch_size = ligand['size'].sum().item()
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'vlb': f"{loss_dict['vlb_loss']:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # Log to wandb
                if self.config.get('use_wandb', False) and self.step % 10 == 0:
                    log_dict = {
                        'train/loss': loss.item(),
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'train/epoch': self.epoch,
                        'train/step': self.step
                    }
                    log_dict.update({f'train/{k}': v for k, v in loss_dict.items()})
                    wandb.log(log_dict, step=self.step)
                
                self.step += 1
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(total_samples, 1)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                try:
                    # Prepare batch
                    prepared_data = self.prepare_batch(batch)
                    if prepared_data is None:
                        continue
                    
                    ligand, pocket = prepared_data
                    
                    # Forward pass
                    if pocket is not None:
                        loss_terms, info = self.model(ligand, pocket, return_info=True)
                    else:
                        # Handle ligand-only case
                        dummy_pocket = {
                            'x': torch.zeros((1, 3), device=self.device),
                            'one_hot': torch.zeros((1, self.dataset_info['residue_nf']), device=self.device),
                            'mask': torch.zeros(1, dtype=torch.long, device=self.device),
                            'size': torch.tensor([1], device=self.device)
                        }
                        loss_terms, info = self.model(ligand, dummy_pocket, return_info=True)
                    
                    # Compute loss
                    loss, loss_dict = self.loss_fn.compute_loss(loss_terms, info)
                    
                    # Update metrics
                    batch_size = ligand['size'].sum().item()
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
                    
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue
        
        avg_loss = total_loss / max(total_samples, 1)
        return avg_loss
    
    def save_checkpoint(self, filepath, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'dataset_info': self.dataset_info,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = filepath.parent / 'best_model.pt'
            torch.save(checkpoint, best_path)
        
        logger.info(f"Saved checkpoint: {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        logger.info(f"Loading checkpoint: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Resumed from epoch {self.epoch}, step {self.step}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Load data
        train_config = self.config['data'].copy()
        train_config.update(self.config.get('training', {}))
        
        train_loader = CrossDockDataLoader.create_train_loader(train_config)
        val_loader = CrossDockDataLoader.create_val_loader(train_config)
        
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Training loop
        num_epochs = self.config['training']['epochs']
        save_every = self.config['training'].get('save_every', 10)
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}")
            
            # Validate
            if val_loader:
                val_loss = self.validate(val_loader)
                logger.info(f"Epoch {epoch}: Val Loss = {val_loss:.6f}")
                
                # Log validation
                if self.config.get('use_wandb', False):
                    wandb.log({
                        'val/loss': val_loss,
                        'val/epoch': epoch
                    }, step=self.step)
                
                # Save best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    logger.info(f"New best validation loss: {val_loss:.6f}")
            else:
                val_loss = train_loss
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0 or is_best:
                checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
                self.save_checkpoint(checkpoint_path, is_best=is_best)
        
        logger.info("Training completed!")
        
        # Final save
        final_path = self.output_dir / 'final_model.pt'
        self.save_checkpoint(final_path)

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Train molecular DDPM model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.device:
        config['device'] = args.device
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Set random seeds
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create trainer
    trainer = DDPMTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save checkpoint on interrupt
        interrupt_path = trainer.output_dir / 'interrupted_checkpoint.pt'
        trainer.save_checkpoint(interrupt_path)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if config.get('use_wandb', False):
            wandb.finish()

if __name__ == "__main__":
    main()