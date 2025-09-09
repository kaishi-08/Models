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
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# Add src to Python path
sys.path.append('src')

# Import model components
from src.models.model import ConditionalDDPMViSNet
from src.data.data_loaders import CrossDockDataLoader
from src.utils.molecular_utils import MolecularMetrics
from src.utils.evaluation_utils import MolecularEvaluator

# Set up logging with UTF-8 encoding to handle non-ASCII paths
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
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
        
        # Additional regularization losses
        reg_loss = torch.tensor(0.0, device=vlb_loss.device, dtype=torch.float64)
        
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
    def __init__(self, config):
        self.config = config
        requested_device = config.get('device', 'cuda')
        self.precision = torch.float64 if config.get('precision', 'float64') == 'float64' else torch.float32
        
        # Check CUDA availability and fall back to CPU if needed
        if requested_device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(requested_device)
        
        logger.info(f"Using device: {self.device}, Precision: {self.precision}")
        
        # Create output directory
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize dataset info
        self.dataset_info = self._load_dataset_info()
        
        # Initialize model
        self.model = self._build_model()
        self.model.to(self.device).to(self.precision)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,}")
        
        # Test initial equivariance
        self._test_model_equivariance()
        
        # Initialize optimizer
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Loss computation
        self.loss_fn = DDPMLoss(config.get('loss', {}))
        
        # Evaluator
        self.evaluator = MolecularEvaluator(config.get('evaluation', {}))
        
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
        logger.info(f"Model config: {model_config}")
        model = ConditionalDDPMViSNet(
            atom_nf=self.dataset_info['atom_nf'],
            residue_nf=self.dataset_info['residue_nf'],
            n_dims=3,
            size_histogram=self.dataset_info.get('size_histogram'),
            hidden_nf=model_config.get('hidden_nf', 256),
            num_layers=model_config.get('num_layers', 6),
            num_heads=model_config.get('num_heads', 8),
            lmax=model_config.get('lmax', 2),
            vecnorm_type=model_config.get('vecnorm_type', 'max_min'),
            trainable_vecnorm=model_config.get('trainable_vecnorm', True),
            edge_cutoff_ligand=model_config.get('edge_cutoff_ligand', 5.0),
            edge_cutoff_pocket=model_config.get('edge_cutoff_pocket', 8.0),
            edge_cutoff_interaction=model_config.get('edge_cutoff_interaction', 5.0),
            activation=model_config.get('activation', 'silu'),
            cutoff=model_config.get('cutoff', 5.0),
            update_pocket_coords=model_config.get('update_pocket_coords', False),
            timesteps=model_config.get('timesteps', 1000),
            parametrization=model_config.get('parametrization', 'eps'),
            noise_schedule=model_config.get('noise_schedule', 'cosine'),
            noise_precision=model_config.get('noise_precision', 1e-4),
            loss_type=model_config.get('loss_type', 'vlb'),
            norm_values=tuple(model_config.get('norm_values', [1.0, 1.0])),
            norm_biases=tuple(model_config.get('norm_biases', [None, 0.0])),
        )
        logger.info(f"Model built with lmax={model_config.get('lmax', 2)}, cutoff={model_config.get('cutoff', 5.0)}")
        return model
    
    def _test_model_equivariance(self):
        """Test that the model maintains SE(3) equivariance"""
        logger.info("Testing SE(3) equivariance...")
        n_atoms = 10
        n_residues = 20
        ligand = {
            'x': torch.randn(n_atoms, 3, device=self.device, dtype=self.precision),
            'one_hot': torch.zeros(n_atoms, self.dataset_info['atom_nf'], device=self.device, dtype=self.precision),
            'mask': torch.zeros(n_atoms, dtype=torch.long, device=self.device),
            'size': torch.tensor([n_atoms], device=self.device)
        }
        pocket = {
            'x': torch.randn(n_residues, 3, device=self.device, dtype=self.precision),
            'one_hot': torch.zeros(n_residues, self.dataset_info['residue_nf'], device=self.device, dtype=self.precision),
            'mask': torch.zeros(n_residues, dtype=torch.long, device=self.device),
            'size': torch.tensor([n_residues], device=self.device)
        }
        ligand['one_hot'][0, 0] = 1  # Carbon atom
        pocket['one_hot'][0, 0] = 1  # Alanine residue
        print("Ligand shapes:", {k: v.shape for k, v in ligand.items()})
        print("Pocket shapes:", {k: v.shape for k, v in pocket.items()})
        try:
            loss_terms, info = self.model(ligand, pocket, return_info=True)
            logger.info("Equivariance test passed (dummy forward successful)")
        except Exception as e:
            logger.error(f"Equivariance test failed: {e}")
            raise
    
    def _build_optimizer(self):
        """Build optimizer"""
        opt_config = self.config.get('optimizer', {})
        lr = opt_config.get('lr', 1e-4)
        weight_decay = opt_config.get('weight_decay', 1e-5)
        betas = opt_config.get('betas', [0.9, 0.999])
        
        if opt_config.get('type', 'adam') == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas
            )
        return torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas
        )
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        sched_config = self.config.get('scheduler', {})
        scheduler_type = sched_config.get('type', 'none')
        
        if scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config.get('T_max', self.config['training']['epochs']),
                eta_min=sched_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            return StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 50),
                gamma=sched_config.get('gamma', 0.5)
            )
        return None
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging with UTF-8 path handling"""
        wandb_config = self.config.get('wandb', {})
        try:
            wandb.init(
                project=wandb_config.get('project', 'molecular-ddpm-generation'),
                entity=wandb_config.get('entity', None),
                name=wandb_config.get('run_name', f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
                config=self.config,
                tags=wandb_config.get('tags', []),
                dir=str(self.output_dir)  # Convert Path to string to avoid encoding issues
            )
            wandb.watch(self.model)
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}. Continuing without W&B logging.")
    
    def prepare_batch(self, batch):
        if batch is None:
            logger.error("Received None batch")
            return None
        try: 
            if isinstance(batch, dict):
                logger.debug(f"Batch keys: {batch.keys()}")
            else:
                logger.debug(f"Batch type: {type(batch)}, content: {batch}")
            if not isinstance(batch, dict) or 'ligand' not in batch or 'pocket' not in batch:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    ligand_data, pocket_data = batch
                elif isinstance(batch, dict) and 'ligand_pos' in batch:
                    ligand = {
                        'x': batch['ligand_pos'].to(self.device).to(self.precision),
                        'one_hot': batch['ligand_features'].to(self.device).to(self.precision),
                        'mask': batch['ligand_mask'].to(self.device),
                        'size': batch['ligand_size'].to(self.device)
                    }
                    pocket = {
                        'x': batch['pocket_pos'].to(self.device).to(self.precision),
                        'one_hot': batch['pocket_features'].to(self.device).to(self.precision),
                        'mask': batch['pocket_mask'].to(self.device),
                        'size': batch['pocket_size'].to(self.device)
                    }
                else:
                    raise KeyError("Batch must be a dict with 'ligand' and 'pocket' keys or a tuple of (ligand, pocket)")
            else:
                ligand = {
                    'x': batch['ligand']['pos'].to(self.device).to(self.precision),
                    'one_hot': batch['ligand']['features'].to(self.device).to(self.precision),
                    'mask': batch['ligand']['mask'].to(self.device),
                    'size': batch['ligand']['size'].to(self.device)
                }
                pocket = {
                    'x': batch['pocket']['pos'].to(self.device).to(self.precision),
                    'one_hot': batch['pocket']['features'].to(self.device).to(self.precision),
                    'mask': batch['pocket']['mask'].to(self.device),
                    'size': batch['pocket']['size'].to(self.device)
                }
            logger.debug(f"Ligand shapes: {{'x': {ligand['x'].shape}, 'one_hot': {ligand['one_hot'].shape}, 'mask': {ligand['mask'].shape}, 'size': {ligand['size'].shape}}}")
            logger.debug(f"Pocket shapes: {{'x': {pocket['x'].shape}, 'one_hot': {pocket['one_hot'].shape}, 'mask': {pocket['mask'].shape}, 'size': {pocket['size'].shape}}}")
            return ligand, pocket
        except Exception as e:
            logger.error(f"Error preparing batch: {e}")
            return None
    
    def train_epoch(self, train_loader):
        """Train one epoch with equivariance monitoring"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {self.epoch}")):
            try:
                prepared_data = self.prepare_batch(batch)
                if prepared_data is None:
                    logger.warning(f"Skipping batch {batch_idx} due to preparation failure")
                    continue
                
                ligand, pocket = prepared_data
                self.optimizer.zero_grad()
                
                loss_terms, info = self.model(ligand, pocket, return_info=True)
                loss, loss_dict = self.loss_fn.compute_loss(loss_terms, info)
                
                loss.backward()
                if self.config['training'].get('clip_grad', 0.0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['clip_grad']
                    )
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                self.step += 1
                
                if self.config.get('use_wandb', False):
                    try:
                        wandb.log({
                            'train/loss': loss.item(),
                            'train/step': self.step,
                            **{f'train/{k}': v for k, v in loss_dict.items()}
                        })
                    except Exception as e:
                        logger.warning(f"W&B logging failed: {e}")
            
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Processed {num_batches} valid batches out of {batch_idx + 1}")
        
        if self.epoch % 10 == 0:
            self._test_model_equivariance()
        
        return avg_loss
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                try:
                    prepared_data = self.prepare_batch(batch)
                    if prepared_data is None:
                        logger.warning(f"Skipping validation batch {batch_idx} due to preparation failure")
                        continue
                    
                    ligand, pocket = prepared_data
                    loss_terms, info = self.model(ligand, pocket, return_info=True)
                    loss, loss_dict = self.loss_fn.compute_loss(loss_terms, info)
                    
                    batch_size = ligand['size'].sum().item()
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
                    
                    if self.config.get('use_wandb', False):
                        try:
                            wandb.log({
                                'val/batch_loss': loss.item(),
                                **{f'val/{k}': v for k, v in loss_dict.items()}
                            }, step=self.step)
                        except Exception as e:
                            logger.warning(f"W&B logging failed: {e}")
                
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / max(total_samples, 1)
        
        if self.epoch % self.config['evaluation'].get('evaluate_every', 20) == 0:
            eval_metrics = self.evaluator.evaluate(
                self.model,
                num_samples=self.config['evaluation'].get('num_samples', 100),
                sample_timesteps=self.config['evaluation'].get('sample_timesteps', 1000)
            )
            logger.info(f"Evaluation metrics: {eval_metrics}")
            if self.config.get('use_wandb', False):
                try:
                    wandb.log({f'eval/{k}': v for k, v in eval_metrics.items()}, step=self.step)
                except Exception as e:
                    logger.warning(f"W&B logging failed: {e}")
        
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
        
        if is_best and self.config['checkpointing'].get('save_best', True):
            best_path = filepath.parent / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
        
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
        self._test_model_equivariance()
    
    def train(self):
        """Main training loop with equivariance monitoring"""
        logger.info("Starting training with SE(3) equivariance monitoring...")
        
        train_config = self.config['data'].copy()
        train_config.update(self.config.get('training', {}))
        
        train_loader = CrossDockDataLoader.create_train_loader(train_config)
        val_loader = CrossDockDataLoader.create_val_loader(train_config)
        
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        num_epochs = self.config['training']['epochs']
        save_every = self.config['training'].get('save_every', 10)
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            train_loss = self.train_epoch(train_loader)
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}")
            
            if val_loader:
                val_loss = self.validate(val_loader)
                logger.info(f"Epoch {epoch}: Val Loss = {val_loss:.6f}")
                
                if self.config.get('use_wandb', False):
                    try:
                        wandb.log({
                            'val/loss': val_loss,
                            'val/epoch': epoch
                        }, step=self.step)
                    except Exception as e:
                        logger.warning(f"W&B logging failed: {e}")
                
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    logger.info(f"New best validation loss: {val_loss:.6f}")
            else:
                val_loss = train_loss
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
            
            if self.scheduler:
                if self.config['scheduler'].get('type') in ['cosine', 'step']:
                    self.scheduler.step()
            
            if (epoch + 1) % save_every == 0 or is_best:
                checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
                self.save_checkpoint(checkpoint_path, is_best=is_best)
        
        if self.config['checkpointing'].get('save_last', True):
            final_path = self.output_dir / 'final_model.pt'
            self.save_checkpoint(final_path)
        
        logger.info("Training completed!")

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Train molecular DDPM model with ViSNet')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--precision', type=str, default='float64', choices=['float32', 'float64'], help='Floating point precision')
    parser.add_argument('--test_equivariance', action='store_true', help='Test equivariance and exit')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.device:
        config['device'] = args.device
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.precision:
        config['precision'] = args.precision
    
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    trainer = DDPMTrainer(config)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    if args.test_equivariance:
        logger.info("Testing equivariance only...")
        return
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        interrupt_path = trainer.output_dir / 'interrupted_checkpoint.pt'
        trainer.save_checkpoint(interrupt_path)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if config.get('use_wandb', False):
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"W&B cleanup failed: {e}")

if __name__ == "__main__":
    main()
