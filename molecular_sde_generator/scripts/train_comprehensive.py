# scripts/train_comprehensive.py - Comprehensive training với monitoring chi tiết

import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import yaml
import wandb
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append('src')

from models.joint_2d_3d_model import Joint2D3DMolecularModel
from models.sde_diffusion import VESDE
from training.sde_trainer import SDEMolecularTrainer
from data.data_loaders import CrossDockDataLoader
from training.callbacks_fixed import (
    WandBLogger, EarlyStopping, ModelCheckpoint, 
    MolecularVisualizationCallback
)
from utils.molecular_utils import MolecularMetrics

class ComprehensiveTrainer:
    """Comprehensive trainer với monitoring và validation chi tiết"""
    
    def __init__(self, config: Dict[str, Any], args):
        self.config = config
        self.args = args
        self.device = self._setup_device()
        self.run_name = self._create_run_name()
        
        # Initialize components
        self.model = None
        self.sde = None
        self.optimizer = None
        self.scheduler = None
        self.trainer = None
        
        # Training state
        self.start_time = time.time()
        self.best_metrics = {'val_loss': float('inf')}
        self.training_history = []
        
        # Setup directories
        self._setup_directories()
        
    def _setup_device(self):
        """Setup computing device"""
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{self.args.gpu}' if self.args.gpu is not None else 'cuda')
            print(f"🖥️  Using GPU: {torch.cuda.get_device_name(device)}")
            print(f"   Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
            torch.cuda.empty_cache()
        else:
            device = torch.device('cpu')
            print("🖥️  Using CPU")
        
        return device
    
    def _create_run_name(self):
        """Create unique run name"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = "joint2d3d"
        hidden_dim = self.config['model']['hidden_dim']
        batch_size = self.config['data']['batch_size']
        lr = self.config['optimizer']['lr']
        
        return f"{model_name}_h{hidden_dim}_b{batch_size}_lr{lr}_{timestamp}"
    
    def _setup_directories(self):
        """Setup output directories"""
        self.output_dir = Path("training_outputs") / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "generated").mkdir(exist_ok=True)
        
        print(f"📁 Output directory: {self.output_dir}")
    
    def create_model_and_sde(self):
        """Create model and SDE"""
        print("🧠 Creating model...")
        
        # Create model
        self.model = Joint2D3DMolecularModel(
            atom_types=self.config['model']['atom_types'],
            bond_types=self.config['model']['bond_types'],
            hidden_dim=self.config['model']['hidden_dim'],
            pocket_dim=self.config['model']['pocket_dim'],
            num_layers=self.config['model']['num_layers'],
            max_radius=self.config['model']['max_radius']
        ).to(self.device)
        
        num_params = self.model.get_num_parameters()
        print(f"   Parameters: {num_params:,}")
        print(f"   Memory estimate: {num_params * 4 / 1024 / 1024:.1f} MB")
        
        # Create SDE
        print("🌊 Creating SDE...")
        self.sde = VESDE(
            sigma_min=self.config['sde']['sigma_min'],
            sigma_max=self.config['sde']['sigma_max'],
            N=self.config['sde']['num_steps']
        )
        
        # Save model architecture
        self._save_model_info()
    
    def _save_model_info(self):
        """Save model architecture information"""
        model_info = {
            'architecture': 'Joint2D3DMolecularModel',
            'parameters': self.model.get_num_parameters(),
            'config': self.config['model'],
            'sde_config': self.config['sde'],
            'device': str(self.device)
        }
        
        with open(self.output_dir / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
    
    def create_optimizers(self):
        """Create optimizer and scheduler"""
        print("⚙️  Creating optimizers...")
        
        # Optimizer
        if self.config['optimizer']['type'].lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['optimizer']['lr'],
                betas=self.config['optimizer']['betas'],
                weight_decay=self.config['optimizer']['weight_decay']
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['optimizer']['lr'],
                weight_decay=self.config['optimizer']['weight_decay']
            )
        
        # Scheduler
        if self.config['scheduler']['type'] == 'cosine_annealing':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['scheduler']['T_max'],
                eta_min=self.config['scheduler']['eta_min']
            )
        elif self.config['scheduler']['type'] == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config['scheduler'].get('gamma', 0.95)
            )
        else:
            self.scheduler = None
        
        print(f"   Optimizer: {type(self.optimizer).__name__}")
        print(f"   Scheduler: {type(self.scheduler).__name__ if self.scheduler else 'None'}")
    
    def create_data_loaders(self):
        """Create data loaders with validation"""
        print("📂 Creating data loaders...")
        
        try:
            # Create loaders
            self.train_loader = CrossDockDataLoader.create_train_loader(self.config)
            self.val_loader = CrossDockDataLoader.create_val_loader(self.config)
            
            print(f"   Train batches: {len(self.train_loader)}")
            print(f"   Val batches: {len(self.val_loader)}")
            
            # Validate first batch
            print("   Validating data loading...")
            train_batch = next(iter(self.train_loader))
            val_batch = next(iter(self.val_loader))
            
            self._validate_batch(train_batch, "train")
            self._validate_batch(val_batch, "val")
            
            return True
            
        except Exception as e:
            print(f"❌ Data loader error: {e}")
            return False
    
    def _validate_batch(self, batch, split_name: str):
        """Validate batch structure"""
        if batch is None:
            print(f"   ❌ {split_name} batch is None")
            return
        
        batch = batch.to(self.device)
        print(f"   ✅ {split_name} batch:")
        print(f"      Molecules: {batch.batch.max().item() + 1}")
        print(f"      Atoms: {batch.x.shape[0]}")
        print(f"      Bonds: {batch.edge_index.shape[1]}")
        
        if hasattr(batch, 'pocket_x'):
            print(f"      Pocket atoms: {batch.pocket_x.shape[0]}")
    
    def create_callbacks(self):
        """Create training callbacks"""
        callbacks = []
        
        # Early stopping
        if self.config.get('early_stopping'):
            callbacks.append(EarlyStopping(
                monitor=self.config['early_stopping']['monitor'],
                patience=self.config['early_stopping']['patience'],
                min_delta=self.config['early_stopping']['min_delta']
            ))
        
        # Model checkpoint
        callbacks.append(ModelCheckpoint(
            save_path=str(self.output_dir / "checkpoints"),
            monitor='val_loss',
            save_best_only=True
        ))
        
        # WandB logger
        if self.args.wandb:
            callbacks.append(WandBLogger(
                project_name=self.config['logging']['project_name'],
                log_frequency=self.config['logging'].get('log_every_n_steps', 50)
            ))
        
        # Molecular visualization
        if self.args.wandb and not self.args.debug:
            callbacks.append(MolecularVisualizationCallback(
                visualization_frequency=10,
                num_samples=4
            ))
        
        return callbacks
    
    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        if self.args.wandb:
            wandb.init(
                project=self.config['logging']['project_name'],
                name=self.run_name,
                config=self.config,
                tags=['crossdock', 'sde', 'molecular-generation'],
                dir=str(self.output_dir / "logs")
            )
            print("📊 WandB logging enabled")
        else:
            os.environ['WANDB_MODE'] = 'disabled'
            print("📊 WandB logging disabled")
    
    def create_trainer(self):
        """Create the main trainer"""
        callbacks = self.create_callbacks()
        
        self.trainer = SDEMolecularTrainer(
            model=self.model,
            sde=self.sde,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
            log_wandb=self.args.wandb,
            callbacks=callbacks
        )
    
    def validate_training_setup(self):
        """Validate training setup với test forward pass"""
        print("🔧 Validating training setup...")
        
        try:
            # Get a test batch
            test_batch = next(iter(self.train_loader))
            test_batch = test_batch.to(self.device)
            
            # Test forward pass
            self.model.train()
            
            # Sample random times
            batch_size = test_batch.batch.max().item() + 1
            t = torch.rand(batch_size, device=self.device)
            t_expanded = t[test_batch.batch]
            
            # Add noise according to SDE
            mean, std = self.sde.marginal_prob(test_batch.pos, t_expanded)
            noise = torch.randn_like(test_batch.pos)
            perturbed_pos = mean + std[:, None] * noise
            
            # Forward pass
            outputs = self.model(
                x=test_batch.x,
                pos=perturbed_pos,
                edge_index=test_batch.edge_index,
                edge_attr=test_batch.edge_attr,
                batch=test_batch.batch,
                pocket_x=getattr(test_batch, 'pocket_x', None),
                pocket_pos=getattr(test_batch, 'pocket_pos', None),
                pocket_edge_index=getattr(test_batch, 'pocket_edge_index', None),
                pocket_batch=getattr(test_batch, 'pocket_batch', None)
            )
            
            # Test loss computation
            score_target = -noise / std[:, None]
            score_pred = outputs['pos_pred']
            loss = F.mse_loss(score_pred, score_target)
            
            # Test backward pass
            loss.backward()
            self.optimizer.zero_grad()
            
            print("   ✅ Forward pass successful")
            print("   ✅ Loss computation successful")
            print("   ✅ Backward pass successful")
            print(f"   Test loss: {loss.item():.6f}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_training(self):
        """Run the main training loop"""
        print("\n🚀 Starting training...")
        print(f"   Epochs: {self.config['training']['num_epochs']}")
        print(f"   Batch size: {self.config['data']['batch_size']}")
        print(f"   Learning rate: {self.config['optimizer']['lr']}")
        print(f"   Device: {self.device}")
        
        # Save config
        with open(self.output_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # Start training
        try:
            self.trainer.train(
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                num_epochs=self.config['training']['num_epochs'],
                save_path=str(self.output_dir / "checkpoints" / "best_model.pth")
            )
            
            print("\n🎉 Training completed successfully!")
            
            # Save final metrics
            self._save_training_summary()
            
            return True
            
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            self._save_training_summary()
            return False
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_training_summary(self):
        """Save training summary"""
        training_time = time.time() - self.start_time
        
        summary = {
            'run_name': self.run_name,
            'training_time_hours': training_time / 3600,
            'best_val_loss': self.trainer.best_val_loss if self.trainer else float('inf'),
            'total_epochs': self.trainer.current_epoch if self.trainer else 0,
            'total_steps': self.trainer.global_step if self.trainer else 0,
            'config': self.config,
            'device': str(self.device),
            'model_parameters': self.model.get_num_parameters() if self.model else 0
        }
        
        with open(self.output_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Training Summary:")
        print(f"   Time: {training_time / 3600:.2f} hours")
        print(f"   Best val loss: {summary['best_val_loss']:.6f}")
        print(f"   Epochs: {summary['total_epochs']}")
        print(f"   Steps: {summary['total_steps']}")
    
    def test_generation(self):
        """Test molecular generation"""
        if not self.model:
            return
        
        print("\n🧪 Testing molecular generation...")
        
        try:
            from inference.conditional_generator import ConditionalMolecularGenerator
            
            generator = ConditionalMolecularGenerator(
                model=self.model,
                sde=self.sde,
                device=self.device
            )
            
            # Generate sample molecules
            samples = generator.generate_molecules(
                pocket_data={},
                num_molecules=5,
                max_atoms=30
            )
            
            smiles_list = generator.molecules_to_smiles(samples['molecules'])
            valid_smiles = [s for s in smiles_list if s is not None]
            
            print(f"   Generated {len(valid_smiles)}/{len(smiles_list)} valid molecules")
            
            if valid_smiles:
                print("   Sample molecules:")
                for i, smiles in enumerate(valid_smiles[:3]):
                    print(f"      {i+1}. {smiles}")
                
                # Compute metrics
                validity = MolecularMetrics.compute_validity(valid_smiles)
                uniqueness = MolecularMetrics.compute_uniqueness(valid_smiles)
                
                print(f"   Validity: {validity:.3f}")
                print(f"   Uniqueness: {uniqueness:.3f}")
                
                # Save generated molecules
                with open(self.output_dir / "generated" / "sample_molecules.txt", 'w') as f:
                    for i, smiles in enumerate(smiles_list):
                        if smiles:
                            f.write(f"{i}\t{smiles}\n")
            
        except Exception as e:
            print(f"   ⚠️  Generation test failed: {e}")
    
    def run_complete_pipeline(self):
        """Run complete training pipeline"""
        print("🔬 Molecular SDE Generator - Comprehensive Training")
        print("=" * 60)
        
        steps = [
            ("Setup WandB", self.setup_wandb),
            ("Create Model & SDE", self.create_model_and_sde),
            ("Create Optimizers", self.create_optimizers),
            ("Create Data Loaders", self.create_data_loaders),
            ("Create Trainer", self.create_trainer),
            ("Validate Setup", self.validate_training_setup),
            ("Run Training", self.run_training),
            ("Test Generation", self.test_generation)
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'='*20} {step_name} {'='*20}")
            try:
                result = step_func()
                if result is False:
                    print(f"❌ {step_name} failed, stopping pipeline")
                    break
                print(f"✅ {step_name} completed")
            except Exception as e:
                print(f"❌ {step_name} failed: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Cleanup
        if self.args.wandb:
            wandb.finish()
        
        print(f"\n📁 All outputs saved to: {self.output_dir}")

def load_config(config_path: str, args) -> Dict[str, Any]:
    """Load and modify config"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command line arguments
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.lr:
        config['optimizer']['lr'] = args.lr
        config['training']['lr'] = args.lr
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    
    # Quick test mode
    if args.quick_test:
        config['training']['num_epochs'] = 2
        config['data']['batch_size'] = min(config['data']['batch_size'], 4)
        config['model']['hidden_dim'] = 64
        config['model']['num_layers'] = 2
        config['sde']['num_steps'] = 100
    
    # Debug mode
    if args.debug:
        config['data']['batch_size'] = 2
        config['data']['num_workers'] = 1
        config['model']['hidden_dim'] = 32
        config['training']['num_epochs'] = 1
    
    return config

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Molecular SDE Training')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to config file')
    parser.add_argument('--wandb', action='store_true', default=False,
                       help='Enable wandb logging')
    parser.add_argument('--debug', action='store_true', default=False,
                       help='Debug mode with minimal settings')
    parser.add_argument('--quick_test', action='store_true', default=False,
                       help='Quick test with 2 epochs')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device ID')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config, args)
    
    # Create and run trainer
    trainer = ComprehensiveTrainer(config, args)
    trainer.run_complete_pipeline()

if __name__ == "__main__":
    main()