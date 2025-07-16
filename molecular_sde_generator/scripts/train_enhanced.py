# scripts/train_enhanced.py - Enhanced training script với đầy đủ features

import torch
import torch.optim as optim
import yaml
import wandb
import argparse
from pathlib import Path
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append('src')

from models.joint_2d_3d_model import Joint2D3DMolecularModel
from models.sde_diffusion import VESDE
from training.sde_trainer import SDEMolecularTrainer
from data.data_loaders import CrossDockDataLoader
from training.callbacks import (
    WandBLogger, EarlyStopping, ModelCheckpoint, 
    MolecularVisualizationCallback
)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Molecular SDE Generator')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--wandb', action='store_true', default=False,
                       help='Enable wandb logging')
    parser.add_argument('--debug', action='store_true', default=False,
                       help='Debug mode with smaller dataset')
    parser.add_argument('--quick_test', action='store_true', default=False,
                       help='Quick test with 2 epochs')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device ID to use')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    
    return parser.parse_args()

def setup_device(gpu_id=None):
    """Setup computing device"""
    if torch.cuda.is_available():
        if gpu_id is not None:
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cuda')
        
        print(f"🖥️  Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"   Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        
        # Clear cache
        torch.cuda.empty_cache()
        
    else:
        device = torch.device('cpu')
        print("🖥️  Using CPU")
    
    return device

def load_config(config_path, args):
    """Load and modify config based on args"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line args
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
        print(f"📊 Override batch_size: {args.batch_size}")
    
    if args.lr:
        config['optimizer']['lr'] = args.lr
        config['training']['lr'] = args.lr
        print(f"📊 Override learning rate: {args.lr}")
    
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
        print(f"📊 Override epochs: {args.epochs}")
    
    if args.quick_test:
        config['training']['num_epochs'] = 2
        config['data']['batch_size'] = min(config['data']['batch_size'], 4)
        config['model']['hidden_dim'] = 64
        config['model']['num_layers'] = 2
        print("⚡ Quick test mode enabled")
    
    if args.debug:
        config['data']['batch_size'] = 2
        config['data']['num_workers'] = 1
        config['model']['hidden_dim'] = 32
        print("🐛 Debug mode enabled")
    
    return config

def create_run_name(config):
    """Create unique run name"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hidden_dim = config['model']['hidden_dim']
    batch_size = config['data']['batch_size']
    lr = config['optimizer']['lr']
    
    return f"crossdock_h{hidden_dim}_b{batch_size}_lr{lr}_{timestamp}"

def setup_logging(config, args, run_name):
    """Setup logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Setup wandb
    if args.wandb:
        wandb.init(
            project=config['logging']['project_name'],
            name=run_name,
            config=config,
            tags=['crossdock', 'sde', 'molecular-generation']
        )
        print("📊 Wandb logging enabled")
    else:
        os.environ['WANDB_MODE'] = 'disabled'
        print("📊 Wandb logging disabled")

def create_model_and_sde(config, device):
    """Create model and SDE"""
    print("🧠 Creating model...")
    
    model = Joint2D3DMolecularModel(
        atom_types=config['model']['atom_types'],
        bond_types=config['model']['bond_types'],
        hidden_dim=config['model']['hidden_dim'],
        pocket_dim=config['model']['pocket_dim'],
        num_layers=config['model']['num_layers'],
        max_radius=config['model']['max_radius']
    ).to(device)
    
    num_params = model.get_num_parameters()
    print(f"   Parameters: {num_params:,}")
    
    # Estimate memory usage
    param_size_mb = num_params * 4 / 1024 / 1024  # 4 bytes per float32
    print(f"   Est. memory: {param_size_mb:.1f} MB")
    
    print("🌊 Creating SDE...")
    sde = VESDE(
        sigma_min=config['sde']['sigma_min'],
        sigma_max=config['sde']['sigma_max'],
        N=config['sde']['num_steps']
    )
    
    return model, sde

def create_optimizer_and_scheduler(model, config):
    """Create optimizer and scheduler"""
    if config['optimizer']['type'].lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['optimizer']['lr'],
            betas=config['optimizer']['betas'],
            weight_decay=config['optimizer']['weight_decay']
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
    
    if config['scheduler']['type'] == 'cosine_annealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['scheduler']['T_max'],
            eta_min=config['scheduler']['eta_min']
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.5
        )
    
    return optimizer, scheduler

def create_callbacks(config, args):
    """Create training callbacks"""
    callbacks = []
    
    # Early stopping
    if config.get('early_stopping'):
        callbacks.append(EarlyStopping(
            monitor=config['early_stopping']['monitor'],
            patience=config['early_stopping']['patience'],
            min_delta=config['early_stopping']['min_delta']
        ))
    
    # Model checkpoint
    save_dir = Path(config['logging']['save_path'])
    save_dir.mkdir(parents=True, exist_ok=True)
    callbacks.append(ModelCheckpoint(
        save_path=str(save_dir),
        monitor='val_loss',
        save_best_only=True
    ))
    
    # Wandb logger
    if args.wandb:
        callbacks.append(WandBLogger(
            project_name=config['logging']['project_name'],
            log_frequency=config['logging'].get('log_every_n_steps', 50)
        ))
    
    # Molecular visualization (only if wandb enabled)
    if args.wandb and not args.debug:
        callbacks.append(MolecularVisualizationCallback(
            visualization_frequency=10,
            num_samples=4
        ))
    
    return callbacks

def create_data_loaders(config):
    """Create data loaders with error handling"""
    print("📂 Creating data loaders...")
    
    try:
        train_loader = CrossDockDataLoader.create_train_loader(config)
        val_loader = CrossDockDataLoader.create_val_loader(config)
        
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        
        # Test loading a batch
        print("   Testing batch loading...")
        for batch in train_loader:
            if batch is not None:
                print(f"   ✅ Sample batch: {batch.x.shape[0]} atoms, {batch.batch.max().item() + 1} molecules")
                if hasattr(batch, 'pocket_x'):
                    print(f"   ✅ Pocket data: {batch.pocket_x.shape[0]} atoms")
                break
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check if data is preprocessed: ls data/processed/")
        print("   2. Run preprocessing: python scripts/preprocess_crossdock_data.py")
        print("   3. Check config paths in config/training_config.yaml")
        raise

def main():
    print("🔬 Molecular SDE Generator Training")
    print("=" * 50)
    
    # Parse arguments
    args = parse_args()
    
    # Setup device
    device = setup_device(args.gpu)
    
    # Load config
    config = load_config(args.config, args)
    
    # Create run name
    run_name = create_run_name(config)
    print(f"🏃 Run name: {run_name}")
    
    # Setup logging
    setup_logging(config, args, run_name)
    
    try:
        # Create data loaders
        train_loader, val_loader = create_data_loaders(config)
        
        # Create model and SDE
        model, sde = create_model_and_sde(config, device)
        
        # Create optimizer and scheduler
        optimizer, scheduler = create_optimizer_and_scheduler(model, config)
        
        # Create callbacks
        callbacks = create_callbacks(config, args)
        
        # Create trainer
        print("🏋️  Creating trainer...")
        trainer = SDEMolecularTrainer(
            model=model,
            sde=sde,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            log_wandb=args.wandb,
            callbacks=callbacks
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            print(f"📂 Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Start training
        print("\n🚀 Starting training...")
        print(f"   Epochs: {config['training']['num_epochs']}")
        print(f"   Batch size: {config['data']['batch_size']}")
        print(f"   Learning rate: {config['optimizer']['lr']}")
        print(f"   Device: {device}")
        
        save_path = Path(config['logging']['save_path']) / f"{run_name}_best.pth"
        
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs'],
            save_path=str(save_path)
        )
        
        print("\n🎉 Training completed!")
        print(f"   Best model saved to: {save_path}")
        
        # Generate sample molecules
        if not args.debug:
            print("\n🧪 Generating sample molecules...")
            try:
                from inference.conditional_generator import ConditionalMolecularGenerator
                
                generator = ConditionalMolecularGenerator(
                    model=model,
                    sde=sde, 
                    device=device
                )
                
                samples = generator.generate_molecules(
                    pocket_data={},
                    num_molecules=10,
                    max_atoms=30
                )
                
                smiles_list = generator.molecules_to_smiles(samples['molecules'])
                valid_smiles = [s for s in smiles_list if s is not None]
                
                print(f"   Generated {len(valid_smiles)}/{len(smiles_list)} valid molecules")
                
                if valid_smiles:
                    print("   Sample molecules:")
                    for i, smiles in enumerate(valid_smiles[:3]):
                        print(f"      {i+1}. {smiles}")
                
            except Exception as e:
                print(f"   ⚠️  Could not generate samples: {e}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n🔧 Troubleshooting:")
        print("   1. Check GPU memory: nvidia-smi")
        print("   2. Reduce batch_size if OOM")
        print("   3. Check data preprocessing")
        print("   4. Run validation: python setup_and_validate.py")
        
        raise
    
    finally:
        if args.wandb:
            wandb.finish()

if __name__ == "__main__":
    main()
    