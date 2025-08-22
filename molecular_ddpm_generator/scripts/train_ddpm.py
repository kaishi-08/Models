# scripts/train_ddpm.py - FIXED imports and training logic
import os
import sys
import yaml
import torch
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Union

# Fix path issues
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

print(f"🎯 Project root: {project_root}")
print(f"🎯 Python path: {sys.path[:3]}")

try:
    from src.models.joint_2d_3d_model import create_joint2d3d_model  # ✅ FIXED
    from src.models.ddpm_diffusion import MolecularDDPM, MolecularDDPMModel
    from src.data.data_loaders import CrossDockDataLoader
    from src.training.ddpm_trainer import DDPMMolecularTrainer
    from src.training.callbacks_fixed import WandBLogger, EarlyStopping, ModelCheckpoint
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install missing dependencies:")
    print("   pip install torch-geometric>=2.2.0")
    print("   pip install rdkit-pypi")
    print("   pip install biopython")
    sys.exit(1)

warnings.filterwarnings('ignore')

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_bool(value: Any, default: bool = False) -> bool:
    """Safely convert value to bool"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    try:
        return bool(value)
    except:
        return default

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and fix config types"""
    print("🔧 Validating configuration...")
    
    # Fix optimizer config
    if 'optimizer' in config:
        opt_config = config['optimizer']
        
        # Ensure numeric types
        opt_config['lr'] = safe_float(opt_config.get('lr', 0.0001))
        opt_config['weight_decay'] = safe_float(opt_config.get('weight_decay', 0.0001))
        opt_config['eps'] = safe_float(opt_config.get('eps', 1e-8))
        
        # Fix betas (should be list of floats)
        if 'betas' in opt_config:
            betas = opt_config['betas']
            if isinstance(betas, list):
                opt_config['betas'] = [safe_float(b) for b in betas]
            else:
                opt_config['betas'] = [0.9, 0.999]
    
    # Fix training config
    if 'training' in config:
        train_config = config['training']
        train_config['num_epochs'] = safe_int(train_config.get('num_epochs', 100))
        train_config['learning_rate'] = safe_float(train_config.get('learning_rate', 0.0001))
        train_config['weight_decay'] = safe_float(train_config.get('weight_decay', 0.0001))
        train_config['grad_clip_norm'] = safe_float(train_config.get('grad_clip_norm', 1.0))
    
    # Fix data config
    if 'data' in config:
        data_config = config['data']
        data_config['batch_size'] = safe_int(data_config.get('batch_size', 16))
        data_config['num_workers'] = safe_int(data_config.get('num_workers', 4))
        data_config['pin_memory'] = safe_bool(data_config.get('pin_memory', True))
        data_config['shuffle'] = safe_bool(data_config.get('shuffle', True))
    
    # Fix model config
    if 'model' in config:
        model_config = config['model']
        model_config['atom_types'] = safe_int(model_config.get('atom_types', 11))
        model_config['bond_types'] = safe_int(model_config.get('bond_types', 4))
        model_config['hidden_dim'] = safe_int(model_config.get('hidden_dim', 256))
        model_config['pocket_dim'] = safe_int(model_config.get('pocket_dim', 256))
        model_config['num_layers'] = safe_int(model_config.get('num_layers', 4))
        model_config['max_radius'] = safe_float(model_config.get('max_radius', 10.0))
        model_config['max_pocket_atoms'] = safe_int(model_config.get('max_pocket_atoms', 1000))
    
    # Fix DDPM config
    if 'ddpm' in config:
        ddpm_config = config['ddpm']
        ddpm_config['num_timesteps'] = safe_int(ddpm_config.get('num_timesteps', 1000))
        ddpm_config['beta_start'] = safe_float(ddpm_config.get('beta_start', 0.0001))
        ddpm_config['beta_end'] = safe_float(ddpm_config.get('beta_end', 0.02))
    
    print("✅ Configuration validated")
    return config

def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU (GPU not available)")
    
    return device

def check_data_files(config):
    """Check if preprocessed data files exist"""
    required_files = [
        config['data']['train_path'],
        config['data']['val_path']
    ]
    
    print("📁 Checking data files...")
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            print(f"❌ Missing: {full_path}")
            print("\n🔧 Run preprocessing first:")
            print("   python scripts/preprocess_crossdock_data.py")
            return False
        else:
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"✅ Found: {file_path} ({size_mb:.1f} MB)")
    
    return True

def create_model(config, device):
    print("🧠 Creating DDPM model...")
    
    try:
        # 🎯 FIXED: Use correct function name and parameters
        base_model = create_joint2d3d_model(
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            conditioning_type=config['model'].get('conditioning_type', 'add')
        ).to(device)
        
        # Create DDPM
        ddpm = MolecularDDPM(
            num_timesteps=config['ddpm']['num_timesteps'],
            beta_schedule=config['ddpm']['beta_schedule'],
            beta_start=config['ddpm']['beta_start'],
            beta_end=config['ddpm']['beta_end']
        )
        
        # Wrap with DDPM
        model = MolecularDDPMModel(base_model, ddpm).to(device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   🎯 Trainable parameters: {trainable_params:,}")
        print(f"   💾 Model size: {total_params * 4 / 1e6:.1f} MB")
        
        return model, ddpm
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def create_data_loaders(config):
    """Create train and validation data loaders"""
    print("📊 Creating data loaders...")
    
    try:
        # Create loaders with validated config
        train_loader = CrossDockDataLoader.create_train_loader(config)
        val_loader = CrossDockDataLoader.create_val_loader(config)
        
        # Test loading a batch
        print("🧪 Testing data loading...")
        try:
            test_batch = next(iter(train_loader))
            if test_batch is not None:
                print(f"   ✅ Sample batch: {test_batch.x.shape[0]} atoms, {test_batch.edge_index.shape[1]} bonds")
                if hasattr(test_batch, 'pocket_x') and test_batch.pocket_x is not None:
                    print(f"   🧬 Pocket data: {test_batch.pocket_x.shape[0]} pocket atoms")
                else:
                    print(f"   ⚠️  No pocket data in batch")
            else:
                print("   ⚠️  First batch is None")
        except Exception as e:
            print(f"   ❌ Error testing batch: {e}")
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def create_trainer(model, ddpm, config, device):
    print("🏃 Creating trainer...")
    
    try:
        # Create optimizer with validated parameters
        opt_config = config['optimizer']

        if opt_config['type'].lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=opt_config['lr'],
                betas=opt_config['betas'],
                weight_decay=opt_config['weight_decay'],
                eps=opt_config['eps']
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay'],
                eps=opt_config['eps']
            )
        
        print("   ✅ Optimizer created")
        
        # Create scheduler
        scheduler = None
        if 'scheduler' in config['training'] and config['training']['scheduler']:
            sched_config = config['training']['scheduler']
            if sched_config['type'] == 'cosine_annealing':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=safe_int(sched_config['T_max'], 100),
                    eta_min=safe_float(sched_config['eta_min'], 1e-5)
                )
                print("   📈 Cosine annealing scheduler created")
        
        # Create callbacks
        callbacks = []
        
        # Early stopping
        if 'early_stopping' in config:
            callbacks.append(EarlyStopping(
                monitor=config['early_stopping']['monitor'],
                patience=safe_int(config['early_stopping']['patience'], 15),
                min_delta=safe_float(config['early_stopping']['min_delta'], 0.001)
            ))
            print("   ⏹️  Early stopping enabled")
        
        # Model checkpointing
        if 'checkpointing' in config:
            save_path = project_root / config['logging']['save_path']
            save_path.mkdir(parents=True, exist_ok=True)
            callbacks.append(ModelCheckpoint(
                save_path=str(save_path),
                monitor=config['checkpointing']['monitor'],
                save_best_only=safe_bool(config['checkpointing']['save_best_only'], True)
            ))
            print(f"   💾 Model checkpointing: {save_path}")
        
        # Create trainer
        trainer = DDPMMolecularTrainer(
            base_model=model.base_model,
            ddpm=ddpm,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            callbacks=callbacks
        )
        
        print(f"   🎯 Trainer created with {len(callbacks)} callbacks")
        return trainer
        
    except Exception as e:
        print(f"❌ Error creating trainer: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='🚀 Train DDPM Molecular Generator')
    parser.add_argument('--config', type=str, default='config/optimized_ddpm_config.yaml',
                       help='Path to config file')
    parser.add_argument('--test', action='store_true', help='Quick test mode')
    parser.add_argument('--epochs', type=int, help='Override epochs')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    
    args = parser.parse_args()
    
    print("🎯 DDPM MOLECULAR GENERATOR TRAINING")
    print("=" * 60)
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configuration
    config_path = project_root / args.config
    print(f"📄 Loading config: {config_path}")
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print("Available configs:")
        config_dir = project_root / "config"
        if config_dir.exists():
            for f in config_dir.glob("*.yaml"):
                print(f"  📄 {f.name}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override for test mode
    if args.test:
        config['training']['num_epochs'] = 2
        config['data']['batch_size'] = 4
        config['logging']['use_wandb'] = False
        print("🧪 Test mode: 2 epochs, batch size 4")
    
    # Apply overrides
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
        print(f"🔧 Epochs override: {args.epochs}")
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
        print(f"🔧 Batch size override: {args.batch_size}")
    
    # Validate configuration
    config = validate_config(config)
    
    # Setup device
    device = setup_device()
    
    # Check data
    if not check_data_files(config):
        return
    
    # Create model
    model, ddpm = create_model(config, device)
    if model is None:
        return
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    if train_loader is None:
        return
    
    # Create trainer
    trainer = create_trainer(model, ddpm, config, device)
    if trainer is None:
        return
    
    # Start training
    print("\n🚀 STARTING TRAINING")
    print("=" * 60)
    
    try:
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs'],
            save_path=str(project_root / config['logging']['save_path'] / "best_model.pth")
        )
        
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💾 Model saved to: {project_root / config['logging']['save_path']}")
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()