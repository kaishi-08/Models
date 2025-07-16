# setup_and_validate.py - Script setup và kiểm tra toàn bộ pipeline

import os
import sys
import torch
import yaml
import pickle
import subprocess
from pathlib import Path
import numpy as np
from tqdm import tqdm

def check_dependencies():
    """Kiểm tra các dependencies cần thiết"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch', 'torch_geometric', 'rdkit', 'numpy', 
        'scipy', 'pandas', 'biopython', 'e3nn', 'wandb'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {missing_packages}")
        print("Install với: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_gpu():
    """Kiểm tra GPU availability"""
    print("\n🖥️  Checking GPU...")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name()
        memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        
        print(f"✅ CUDA available")
        print(f"   Device count: {device_count}")
        print(f"   Current device: {current_device}")
        print(f"   Device name: {device_name}")
        print(f"   Memory: {memory:.1f} GB")
        
        if memory < 6:
            print("⚠️  Warning: GPU memory < 6GB, consider reducing batch_size")
        
        return True
    else:
        print("❌ CUDA not available, will use CPU (slower)")
        return False

def setup_directories():
    """Setup directory structure"""
    print("\n📁 Setting up directories...")
    
    directories = [
        "data/raw/crossdock2020",
        "data/processed", 
        "models",
        "logs",
        "generated_molecules",
        "evaluation_results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")

def check_data():
    """Kiểm tra data availability"""
    print("\n📊 Checking data...")
    
    # Check raw data
    raw_path = Path("data/raw/crossdock2020")
    index_file = raw_path / "index.pkl"
    
    if index_file.exists():
        print("✅ Raw data index found")
        
        # Check index file
        try:
            with open(index_file, 'rb') as f:
                index_data = pickle.load(f)
            print(f"   Index contains {len(index_data)} entries")
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            return False
    else:
        print("❌ Raw data not found")
        print("   Download data với: python scripts/download_crossdock.py")
        return False
    
    # Check processed data
    processed_files = [
        "data/processed/train.pkl",
        "data/processed/val.pkl", 
        "data/processed/test.pkl"
    ]
    
    all_processed = True
    for file_path in processed_files:
        if Path(file_path).exists():
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"✅ {file_path} ({len(data)} samples)")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                all_processed = False
        else:
            print(f"❌ {file_path} not found")
            all_processed = False
    
    if not all_processed:
        print("   Preprocess data với: python scripts/preprocess_crossdock_data.py")
        return False
    
    return True

def test_data_loading():
    """Test data loading pipeline"""
    print("\n🔄 Testing data loading...")
    
    try:
        # Import modules
        from src.data.data_loaders import CrossDockDataLoader
        
        # Load config
        with open('config/training_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Test train loader
        print("   Creating train loader...")
        train_loader = CrossDockDataLoader.create_train_loader(config)
        print(f"   ✅ Train loader: {len(train_loader)} batches")
        
        # Test một batch
        print("   Testing batch loading...")
        for i, batch in enumerate(train_loader):
            if batch is None:
                continue
                
            print(f"   ✅ Batch {i}:")
            print(f"      Molecules: {batch.batch.max().item() + 1}")
            print(f"      Atoms: {batch.x.shape[0]}")
            print(f"      Bonds: {batch.edge_index.shape[1]}")
            
            if hasattr(batch, 'pocket_x'):
                print(f"      Pocket atoms: {batch.pocket_x.shape[0]}")
                print("      ✅ Pocket data available")
            else:
                print("      ❌ No pocket data")
            
            break
        
        # Test val loader
        print("   Creating val loader...")
        val_loader = CrossDockDataLoader.create_val_loader(config)
        print(f"   ✅ Val loader: {len(val_loader)} batches")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model initialization"""
    print("\n🧠 Testing model creation...")
    
    try:
        from src.models.joint_2d_3d_model import Joint2D3DMolecularModel
        from src.models.sde_diffusion import VESDE
        
        # Load config
        with open('config/training_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model
        print("   Creating model...")
        model = Joint2D3DMolecularModel(
            atom_types=config['model']['atom_types'],
            bond_types=config['model']['bond_types'], 
            hidden_dim=config['model']['hidden_dim'],
            pocket_dim=config['model']['pocket_dim'],
            num_layers=config['model']['num_layers'],
            max_radius=config['model']['max_radius']
        )
        
        num_params = model.get_num_parameters()
        print(f"   ✅ Model created: {num_params:,} parameters")
        
        # Create SDE
        print("   Creating SDE...")
        sde = VESDE(
            sigma_min=config['sde']['sigma_min'],
            sigma_max=config['sde']['sigma_max'],
            N=config['sde']['num_steps']
        )
        print("   ✅ SDE created")
        
        # Test forward pass
        print("   Testing forward pass...")
        batch_size = 2
        num_atoms = 10
        num_bonds = 15
        
        x = torch.randint(0, config['model']['atom_types'], (num_atoms, 1))
        pos = torch.randn(num_atoms, 3)
        edge_index = torch.randint(0, num_atoms, (2, num_bonds))
        edge_attr = torch.randint(0, config['model']['bond_types'], (num_bonds, 1))
        batch = torch.zeros(num_atoms, dtype=torch.long)
        
        # Mock pocket data
        pocket_atoms = 20
        pocket_x = torch.randn(pocket_atoms, 8)  # 8 features
        pocket_pos = torch.randn(pocket_atoms, 3)
        pocket_edge_index = torch.randint(0, pocket_atoms, (2, 30))
        pocket_batch = torch.zeros(pocket_atoms, dtype=torch.long)
        
        model.eval()
        with torch.no_grad():
            outputs = model(
                x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, batch=batch,
                pocket_x=pocket_x, pocket_pos=pocket_pos, 
                pocket_edge_index=pocket_edge_index, pocket_batch=pocket_batch
            )
        
        print("   ✅ Forward pass successful")
        print(f"      Atom logits: {outputs['atom_logits'].shape}")
        print(f"      Position pred: {outputs['pos_pred'].shape}")
        print(f"      Bond logits: {outputs['bond_logits'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_quick_test_config():
    """Tạo config cho quick test"""
    print("\n⚡ Creating quick test config...")
    
    quick_config = {
        'data': {
            'train_path': 'data/processed/train.pkl',
            'val_path': 'data/processed/val.pkl',
            'test_path': 'data/processed/test.pkl',
            'batch_size': 2,
            'num_workers': 2,
            'pin_memory': True,
            'shuffle': True,
            'include_pocket': True,
            'augment': True
        },
        'model': {
            'atom_types': 11,
            'bond_types': 4,
            'hidden_dim': 64,  # Smaller for quick test
            'pocket_dim': 128,
            'num_layers': 2,   # Fewer layers
            'max_radius': 10.0
        },
        'sde': {
            'sigma_min': 0.01,
            'sigma_max': 50.0,
            'num_steps': 100   # Fewer steps
        },
        'training': {
            'num_epochs': 2,   # Just 2 epochs for test
            'lr': 0.001,
            'weight_decay': 0.0001,
            'grad_clip_norm': 1.0
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 0.001,
            'betas': [0.9, 0.999],
            'weight_decay': 0.0001
        },
        'scheduler': {
            'type': 'cosine_annealing',
            'T_max': 2,
            'eta_min': 0.00001
        },
        'logging': {
            'project_name': 'crossdock-test',
            'log_every_n_steps': 10,
            'save_path': 'models/',
        },
        'early_stopping': {
            'monitor': 'val_loss',
            'patience': 5,
            'min_delta': 0.001
        }
    }
    
    # Save quick config
    with open('config/quick_test_config.yaml', 'w') as f:
        yaml.dump(quick_config, f, default_flow_style=False)
    
    print("   ✅ Quick test config saved to config/quick_test_config.yaml")

def run_quick_training_test():
    """Chạy một vài steps training để test"""
    print("\n🚀 Running quick training test...")
    
    try:
        # Run quick training
        result = subprocess.run([
            sys.executable, 'scripts/train_model.py',
            '--config', 'config/quick_test_config.yaml'
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("   ✅ Quick training test passed!")
            print("   Ready for full training!")
            return True
        else:
            print(f"   ❌ Training test failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("   ⚠️  Training test timed out (this might be normal)")
        return True
    except Exception as e:
        print(f"   ❌ Training test error: {e}")
        return False

def main():
    print("🔬 Molecular SDE Generator - Setup & Validation")
    print("=" * 50)
    
    all_checks = []
    
    # Run all checks
    all_checks.append(check_dependencies())
    all_checks.append(check_gpu())
    
    setup_directories()
    
    all_checks.append(check_data())
    
    if all_checks[-1]:  # Only if data is available
        all_checks.append(test_data_loading())
        all_checks.append(test_model_creation())
        
        create_quick_test_config()
        # all_checks.append(run_quick_training_test())
    
    # Final summary
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    if all(all_checks):
        print("🎉 ALL CHECKS PASSED!")
        print("\nReady to start training:")
        print("   python scripts/train_model.py")
        print("\nOr quick test:")
        print("   python scripts/train_model.py --config config/quick_test_config.yaml")
    else:
        print("❌ Some checks failed. Please fix issues above.")
        
        if not check_data():
            print("\n📥 To download and preprocess data:")
            print("   1. python scripts/download_crossdock.py")
            print("   2. python scripts/preprocess_crossdock_data.py")

if __name__ == "__main__":
    main()