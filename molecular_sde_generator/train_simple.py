# train_simple.py - Simple training script for CrossDock
import torch
import torch.optim as optim
import yaml
import argparse
from pathlib import Path
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.models.joint_2d_3d_model import Joint2D3DMolecularModel
from src.models.sde_diffusion import VESDE
from src.training.sde_trainer import SDEMolecularTrainer
from src.data.data_loaders import CrossDockDataLoader
from src.training.callbacks_fixed import EarlyStopping, ModelCheckpoint

def main():
    parser = argparse.ArgumentParser(description='Simple CrossDock Training')
    parser.add_argument('--config', type=str, default='config/simple_config.yaml')
    parser.add_argument('--test', action='store_true', help='Quick test mode')
    parser.add_argument('--gpu', type=int, default=None, help='GPU device ID')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Test mode
    if args.test:
        config_path = 'config/test_config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("🧪 Test mode enabled with test config")
    
    # Override with command line args
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    
    # Setup device
    if torch.cuda.is_available() and args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"🖥️  Using device: {device}")
    
    # Create run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"crossdock_simple_{timestamp}"
    
    print(f"🏃 Run name: {run_name}")
    
    # Create output directory
    output_dir = Path("outputs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    try:
        # Check if processed data exists
        required_files = [
            config['data']['train_path'],
            config['data']['val_path']
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            print(f"❌ Required files not found: {missing_files}")
            print("Please run preprocessing first:")
            print("  python preprocess_crossdock_data.py")
            
            # Check if we need to create validation split
            split_file = Path("data/split_by_name.pt")
            if split_file.exists():
                splits = torch.load(split_file)
                if 'val' not in splits:
                    print("\n⚠️  No validation split found, creating one...")
                    print("Run preprocessing to create validation split:")
                    print("  python preprocess_crossdock_data.py --max_samples 1000")
            return
        
        # Create data loaders
        print("📂 Creating data loaders...")
        train_loader = CrossDockDataLoader.create_train_loader(config)
        val_loader = CrossDockDataLoader.create_val_loader(config)
        
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        
        # Test data loading
        print("   Testing data loading...")
        test_batch = next(iter(train_loader))
        if test_batch is None:
            print("❌ Failed to load test batch")
            return
        
        test_batch = test_batch.to(device)
        print(f"   ✅ Test batch: {test_batch.x.shape[0]} atoms, {test_batch.batch.max().item() + 1} molecules")
        
        # Create model
        print("🧠 Creating model...")
        model = Joint2D3DMolecularModel(
            atom_types=config['model']['atom_types'],
            bond_types=config['model']['bond_types'],
            hidden_dim=config['model']['hidden_dim'],
            pocket_dim=config['model']['pocket_dim'],
            num_layers=config['model']['num_layers'],
            max_radius=config['model']['max_radius']
        ).to(device)
        
        print(f"   Parameters: {model.get_num_parameters():,}")
        
        # Create SDE
        print("🌊 Creating SDE...")
        sde = VESDE(
            sigma_min=config['sde']['sigma_min'],
            sigma_max=config['sde']['sigma_max'],
            N=config['sde']['num_steps']
        )
        
        # Create optimizer
        print("⚙️  Creating optimizer...")
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['optimizer']['lr'],
            betas=config['optimizer']['betas'],
            weight_decay=config['optimizer']['weight_decay']
        )
        
        # Create scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['scheduler']['T_max'],
            eta_min=config['scheduler']['eta_min']
        )
        
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor=config['early_stopping']['monitor'],
                patience=config['early_stopping']['patience'],
                min_delta=config['early_stopping']['min_delta']
            ),
            ModelCheckpoint(
                save_path=str(output_dir / "checkpoints"),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Create trainer
        print("🏋️  Creating trainer...")
        trainer = SDEMolecularTrainer(
            model=model,
            sde=sde,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            log_wandb=False,  # Disable wandb for simplicity
            callbacks=callbacks
        )
        
        # Test forward pass
        print("🔧 Testing forward pass...")
        model.eval()
        with torch.no_grad():
            batch_size = test_batch.batch.max().item() + 1
            t = torch.rand(batch_size, device=device)
            t_expanded = t[test_batch.batch]
            
            mean, std = sde.marginal_prob(test_batch.pos, t_expanded)
            noise = torch.randn_like(test_batch.pos)
            perturbed_pos = mean + std[:, None] * noise
            
            outputs = model(
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
            
            print(f"   ✅ Forward pass successful")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"      {key}: {value.shape}")
        
        # Start training
        print("\n🚀 Starting training...")
        print(f"   Epochs: {config['training']['num_epochs']}")
        print(f"   Batch size: {config['data']['batch_size']}")
        print(f"   Learning rate: {config['optimizer']['lr']}")
        
        save_path = output_dir / "best_model.pth"
        
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs'],
            save_path=str(save_path)
        )
        
        print(f"\n🎉 Training completed!")
        print(f"   Model saved to: {save_path}")
        print(f"   Output directory: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n🔧 Troubleshooting:")
        print("   1. Check if data is preprocessed:")
        print("      python preprocess_crossdock_data.py --max_samples 1000")
        print("   2. Check data structure:")
        print("      python check_data_structure.py")
        print("   3. Try test mode:")
        print("      python train_simple.py --test")

if __name__ == "__main__":
    main()