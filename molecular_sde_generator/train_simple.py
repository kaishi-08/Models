# train_simple.py - With smart pocket selection
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

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description='Smart Pocket Selection Training')
    parser.add_argument('--config', type=str, default='config/smart_config.yaml')
    parser.add_argument('--test', action='store_true', help='Quick test mode')
    parser.add_argument('--gpu', type=int, default=None, help='GPU device ID')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--use_train_split', action='store_true', 
                       help='Split train data for validation (recommended)')
    
    # Smart pocket selection options
    parser.add_argument('--max_pocket_atoms', type=int, default=1000,
                       help='Maximum number of pocket atoms to process')
    parser.add_argument('--pocket_strategy', type=str, default='adaptive',
                       choices=['adaptive', 'distance', 'surface', 'residue', 'binding_site'],
                       help='Pocket atom selection strategy')
    parser.add_argument('--interaction_radius', type=float, default=8.0,
                       help='Interaction radius for binding_site strategy')
    parser.add_argument('--compare_strategies', action='store_true',
                       help='Compare different selection strategies')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Test mode
    if args.test:
        config['training']['num_epochs'] = 2
        config['data']['batch_size'] = 4
        config['model']['hidden_dim'] = 64
        config['model']['num_layers'] = 2
        config['sde']['num_steps'] = 100
        print("🧪 Test mode enabled")
    
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
    print(f"🧠 Pocket selection strategy: {args.pocket_strategy}")
    print(f"📊 Max pocket atoms: {args.max_pocket_atoms}")
    
    # Create run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"smart_pocket_{args.pocket_strategy}_{timestamp}"
    
    print(f"🏃 Run name: {run_name}")
    
    # Create output directory
    output_dir = Path("outputs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config with command line overrides
    config['model']['max_pocket_atoms'] = args.max_pocket_atoms
    config['model']['pocket_selection_strategy'] = args.pocket_strategy
    config['model']['interaction_radius'] = args.interaction_radius
    
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    try:
        # Check if processed data exists
        required_files = [
            config['data']['train_path'],
            config['data']['test_path']
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            print(f"❌ Required files not found: {missing_files}")
            print("Please run preprocessing first:")
            print("  python scripts/preprocess_crossdock_data.py --max_samples 1000")
            return
        
        # Create data loaders
        print("📂 Creating data loaders...")
        
        if args.use_train_split:
            train_loader, val_loader = CrossDockDataLoader.create_train_val_split_loader(
                config, val_ratio=0.1
            )
            print(f"   Train split: {len(train_loader)} batches")
            print(f"   Val split: {len(val_loader)} batches")
        else:
            train_loader = CrossDockDataLoader.create_train_loader(config)
            val_loader = CrossDockDataLoader.create_val_loader(config)
            print(f"   Train: {len(train_loader)} batches")
            print(f"   Val (test): {len(val_loader)} batches")
        
        # Test data loading and pocket selection
        print("   Testing data loading and pocket selection...")
        test_batch = next(iter(train_loader))
        if test_batch is None:
            print("❌ Failed to load test batch")
            return
        
        test_batch = test_batch.to(device)
        print(f"   ✅ Test batch: {test_batch.x.shape[0]} atoms, {test_batch.batch.max().item() + 1} molecules")
        
        # Check pocket data and analyze selection
        has_pocket = hasattr(test_batch, 'pocket_x') and test_batch.pocket_x is not None
        print(f"   Pocket data: {'✅ Available' if has_pocket else '❌ Not available'}")
        
        if has_pocket:
            original_pocket_size = test_batch.pocket_x.shape[0]
            print(f"      Original pocket atoms: {original_pocket_size}")
            
            if original_pocket_size > args.max_pocket_atoms:
                print(f"      Will apply {args.pocket_strategy} selection")
                print(f"      Reduction: {original_pocket_size} → {args.max_pocket_atoms} atoms")
                reduction_ratio = args.max_pocket_atoms / original_pocket_size
                print(f"      Keeping: {reduction_ratio:.1%} of atoms")
        
        # Compare strategies if requested
        if args.compare_strategies and has_pocket:
            print("\n🔍 Comparing selection strategies...")
            compare_pocket_strategies(test_batch, args.max_pocket_atoms)
        
        # Create model with smart pocket selection
        print("🧠 Creating model with smart pocket selection...")
        model = Joint2D3DMolecularModel(
            atom_types=config['model']['atom_types'],
            bond_types=config['model']['bond_types'],
            hidden_dim=config['model']['hidden_dim'],
            pocket_dim=config['model']['pocket_dim'],
            num_layers=config['model']['num_layers'],
            max_radius=config['model']['max_radius'],
            max_pocket_atoms=args.max_pocket_atoms,
            selection_strategy=args.pocket_strategy  # Pass strategy to model
        ).to(device)
        
        # Count parameters
        num_params = count_parameters(model)
        print(f"   Parameters: {num_params:,}")
        
        # Create SDE
        print("🌊 Creating SDE...")
        sde = VESDE(
            sigma_min=config['sde']['sigma_min'],
            sigma_max=config['sde']['sigma_max'],
            N=config['sde']['num_steps']
        )
        
        # Create optimizer
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
                monitor='val_total_loss',
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
            log_wandb=False,
            callbacks=callbacks
        )
        
        # Test forward pass
        print("🔧 Testing forward pass with smart selection...")
        test_forward_pass(model, sde, test_batch, device)
        
        # Start training
        print(f"\n🚀 Starting training with {args.pocket_strategy} pocket selection...")
        print(f"   Strategy: {args.pocket_strategy}")
        print(f"   Max pocket atoms: {args.max_pocket_atoms}")
        print(f"   Epochs: {config['training']['num_epochs']}")
        print(f"   Batch size: {config['data']['batch_size']}")
        
        save_path = output_dir / "best_model.pth"
        
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs'],
            save_path=str(save_path)
        )
        
        print(f"\n🎉 Training completed!")
        print(f"   Model saved to: {save_path}")
        print(f"   Strategy used: {args.pocket_strategy}")
        
        # Log selection statistics
        log_selection_statistics(output_dir, args, config)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

def compare_pocket_strategies(batch, max_atoms):
    """Compare different pocket selection strategies"""
    from src.models.pocket_encoder import SmartPocketAtomSelector
    
    pos = batch.pocket_pos
    x = batch.pocket_x
    
    strategies = ['distance', 'surface', 'residue']
    pocket_center = torch.mean(pos, dim=0)
    
    print(f"   Original pocket size: {len(pos)} atoms")
    
    for strategy in strategies:
        try:
            selector = SmartPocketAtomSelector()
            if strategy == 'distance':
                indices = selector.select_by_distance_to_center(pos, pocket_center, max_atoms)
            elif strategy == 'surface':
                indices = selector.select_by_surface_accessibility(pos, x, max_atoms)
            elif strategy == 'residue':
                indices = selector.select_by_residue_importance(pos, x, max_atoms, pocket_center)
            
            print(f"   {strategy:12}: {len(indices):4d} atoms selected")
            
            # Analyze selected atoms
            selected_pos = pos[indices]
            center_distances = torch.norm(selected_pos - pocket_center, dim=1)
            print(f"   {'':<12}  Distance to center: {center_distances.mean():.2f} ± {center_distances.std():.2f} Å")
            
        except Exception as e:
            print(f"   {strategy:12}: Error - {e}")

def test_forward_pass(model, sde, batch, device):
    """Test forward pass with smart pocket selection"""
    model.eval()
    with torch.no_grad():
        batch_size = batch.batch.max().item() + 1
        t = torch.rand(batch_size, device=device)
        t_expanded = t[batch.batch]
        
        mean, std = sde.marginal_prob(batch.pos, t_expanded)
        noise = torch.randn_like(batch.pos)
        perturbed_pos = mean + std[:, None] * noise
        
        try:
            outputs = model(
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
            
            print(f"   ✅ Forward pass successful with smart selection")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"      {key}: {value.shape}")
                    
        except Exception as e:
            print(f"   ❌ Forward pass failed: {e}")
            raise e

def log_selection_statistics(output_dir, args, config):
    """Log pocket selection statistics"""
    stats = {
        'pocket_selection': {
            'strategy': args.pocket_strategy,
            'max_atoms': args.max_pocket_atoms,
            'interaction_radius': args.interaction_radius,
        },
        'model_config': {
            'hidden_dim': config['model']['hidden_dim'],
            'pocket_dim': config['model']['pocket_dim'],
            'num_layers': config['model']['num_layers'],
        },
        'training_config': {
            'epochs': config['training']['num_epochs'],
            'batch_size': config['data']['batch_size'],
            'learning_rate': config['optimizer']['lr'],
        }
    }
    
    # Save statistics
    import json
    with open(output_dir / "selection_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"   Selection statistics saved to: {output_dir}/selection_stats.json")

if __name__ == "__main__":
    main()