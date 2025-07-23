# test_fixed_training.py - Fixed version with corrected variable names
import torch
import sys
from pathlib import Path

# Setup paths
project_root = Path.cwd()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_imports():
    """Test all imports first"""
    print("🔍 Testing imports...")
    
    try:
        from src.models.ddpm_diffusion import MolecularDDPM, MolecularDDPMModel
        print("   ✅ DDPM models imported")
    except Exception as e:
        print(f"   ❌ DDPM import failed: {e}")
        return False
    
    try:
        from src.models.joint_2d_3d_model import Joint2D3DMolecularModel
        print("   ✅ Joint2D3D model imported")
    except Exception as e:
        print(f"   ❌ Joint2D3D import failed: {e}")
        return False
    
    try:
        from src.data.data_loaders import CrossDockDataLoader
        print("   ✅ Data loader imported")
    except Exception as e:
        print(f"   ❌ Data loader import failed: {e}")
        return False
    
    try:
        from src.training.ddpm_trainer import DDPMMolecularTrainer
        print("   ✅ Trainer imported")
    except Exception as e:
        print(f"   ❌ Trainer import failed: {e}")
        return False
    
    return True

def test_fixed_data_loader():
    """Test fixed data loader with pocket_batch"""
    print("\n🧪 Testing fixed data loader...")
    
    try:
        from src.data.data_loaders import CrossDockDataLoader
        
        config = {
            'data': {
                'train_path': 'data/processed/train.pkl',
                'batch_size': 2,  # Small batch for testing
                'num_workers': 1,
                'shuffle': False,
                'pin_memory': False
            },
            'include_pocket': True,
            'max_atoms': 30,
            'augment': False
        }
        
        print("   Creating data loader...")
        train_loader = CrossDockDataLoader.create_train_loader(config)
        
        # Test first batch
        batch = next(iter(train_loader))
        
        if batch is not None:
            print("   ✅ Batch loaded successfully")
            
            # List key components
            key_components = ['x', 'pos', 'edge_index', 'batch', 'pocket_x', 'pocket_pos', 'pocket_batch']
            for comp in key_components:
                if hasattr(batch, comp):
                    value = getattr(batch, comp)
                    if value is not None:
                        print(f"      {comp}: {value.shape} on {value.device}")
                    else:
                        print(f"      {comp}: None")
                else:
                    print(f"      {comp}: Missing")
            
            # Check pocket_batch specifically
            if hasattr(batch, 'pocket_batch') and batch.pocket_batch is not None:
                print(f"   ✅ pocket_batch present: {batch.pocket_batch.shape}")
                print(f"   ✅ pocket_batch range: {batch.pocket_batch.min()}-{batch.pocket_batch.max()}")
                
                # Verify pocket_batch consistency
                mol_batch_max = batch.batch.max().item()
                pocket_batch_max = batch.pocket_batch.max().item()
                
                if pocket_batch_max <= mol_batch_max:
                    print("   ✅ pocket_batch indices are valid")
                    return True
                else:
                    print(f"   ❌ pocket_batch indices out of range: {pocket_batch_max} > {mol_batch_max}")
                    return False
                
            else:
                print("   ❌ pocket_batch is missing or None")
                print("   💡 Data loader needs to be updated with fixed collate function")
                return False
            
        else:
            print("   ❌ Batch is None")
            return False
            
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    """Test one training step with fixes"""
    print("\n🏋️ Testing training step...")
    
    try:
        from src.models.joint_2d_3d_model import Joint2D3DMolecularModel
        from src.models.ddpm_diffusion import MolecularDDPM, MolecularDDPMModel
        from src.training.ddpm_trainer import DDPMMolecularTrainer
        from src.data.data_loaders import CrossDockDataLoader
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Using device: {device}")
        
        # Create small model for testing
        base_model = Joint2D3DMolecularModel(
            atom_types=11, bond_types=4, hidden_dim=64, pocket_dim=64,
            num_layers=2, max_radius=10.0, max_pocket_atoms=200,
            conditioning_type="add"
        ).to(device)
        
        ddpm = MolecularDDPM(num_timesteps=100, beta_schedule="cosine")
        model = MolecularDDPMModel(base_model, ddpm).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        trainer = DDPMMolecularTrainer(
            base_model=model.base_model,
            ddpm=ddpm,
            optimizer=optimizer,
            device=device
        )
        
        print("   ✅ Trainer created")
        
        # Create data loader
        config = {
            'data': {
                'train_path': 'data/processed/train.pkl',
                'batch_size': 2,
                'num_workers': 1,
                'shuffle': False,
                'pin_memory': False
            },
            'include_pocket': True,
            'max_atoms': 30,
            'augment': False
        }
        
        train_loader = CrossDockDataLoader.create_train_loader(config)
        batch = next(iter(train_loader))
        
        if batch is None:
            print("   ❌ Could not load batch")
            return False
        
        print("   📦 Batch loaded for training test")
        print(f"      Molecules: {batch.batch.max().item() + 1}")
        print(f"      Ligand atoms: {batch.x.shape[0]}")
        if hasattr(batch, 'pocket_x') and batch.pocket_x is not None:
            print(f"      Pocket atoms: {batch.pocket_x.shape[0]}")
            if hasattr(batch, 'pocket_batch') and batch.pocket_batch is not None:
                print(f"      Pocket batch range: {batch.pocket_batch.min()}-{batch.pocket_batch.max()}")
            else:
                print("      ⚠️  Pocket batch missing")
        
        # Move to device
        batch = batch.to(device)
        
        # Test training step
        model.train()
        optimizer.zero_grad()
        
        # FIXED: Prepare kwargs with correct variable name
        ddpm_kwargs = {  # Fixed variable name from ddmp_kwargs to ddpm_kwargs
            'atom_features': batch.x,  # Avoid 'x' conflict
            'edge_index': batch.edge_index,
            'edge_attr': batch.edge_attr,
            'batch': batch.batch
        }
        
        # Add pocket data if available
        if hasattr(batch, 'pocket_x') and batch.pocket_x is not None:
            ddpm_kwargs['pocket_x'] = batch.pocket_x
            ddpm_kwargs['pocket_pos'] = batch.pocket_pos
            ddpm_kwargs['pocket_edge_index'] = batch.pocket_edge_index
            if hasattr(batch, 'pocket_batch') and batch.pocket_batch is not None:
                ddpm_kwargs['pocket_batch'] = batch.pocket_batch
        
        # Forward pass
        loss, loss_dict = ddpm.compute_loss(
            model=model,
            x0=batch.pos,
            **ddpm_kwargs
        )
        
        print(f"   ✅ Forward pass successful! Loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print("   ✅ Backward pass successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 Testing Fixed Training Components")
    print("=" * 50)
    
    # Check if required files exist
    print("📂 Checking required files...")
    required_files = [
        'src/models/ddpm_diffusion.py',
        'src/models/joint_2d_3d_model.py',
        'src/data/data_loaders.py',
        'data/processed/train.pkl'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"   ❌ Missing: {file_path}")
        else:
            print(f"   ✅ Found: {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        print("\n💡 Files to create:")
        print("   1. Copy 'Complete DDPM Diffusion File' → src/models/ddpm_diffusion.py")
        print("   2. Copy 'Complete Data Loaders with Pocket Batch Fix' → src/data/data_loaders.py")
        return
    
    # Test imports
    import_ok = test_imports()
    if not import_ok:
        print("\n❌ Import test failed - check file contents")
        return
    
    # Test data loader fix
    data_ok = test_fixed_data_loader()
    
    # Test training step
    train_ok = test_training_step()
    
    print("\n📋 Test Summary:")
    print(f"   Imports: {'✅' if import_ok else '❌'}")
    print(f"   Data loader: {'✅' if data_ok else '❌'}")
    print(f"   Training step: {'✅' if train_ok else '❌'}")
    
    if import_ok and data_ok and train_ok:
        print("\n🎉 All tests passed! Ready for full training!")
        print("\n🚀 Next steps:")
        print("   1. python scripts/train_ddpm.py --test --batch_size 2")
        print("   2. If successful: python scripts/train_ddmp.py --test --batch_size 4")
        print("   3. Full training: python scripts/train_ddpm.py --epochs 10")
    else:
        print("\n❌ Some tests failed.")
        
        if not data_ok:
            print("   🔧 Fix: Update src/data/data_loaders.py with 'Complete Data Loaders' artifact")
        if not train_ok:
            print("   🔧 Fix: Check src/models/ddpm_diffusion.py implementation")
        
        print("\n📋 Action items:")
        print("   1. Replace src/models/ddpm_diffusion.py with artifact content")
        print("   2. Replace src/data/data_loaders.py with artifact content")
        print("   3. Run: python test_fixed_training.py")

if __name__ == "__main__":
    main()