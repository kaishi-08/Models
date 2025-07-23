# check_device_consistency.py - Check device issues in data loading
import torch
import sys
from pathlib import Path

# Setup paths
project_root = Path.cwd()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def check_batch_devices():
    """Check device consistency in data batches"""
    print("🔍 Checking batch device consistency...")
    
    try:
        from src.data.data_loaders import CrossDockDataLoader
        
        # Create data loader
        config = {
            'data': {
                'train_path': 'data/processed/train.pkl',
                'batch_size': 4,
                'num_workers': 1,
                'shuffle': False,
                'pin_memory': False  # Try without pin_memory first
            },
            'include_pocket': True,
            'max_atoms': 30,
            'augment': False
        }
        
        print("📊 Creating data loader...")
        train_loader = CrossDockDataLoader.create_train_loader(config)
        
        # Check several batches
        print("🧪 Checking first 3 batches...")
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 3:
                break
                
            print(f"\n📦 Batch {batch_idx}:")
            
            if batch is None:
                print("   ❌ Batch is None")
                continue
            
            # Check all tensor attributes
            attrs_to_check = ['x', 'pos', 'edge_index', 'edge_attr', 'batch',
                             'pocket_x', 'pocket_pos', 'pocket_edge_index', 'pocket_batch']
            
            devices = {}
            shapes = {}
            
            for attr in attrs_to_check:
                if hasattr(batch, attr):
                    value = getattr(batch, attr)
                    if value is not None and isinstance(value, torch.Tensor):
                        devices[attr] = str(value.device)
                        shapes[attr] = tuple(value.shape)
                        print(f"   {attr}: {value.device} {value.shape} {value.dtype}")
                    else:
                        print(f"   {attr}: None or not tensor")
                else:
                    print(f"   {attr}: Missing")
            
            # Check for device mismatches
            unique_devices = set(devices.values())
            if len(unique_devices) > 1:
                print(f"   ⚠️  DEVICE MISMATCH: {unique_devices}")
            else:
                print(f"   ✅ All tensors on same device: {unique_devices}")
            
            # Test moving to GPU
            if torch.cuda.is_available():
                print("   🚀 Testing GPU transfer...")
                try:
                    batch_gpu = batch.to('cuda')
                    print("   ✅ GPU transfer successful")
                    
                    # Check after GPU transfer
                    gpu_devices = []
                    for attr in attrs_to_check:
                        if hasattr(batch_gpu, attr):
                            value = getattr(batch_gpu, attr)
                            if value is not None and isinstance(value, torch.Tensor):
                                gpu_devices.append(str(value.device))
                    
                    unique_gpu_devices = set(gpu_devices)
                    if len(unique_gpu_devices) == 1 and 'cuda' in list(unique_gpu_devices)[0]:
                        print("   ✅ All tensors successfully moved to GPU")
                    else:
                        print(f"   ❌ GPU transfer incomplete: {unique_gpu_devices}")
                        
                except Exception as e:
                    print(f"   ❌ GPU transfer failed: {e}")
            
        print("\n" + "="*50)
        
        return True
        
    except Exception as e:
        print(f"❌ Device check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_model_device_compatibility():
    """Check if model can handle device transfers"""
    print("🧠 Checking model device compatibility...")
    
    try:
        from src.models.joint_2d_3d_model import Joint2D3DMolecularModel
        from src.models.ddpm_diffusion import MolecularDDPM, MolecularDDPMModel
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Using device: {device}")
        
        base_model = Joint2D3DMolecularModel(
            atom_types=11, bond_types=4, hidden_dim=64, pocket_dim=64,
            num_layers=2, max_radius=10.0, max_pocket_atoms=100
        ).to(device)
        
        ddpm = MolecularDDPM(num_timesteps=100)
        model = MolecularDDPMModel(base_model, ddpm).to(device)
        
        print("   ✅ Model created and moved to device")
        
        # Create test tensors on different devices
        cpu_tensor = torch.randn(10, 8)
        print(f"   CPU tensor device: {cpu_tensor.device}")
        
        if torch.cuda.is_available():
            gpu_tensor = cpu_tensor.to('cuda')
            print(f"   GPU tensor device: {gpu_tensor.device}")
            
            # Test model with GPU tensor
            test_data = {
                'x': gpu_tensor,
                'pos': torch.randn(10, 3, device=device),
                'edge_index': torch.randint(0, 10, (2, 18), device=device),
                'edge_attr': torch.randn(18, 3, device=device),
                'batch': torch.zeros(10, dtype=torch.long, device=device)
            }
            
            try:
                output = base_model(**test_data)
                print("   ✅ Model forward pass successful with GPU tensors")
            except Exception as e:
                print(f"   ❌ Model forward pass failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model device check failed: {e}")
        return False

def suggest_fixes():
    """Suggest potential fixes for device issues"""
    print("\n💡 Potential fixes for device issues:")
    print("=" * 50)
    
    print("1. 🔧 Data Loader Fixes:")
    print("   - Set pin_memory=False in config")
    print("   - Set num_workers=0 for debugging")
    print("   - Use persistent_workers=False")
    
    print("\n2. 🔧 Model Fixes:")
    print("   - Explicitly move all tensors to same device in trainer")
    print("   - Add device checks in forward method")
    print("   - Use batch.to(device) before model call")
    
    print("\n3. 🔧 Training Loop Fixes:")
    print("   - Add comprehensive device transfer in trainer")
    print("   - Check tensor devices before each operation")
    print("   - Use try-except for device-related errors")
    
    print("\n4. 🧪 Quick Test:")
    print("   python check_device_consistency.py")
    print("   python scripts/train_ddpm.py --test --batch_size 2")

def main():
    print("🔧 Device Consistency Checker")
    print("=" * 50)
    
    # Check PyTorch and CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    print("\n" + "="*50)
    
    # Run checks
    batch_ok = check_batch_devices()
    model_ok = check_model_device_compatibility()
    
    print("\n📋 Summary:")
    print(f"   Batch device check: {'✅' if batch_ok else '❌'}")
    print(f"   Model device check: {'✅' if model_ok else '❌'}")
    
    if not (batch_ok and model_ok):
        suggest_fixes()
    else:
        print("\n🎉 All device checks passed!")
        print("   Try: python scripts/train_ddpm.py --test")

if __name__ == "__main__":
    main()