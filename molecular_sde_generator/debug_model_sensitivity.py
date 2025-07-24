# debug_model_sensitivity.py - Debug why model ignores inputs
import torch
import sys
from pathlib import Path

# Setup paths
project_root = Path.cwd()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_base_model_sensitivity():
    """Test if base model is sensitive to inputs"""
    print("🔍 TESTING BASE MODEL SENSITIVITY")
    print("=" * 60)
    
    try:
        from src.models.joint_2d_3d_model import Joint2D3DMolecularModel
        from src.data.data_loaders import CrossDockDataLoader
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create simpler model for testing
        model = Joint2D3DMolecularModel(
            atom_types=11, bond_types=4, 
            hidden_dim=64,  # Smaller for testing
            pocket_dim=64,
            num_layers=2,   # Fewer layers
            conditioning_type="add"
        ).to(device)
        
        print(f"   ✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Get real batch
        config = {
            'data': {
                'train_path': 'data/processed_full_normalized/train_generation.pkl',
                'batch_size': 2,
                'num_workers': 1,
                'shuffle': False,
                'pin_memory': False
            },
            'include_pocket': True,
            'max_atoms': 20,  # Smaller for testing
            'augment': False
        }
        
        train_loader = CrossDockDataLoader.create_train_loader(config)
        batch = next(iter(train_loader))
        batch = batch.to(device)
        
        print(f"   ✅ Loaded batch: {batch.x.shape[0]} atoms")
        
        # Test model sensitivity to different inputs
        model.eval()
        
        # Original input
        with torch.no_grad():
            outputs1 = model(
                x=batch.x,
                pos=batch.pos,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                pocket_x=batch.pocket_x,
                pocket_pos=batch.pocket_pos,
                pocket_edge_index=batch.pocket_edge_index,
                pocket_batch=batch.pocket_batch
            )
        
        pos_pred1 = outputs1['pos_pred']
        print(f"   Original prediction: mean={pos_pred1.mean():.4f}, std={pos_pred1.std():.4f}")
        
        # Test 1: Change positions dramatically
        modified_pos = batch.pos + torch.randn_like(batch.pos) * 5.0  # Large change
        
        with torch.no_grad():
            outputs2 = model(
                x=batch.x,
                pos=modified_pos,  # CHANGED
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                pocket_x=batch.pocket_x,
                pocket_pos=batch.pocket_pos,
                pocket_edge_index=batch.pocket_edge_index,
                pocket_batch=batch.pocket_batch
            )
        
        pos_pred2 = outputs2['pos_pred']
        diff_pos = torch.norm(pos_pred2 - pos_pred1, dim=-1).mean()
        print(f"   Position change sensitivity: {diff_pos:.4f}")
        
        # Test 2: Change atom features
        modified_x = batch.x + torch.randn_like(batch.x) * 0.5
        
        with torch.no_grad():
            outputs3 = model(
                x=modified_x,  # CHANGED
                pos=batch.pos,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                pocket_x=batch.pocket_x,
                pocket_pos=batch.pocket_pos,
                pocket_edge_index=batch.pocket_edge_index,
                pocket_batch=batch.pocket_batch
            )
        
        pos_pred3 = outputs3['pos_pred']
        diff_x = torch.norm(pos_pred3 - pos_pred1, dim=-1).mean()
        print(f"   Atom feature sensitivity: {diff_x:.4f}")
        
        # Test 3: Remove pocket data
        with torch.no_grad():
            outputs4 = model(
                x=batch.x,
                pos=batch.pos,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                pocket_x=None,  # REMOVED
                pocket_pos=None,  # REMOVED
                pocket_edge_index=None,
                pocket_batch=None
            )
        
        pos_pred4 = outputs4['pos_pred']
        diff_pocket = torch.norm(pos_pred4 - pos_pred1, dim=-1).mean()
        print(f"   Pocket removal sensitivity: {diff_pocket:.4f}")
        
        # Analysis
        print(f"\n   📊 SENSITIVITY ANALYSIS:")
        if diff_pos < 0.001:
            print(f"      🚨 Model IGNORES position changes!")
        else:
            print(f"      ✅ Model responds to position changes")
        
        if diff_x < 0.001:
            print(f"      🚨 Model IGNORES atom features!")
        else:
            print(f"      ✅ Model responds to atom features")
        
        if diff_pocket < 0.001:
            print(f"      🚨 Model IGNORES pocket data!")
        else:
            print(f"      ✅ Model responds to pocket data")
        
        # Test 4: Check if model is just returning constant
        all_same = torch.allclose(pos_pred1[0], pos_pred1[1:], atol=1e-3)
        if all_same:
            print(f"      🚨 Model returns SAME output for all atoms!")
        else:
            print(f"      ✅ Model produces different outputs per atom")
        
        return diff_pos > 0.001 and diff_x > 0.001
        
    except Exception as e:
        print(f"   ❌ Base model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_model():
    """Test extremely simple model to verify basic functionality"""
    print("\n🧪 TESTING SIMPLE MODEL")
    print("=" * 60)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create minimal model
        class SimpleTestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)  # Just position -> position
            
            def forward(self, pos, **kwargs):
                return self.linear(pos)
        
        simple_model = SimpleTestModel().to(device)
        
        # Test with random positions
        pos1 = torch.randn(10, 3, device=device)
        pos2 = torch.randn(10, 3, device=device)
        
        out1 = simple_model(pos1)
        out2 = simple_model(pos2)
        
        diff = torch.norm(out2 - out1, dim=-1).mean()
        print(f"   Simple model sensitivity: {diff:.4f}")
        
        if diff > 0.1:
            print(f"   ✅ Simple model works correctly")
            return True
        else:
            print(f"   🚨 Even simple model has issues!")
            return False
            
    except Exception as e:
        print(f"   ❌ Simple model test failed: {e}")
        return False

def check_gradient_flow():
    """Check if gradients are flowing through the model"""
    print("\n🌊 CHECKING GRADIENT FLOW")
    print("=" * 60)
    
    try:
        from src.models.joint_2d_3d_model import Joint2D3DMolecularModel
        from src.data.data_loaders import CrossDockDataLoader
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        model = Joint2D3DMolecularModel(
            atom_types=11, bond_types=4, 
            hidden_dim=64, pocket_dim=64, num_layers=2,
            conditioning_type="add"
        ).to(device)
        
        # Get batch
        config = {
            'data': {
                'train_path': 'data/processed_full_normalized/train_generation.pkl',
                'batch_size': 1, 'num_workers': 1, 'shuffle': False, 'pin_memory': False
            },
            'include_pocket': True, 'max_atoms': 10, 'augment': False
        }
        
        train_loader = CrossDockDataLoader.create_train_loader(config)
        batch = next(iter(train_loader))
        batch = batch.to(device)
        
        # Enable gradients
        batch.pos.requires_grad_(True)
        batch.x.requires_grad_(True)
        
        model.train()
        
        # Forward pass
        outputs = model(
            x=batch.x, pos=batch.pos, edge_index=batch.edge_index,
            edge_attr=batch.edge_attr, batch=batch.batch,
            pocket_x=batch.pocket_x, pocket_pos=batch.pocket_pos,
            pocket_edge_index=batch.pocket_edge_index, pocket_batch=batch.pocket_batch
        )
        
        pos_pred = outputs['pos_pred']
        
        # Create loss
        target = torch.randn_like(pos_pred)
        loss = torch.nn.MSELoss()(pos_pred, target)
        
        # Backward
        loss.backward()
        
        # Check gradients
        param_grads = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_grads.append((name, grad_norm))
            else:
                param_grads.append((name, 0.0))
        
        print(f"   📊 Gradient norms:")
        for name, grad_norm in param_grads[:10]:  # Show first 10
            print(f"      {name}: {grad_norm:.6f}")
        
        # Check input gradients
        if batch.pos.grad is not None:
            pos_grad_norm = batch.pos.grad.norm().item()
            print(f"   Position input gradient: {pos_grad_norm:.6f}")
        
        if batch.x.grad is not None:
            x_grad_norm = batch.x.grad.norm().item()
            print(f"   Feature input gradient: {x_grad_norm:.6f}")
        
        # Analysis
        param_grad_norms = [grad for _, grad in param_grads if grad > 0]
        if len(param_grad_norms) == 0:
            print(f"   🚨 NO GRADIENTS FLOWING!")
            return False
        elif max(param_grad_norms) < 1e-6:
            print(f"   🚨 VANISHING GRADIENTS!")
            return False
        else:
            print(f"   ✅ Gradients flowing (max: {max(param_grad_norms):.6f})")
            return True
        
    except Exception as e:
        print(f"   ❌ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔍 MODEL SENSITIVITY DEBUGGING")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Base Model Sensitivity", test_base_model_sensitivity()))
    results.append(("Simple Model", test_simple_model()))
    results.append(("Gradient Flow", check_gradient_flow()))
    
    # Summary
    print("\n📋 SENSITIVITY TEST RESULTS:")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    failed_tests = [name for name, passed in results if not passed]
    
    if failed_tests:
        print(f"\n🚨 FAILED TESTS: {', '.join(failed_tests)}")
        print("\n🔧 LIKELY ISSUES:")
        if "Base Model Sensitivity" in failed_tests:
            print("   - Model architecture not connecting inputs to outputs")
            print("   - Position head not properly connected")
            print("   - Feature fusion layer broken")
        if "Gradient Flow" in failed_tests:
            print("   - Vanishing gradients")
            print("   - Broken backpropagation")
            print("   - Dead ReLUs or other activation issues")
    else:
        print(f"\n✅ All sensitivity tests passed!")
        print("   Issue might be in DDPM wrapper or training loop")

if __name__ == "__main__":
    main()