# src/models/__init__.py - UPDATED for EGNN backend
from .base_model import BaseModel, MolecularModel
from .ddpm_diffusion import MolecularDDPM, MolecularDDPMModel

# 🎯 UPDATED: Import EGNN-based models instead of SchNet
try:
    from .egnn import (
        EGNNLayer,
        EGNNBackbone, 
        Joint2D3DEGNNModel,
        GraphConvLayer,
        create_egnn_model,
        create_joint2d3d_egnn_model
    )
    from .joint_2d_3d_model import (
        Joint2D3DModel,
        create_joint2d3d_model,
        create_joint2d3d_egnn_model as create_joint2d3d_model_direct
    )
    EGNN_AVAILABLE = True
    print("✅ EGNN models available - E(n) equivariant backend loaded")
except ImportError as e:
    print(f"Warning: EGNN models not available: {e}")
    EGNN_AVAILABLE = False

# Try to import pocket encoders
try:
    from .pocket_encoder import create_improved_pocket_encoder, SmartPocketAtomSelector
    POCKET_ENCODER_AVAILABLE = True
except ImportError:
    print("Warning: PocketEncoder not available")
    POCKET_ENCODER_AVAILABLE = False

# 🎯 MAIN EXPORTS - Updated for EGNN
__all__ = [
    # Core DDPM models
    'MolecularDDPM',
    'MolecularDDPMModel',
    
    # Base models
    'BaseModel',
    'MolecularModel',
]

# Add EGNN components if available
if EGNN_AVAILABLE:
    __all__.extend([
        # 🎯 EGNN components (updated from SchNet)
        'EGNNLayer',
        'EGNNBackbone',
        'Joint2D3DEGNNModel',
        'Joint2D3DModel',
        'GraphConvLayer',
        'create_egnn_model',
        'create_joint2d3d_egnn_model',
        'create_joint2d3d_model',
    ])

# Add pocket encoders if available
if POCKET_ENCODER_AVAILABLE:
    __all__.extend([
        'create_improved_pocket_encoder',
        'SmartPocketAtomSelector'
    ])

# Framework info
FRAMEWORK = "DDPM"
BACKEND = "EGNN"  # Updated from SchNet
SUPPORTED_DIFFUSION = ["DDPM", "DDIM"]

# 🎯 Backward compatibility aliases
if EGNN_AVAILABLE:
    # Redirect old SchNet calls to EGNN
    create_joint2d3d_schnet_model = create_joint2d3d_egnn_model
    Joint2D3DSchNetModel = Joint2D3DModel
    Joint2D3DMolecularModel = Joint2D3DModel
    
    # Add to exports
    __all__.extend([
        'create_joint2d3d_schnet_model',  # Backward compatibility
        'Joint2D3DSchNetModel',         # Backward compatibility
        'Joint2D3DMolecularModel'       # Legacy alias
    ])

# 🎯 Updated compatibility check function
def check_model_availability():
    """Check which models are available"""
    status = {
        'egnn_backend': EGNN_AVAILABLE,
        'pocket_encoder': POCKET_ENCODER_AVAILABLE,
        'backend': 'EGNN' if EGNN_AVAILABLE else 'None',
        'equivariant': EGNN_AVAILABLE
    }
    
    print("🔍 Model Availability (EGNN Backend):")
    for name, available in status.items():
        if name == 'backend':
            print(f"   {name}: {available}")
        elif name == 'equivariant':
            print(f"   E(n)-equivariant: {'✅' if available else '❌'}")
        else:
            print(f"   {name}: {'✅' if available else '❌'}")
    
    return status

# 🎯 Updated factory function
def create_model(model_type: str = "joint2d3d_egnn", **kwargs):
    """Create model with EGNN backend"""
    
    if model_type in ["joint2d3d", "joint2d3d_egnn", "egnn"]:
        if EGNN_AVAILABLE:
            return create_joint2d3d_egnn_model(**kwargs)
        else:
            raise ImportError("EGNN models not available")
    elif model_type in ["joint2d3d_schnet", "schnet"]:
        # Redirect to EGNN for backward compatibility
        print("⚠️  SchNet requested but redirecting to EGNN for better performance")
        if EGNN_AVAILABLE:
            return create_joint2d3d_egnn_model(**kwargs)
        else:
            raise ImportError("EGNN models not available")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# 🎯 Model recommendation function
def get_recommended_model(**kwargs):
    """Get recommended model configuration"""
    if EGNN_AVAILABLE:
        return create_joint2d3d_egnn_model(
            hidden_dim=kwargs.get('hidden_dim', 256),
            num_layers=kwargs.get('num_layers', 4),  # EGNN works well with 4 layers
            cutoff=kwargs.get('cutoff', 10.0),
            **{k: v for k, v in kwargs.items() if k not in ['hidden_dim', 'num_layers', 'cutoff']}
        )
    else:
        raise ImportError("No models available. Install torch-geometric.")

if __name__ == "__main__":
    print("🎯 Models Module - EGNN Backend")
    print("=" * 50)
    check_model_availability()
    
    if EGNN_AVAILABLE:
        print("\n✅ Ready for training with EGNN!")
        print("Use: from src.models import create_joint2d3d_egnn_model")
        print("\n🌟 EGNN Benefits:")
        print("- E(n) equivariant (rotation/translation invariant)")
        print("- Proven in Pocket2Mol, SBDDiff")
        print("- More stable than SchNet")
        print("- Better molecular generation quality")
    else:
        print("\n❌ Models not ready. Install torch-geometric>=2.0")