# src/models/__init__.py - UPDATED for SchNet backend
from .base_model import BaseModel, MolecularModel
from .ddpm_diffusion import MolecularDDPM, MolecularDDPMModel

# 🎯 UPDATED: Import SchNet-based models instead of EGNN
try:
    from .joint_2d_3d_model import (
        Joint2D3DSchNetModel,
        GraphConvLayer, 
        SchNetPocketEncoder,
        Enhanced2D3DFusion,
        create_joint2d3d_schnet_model
    )
    JOINT2D3D_AVAILABLE = True
    print("✅ Joint2D3D SchNet model available")
except ImportError as e:
    print(f"Warning: Joint2D3D SchNet model not available: {e}")
    JOINT2D3D_AVAILABLE = False

# Try to import pocket encoders
try:
    from .pocket_encoder import create_improved_pocket_encoder, SmartPocketAtomSelector
    POCKET_ENCODER_AVAILABLE = True
except ImportError:
    print("Warning: PocketEncoder not available")
    POCKET_ENCODER_AVAILABLE = False

# 🎯 MAIN EXPORTS - Updated for SchNet
__all__ = [
    # Core DDPM models
    'MolecularDDPM',
    'MolecularDDPMModel',
    
    # Base models
    'BaseModel',
    'MolecularModel',
    
    # 🎯 SchNet-based components (updated from EGNN)
    'Joint2D3DSchNetModel',
    'GraphConvLayer',
    'SchNetPocketEncoder',
    'Enhanced2D3DFusion', 
    'create_joint2d3d_schnet_model',
]

# Add pocket encoders if available
if POCKET_ENCODER_AVAILABLE:
    __all__.extend([
        'create_improved_pocket_encoder',
        'SmartPocketAtomSelector'
    ])

# Framework info
FRAMEWORK = "DDPM"
BACKEND = "SchNet"  # Updated from EGNN
SUPPORTED_DIFFUSION = ["DDPM", "DDIM"]

# 🎯 Compatibility aliases for backward compatibility
create_joint2d3d_egnn_model = create_joint2d3d_schnet_model  # Backward compatibility
Joint2D3DMolecularModel = Joint2D3DSchNetModel  # Backward compatibility

# 🎯 Updated compatibility check function
def check_model_availability():
    """Check which models are available"""
    status = {
        'joint2d3d_schnet': JOINT2D3D_AVAILABLE,
        'pocket_encoder': POCKET_ENCODER_AVAILABLE,
        'backend': 'SchNet' if JOINT2D3D_AVAILABLE else 'None'
    }
    
    print("🔍 Model Availability (SchNet Backend):")
    for name, available in status.items():
        if name == 'backend':
            print(f"   {name}: {available}")
        else:
            print(f"   {name}: {'✅' if available else '❌'}")
    
    return status

# 🎯 Updated factory function
def create_model(model_type: str = "joint2d3d_schnet", **kwargs):
    """Create model with SchNet backend"""
    
    if model_type in ["joint2d3d", "joint2d3d_schnet"]:
        if JOINT2D3D_AVAILABLE:
            return create_joint2d3d_schnet_model(**kwargs)
        else:
            raise ImportError("Joint2D3D SchNet model not available")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    print("🎯 Models Module - SchNet Backend")
    print("=" * 50)
    check_model_availability()
    
    if JOINT2D3D_AVAILABLE:
        print("\n✅ Ready for training with SchNet!")
        print("Use: from src.models import create_joint2d3d_schnet_model")
    else:
        print("\n❌ Models not ready. Check torch-geometric installation.")