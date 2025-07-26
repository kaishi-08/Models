# src/models/__init__.py - FIXED for training compatibility
from .base_model import BaseModel, MolecularModel
from .ddpm_diffusion import MolecularDDPM, MolecularDDPMModel

# 🔧 FIXED: Import all classes that training script expects
try:
    from .joint_2d_3d_model import (
        Joint2D3DMolecularModel, 
        GraphConvLayer, 
        SimplePocketEncoder,
        Enhanced2D3DFusion,
        EnhancedPocketEncoder,
        create_joint2d3d_egnn_model
    )
    JOINT2D3D_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Joint2D3D model not available: {e}")
    JOINT2D3D_AVAILABLE = False

# Try to import pocket encoders
try:
    from .pocket_encoder import ProteinPocketEncoder, CrossAttentionPocketConditioner, SmartPocketAtomSelector
    POCKET_ENCODER_AVAILABLE = True
except ImportError:
    print("Warning: PocketEncoder not available")
    POCKET_ENCODER_AVAILABLE = False

# Try improved pocket encoder
try:
    from pocket_encoder import ImprovedProteinPocketEncoder, create_improved_pocket_encoder
    IMPROVED_POCKET_AVAILABLE = True
except ImportError:
    print("Warning: ImprovedPocketEncoder not available")
    IMPROVED_POCKET_AVAILABLE = False

# Optional E3 components (fallback if not available)
try:
    from .e3_egnn import E3EquivariantGNN, E3EquivariantLayer, GaussianSmearing
    E3_AVAILABLE = True
except ImportError:
    print("E3NN not available, using fallback implementations")
    E3_AVAILABLE = False

# 🔧 MAIN EXPORTS - All classes training script expects
__all__ = [
    # Core DDPM models
    'MolecularDDPM',
    'MolecularDDPMModel',
    
    # Base models
    'BaseModel',
    'MolecularModel',
    
    # 🔧 Joint2D3D components (training script expects these)
    'Joint2D3DMolecularModel',
    'GraphConvLayer',
    'SimplePocketEncoder',
    'Enhanced2D3DFusion', 
    'EnhancedPocketEncoder',
    'create_joint2d3d_egnn_model',
]

# Add pocket encoders if available
if POCKET_ENCODER_AVAILABLE:
    __all__.extend([
        'ProteinPocketEncoder',
        'CrossAttentionPocketConditioner', 
        'SmartPocketAtomSelector'
    ])

if IMPROVED_POCKET_AVAILABLE:
    __all__.extend([
        'ImprovedProteinPocketEncoder',
        'create_improved_pocket_encoder'
    ])

# Add E3 components if available
if E3_AVAILABLE:
    __all__.extend(['E3EquivariantGNN', 'E3EquivariantLayer', 'GaussianSmearing'])

# Framework info
FRAMEWORK = "DDPM"
SUPPORTED_DIFFUSION = ["DDPM", "DDIM"]

# 🔧 Compatibility check function
def check_model_availability():
    """Check which models are available"""
    status = {
        'joint2d3d': JOINT2D3D_AVAILABLE,
        'pocket_encoder': POCKET_ENCODER_AVAILABLE, 
        'improved_pocket': IMPROVED_POCKET_AVAILABLE,
        'e3_components': E3_AVAILABLE
    }
    
    print("🔍 Model Availability:")
    for name, available in status.items():
        print(f"   {name}: {'✅' if available else '❌'}")
    
    return status

# 🔧 Factory function with fallbacks
def create_model(model_type: str = "joint2d3d", **kwargs):
    """Create model with automatic fallbacks"""
    
    if model_type == "joint2d3d":
        if JOINT2D3D_AVAILABLE:
            return create_joint2d3d_egnn_model(**kwargs)
        else:
            raise ImportError("Joint2D3D model not available")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    print("🔧 Models Module - Compatibility Check")
    print("=" * 50)
    check_model_availability()
    
    if JOINT2D3D_AVAILABLE:
        print("\n✅ Ready for training!")
        print("Use: from src.models import create_joint2d3d_egnn_model")
    else:
        print("\n❌ Models not ready. Check imports and dependencies.")