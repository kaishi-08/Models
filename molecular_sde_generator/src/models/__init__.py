from .base_model import BaseModel, MolecularModel
from .joint_2d_3d_model import Joint2D3DMolecularModel, GraphConvLayer, SimplePocketEncoder
from .pocket_encoder import ProteinPocketEncoder, CrossAttentionPocketConditioner, SmartPocketAtomSelector
from .ddpm_diffusion import MolecularDDPM, MolecularDDPMModel

# Optional E3 components (fallback if not available)
try:
    from .e3_egnn import E3EquivariantGNN, E3EquivariantLayer, GaussianSmearing
    E3_AVAILABLE = True
except ImportError:
    print("E3NN not available, using fallback implementations")
    E3_AVAILABLE = False

__all__ = [
    # Core DDPM models
    'MolecularDDPM',
    'MolecularDDPMModel',
    
    # Base models
    'BaseModel',
    'MolecularModel',
    'Joint2D3DMolecularModel',
    'GraphConvLayer',
    'SimplePocketEncoder',
    
    # Pocket encoding
    'ProteinPocketEncoder',
    'CrossAttentionPocketConditioner',
    'SmartPocketAtomSelector'
]

# Add E3 components if available
if E3_AVAILABLE:
    __all__.extend(['E3EquivariantGNN', 'E3EquivariantLayer', 'GaussianSmearing'])

# Framework info
FRAMEWORK = "DDPM"
SUPPORTED_DIFFUSION = ["DDPM", "DDIM"]  # Future extensions