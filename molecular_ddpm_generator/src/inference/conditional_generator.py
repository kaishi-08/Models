# src/inference/conditional_generator.py - DDMP-focused generator
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from ..models.joint_2d_3d_model import Joint2D3DModel
from ..models.ddpm_diffusion import MolecularDDPM

class DDPMMolecularGenerator:
    """DDPM-based molecular generation"""
    
    def __init__(self, model: Joint2D3DModel, ddpm: MolecularDDPM,
                 device: str = 'cuda'):
        self.model = model
        self.ddpm = ddpm
        self.device = device
        self.model.eval()
        
    def generate_molecules(self, pocket_data: Dict[str, torch.Tensor] = None,
                          num_molecules: int = 1, max_atoms: int = 50,
                          guidance_scale: float = 1.0) -> Dict[str, Any]:
        """Generate molecules using DDMP process"""
        
        generated_molecules = []
        
        for _ in range(num_molecules):
            # Sample molecule size
            num_atoms = torch.randint(10, max_atoms + 1, (1,)).item()
            
            # Initialize molecule structure
            molecule = self._generate_single_molecule(
                num_atoms=num_atoms,
                pocket_data=pocket_data,
                guidance_scale=guidance_scale
            )
            
            generated_molecules.append(molecule)
        
        return {
            'molecules': generated_molecules,
            'pocket_data': pocket_data,
            'generation_method': 'DDPM'
        }
    
    def _generate_single_molecule(self, num_atoms: int, 
                                 pocket_data: Dict[str, torch.Tensor] = None,
                                 guidance_scale: float = 1.0) -> Dict[str, torch.Tensor]:
        """Generate single molecule using DDPM sampling"""
        
        # Initialize random structure
        x_init = torch.randint(0, self.model.atom_types, (num_atoms, 1)).float().to(self.device)
        pos_init = torch.randn(num_atoms, 3).to(self.device)
        
        # Create basic connectivity
        edge_index = self._create_initial_edges(num_atoms)
        edge_attr = torch.randint(0, self.model.bond_types, (edge_index.size(1), 1)).float().to(self.device)
        batch = torch.zeros(num_atoms, dtype=torch.long).to(self.device)
        
        # DDPM reverse process
        current_pos = pos_init.clone()
        
        # Reverse diffusion process
        for t in reversed(range(self.ddmp.num_timesteps)):
            t_batch = torch.full((1,), t, dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                # Predict noise
                noise_pred = self.model(
                    x=x_init,
                    pos=current_pos,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch,
                    t=t_batch,
                    **(pocket_data or {})
                )
                
                # DDPM reverse step
                current_pos, _ = self.ddpm.reverse_process(
                    self.model, current_pos, t_batch, 
                    x=x_init, edge_index=edge_index, edge_attr=edge_attr, 
                    batch=batch, **(pocket_data or {})
                )
        
        return {
            'x': x_init,
            'pos': current_pos,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'batch': batch
        }
    
    def _create_initial_edges(self, num_atoms: int) -> torch.Tensor:
        """Create initial molecular connectivity"""
        edges = []
        
        # Create chain connectivity
        for i in range(num_atoms - 1):
            edges.extend([[i, i + 1], [i + 1, i]])
        
        # Add some random connections
        for _ in range(min(num_atoms // 3, 5)):
            i, j = torch.randint(0, num_atoms, (2,)).tolist()
            if i != j:
                edges.extend([[i, j], [j, i]])
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
    
    def molecules_to_smiles(self, molecules: List[Dict]) -> List[str]:
        """Convert generated molecules to SMILES"""
        smiles_list = []
        
        for mol_data in molecules:
            try:
                mol = self._tensor_to_rdkit_mol(mol_data)
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)
                    smiles_list.append(smiles)
                else:
                    smiles_list.append(None)
            except Exception as e:
                print(f"Error converting to SMILES: {e}")
                smiles_list.append(None)
        
        return smiles_list
    
    def _tensor_to_rdkit_mol(self, mol_data: Dict[str, torch.Tensor]):
        """Convert tensor to RDKit molecule"""
        try:
            x = mol_data['x'].cpu().numpy()
            pos = mol_data['pos'].cpu().numpy()
            edge_index = mol_data['edge_index'].cpu().numpy()
            edge_attr = mol_data['edge_attr'].cpu().numpy()
            
            # Create molecule
            mol = Chem.RWMol()
            
            # Add atoms
            atom_map = {}
            for i, atom_type in enumerate(x):
                # Map to atomic numbers (simplified)
                atomic_num = min(int(atom_type[0]) + 1, 6)  # C, N, O mainly
                atom_idx = mol.AddAtom(Chem.Atom(atomic_num))
                atom_map[i] = atom_idx
            
            # Add bonds
            added_bonds = set()
            for i in range(edge_index.shape[1]):
                atom1, atom2 = edge_index[:, i]
                if atom1 < atom2:  # Avoid duplicates
                    bond_type = int(edge_attr[i, 0])
                    bond_type_rdkit = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, 
                                      Chem.BondType.TRIPLE][min(bond_type, 2)]
                    
                    if (atom1, atom2) not in added_bonds:
                        mol.AddBond(atom_map[atom1], atom_map[atom2], bond_type_rdkit)
                        added_bonds.add((atom1, atom2))
            
            # Sanitize
            Chem.SanitizeMol(mol)
            return mol
            
        except:
            return None

# Backward compatibility
ConditionalMolecularGenerator = DDPMMolecularGenerator