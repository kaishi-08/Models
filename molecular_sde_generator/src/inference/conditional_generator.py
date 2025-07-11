# src/inference/conditional_generator.py
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

class ConditionalMolecularGenerator:
    """Conditional molecular generation using SDE diffusion"""
    
    def __init__(self, model: Joint2D3DMolecularModel, sde: VESDE,
                 device: str = 'cuda'):
        self.model = model
        self.sde = sde
        self.device = device
        self.model.eval()
        
    def generate_molecules(self, pocket_data: Dict[str, torch.Tensor],
                          num_molecules: int = 1, max_atoms: int = 50,
                          guidance_scale: float = 1.0) -> Dict[str, Any]:
        """Generate molecules conditioned on protein pocket"""
        
        generated_molecules = []
        
        for _ in range(num_molecules):
            # Sample initial noise
            num_atoms = torch.randint(10, max_atoms + 1, (1,)).item()
            
            # Initialize from noise
            x_init = torch.randint(0, self.model.atom_types, (num_atoms, 1)).to(self.device)
            pos_init = self.sde.prior_sampling((num_atoms, 3)).to(self.device)
            
            # Create initial molecular graph
            edge_index = self._create_initial_edges(num_atoms)
            edge_attr = torch.randint(0, self.model.bond_types, (edge_index.size(1), 1)).to(self.device)
            batch = torch.zeros(num_atoms, dtype=torch.long).to(self.device)
            
            # Generate molecule using reverse SDE
            molecule = self._reverse_sde_sampling(
                x_init, pos_init, edge_index, edge_attr, batch,
                pocket_data, guidance_scale
            )
            
            generated_molecules.append(molecule)
        
        return {
            'molecules': generated_molecules,
            'pocket_data': pocket_data
        }
    
    def _reverse_sde_sampling(self, x: torch.Tensor, pos: torch.Tensor,
                             edge_index: torch.Tensor, edge_attr: torch.Tensor,
                             batch: torch.Tensor, pocket_data: Dict[str, torch.Tensor],
                             guidance_scale: float = 1.0) -> Dict[str, torch.Tensor]:
        """Reverse SDE sampling to generate molecule"""
        
        # Time steps
        time_steps = torch.linspace(1., 0., self.sde.N + 1).to(self.device)
        
        current_x = x.clone()
        current_pos = pos.clone()
        current_edge_attr = edge_attr.clone()
        
        with torch.no_grad():
            for i in range(self.sde.N):
                t = time_steps[i]
                t_batch = t.repeat(batch.max().item() + 1)
                
                # Predict score and other properties
                outputs = self.model(
                    x=current_x,
                    pos=current_pos,
                    edge_index=edge_index,
                    edge_attr=current_edge_attr,
                    batch=batch,
                    **pocket_data
                )
                
                # Get drift and diffusion coefficients
                drift, diffusion = self.sde.sde(current_pos, t_batch)
                
                # Compute score
                score = outputs['pos_pred']
                
                # Apply guidance
                if guidance_scale != 1.0:
                    # Unconditional score
                    outputs_uncond = self.model(
                        x=current_x,
                        pos=current_pos,
                        edge_index=edge_index,
                        edge_attr=current_edge_attr,
                        batch=batch
                    )
                    score_uncond = outputs_uncond['pos_pred']
                    
                    # Classifier-free guidance
                    score = score_uncond + guidance_scale * (score - score_uncond)
                
                # Reverse SDE step
                dt = time_steps[i + 1] - time_steps[i]
                drift_corrected = drift - (diffusion ** 2)[:, None] * score
                
                # Update positions
                current_pos = current_pos + drift_corrected * dt
                
                if i < self.sde.N - 1:  # Don't add noise in the last step
                    noise = torch.randn_like(current_pos)
                    current_pos = current_pos + diffusion[:, None] * np.sqrt(-dt) * noise
                
                # Update discrete features periodically
                if i % 100 == 0:
                    atom_probs = F.softmax(outputs['atom_logits'], dim=-1)
                    current_x = torch.multinomial(atom_probs, 1)
                    
                    if 'bond_logits' in outputs:
                        bond_probs = F.softmax(outputs['bond_logits'], dim=-1)
                        current_edge_attr = torch.multinomial(bond_probs, 1)
        
        return {
            'x': current_x,
            'pos': current_pos,
            'edge_index': edge_index,
            'edge_attr': current_edge_attr,
            'batch': batch
        }
    
    def _create_initial_edges(self, num_atoms: int) -> torch.Tensor:
        """Create initial edge connectivity"""
        # Create a simple connectivity pattern
        edges = []
        for i in range(num_atoms - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        
        # Add some random edges
        for _ in range(min(num_atoms // 2, 5)):
            i, j = torch.randint(0, num_atoms, (2,)).tolist()
            if i != j:
                edges.append([i, j])
                edges.append([j, i])
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
    
    def molecules_to_smiles(self, molecules: list) -> list:
        """Convert generated molecules to SMILES strings"""
        smiles_list = []
        
        for mol_data in molecules:
            try:
                # Create RDKit molecule
                mol = self._tensor_to_rdkit_mol(mol_data)
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)
                    smiles_list.append(smiles)
                else:
                    smiles_list.append(None)
            except Exception as e:
                print(f"Error converting molecule to SMILES: {e}")
                smiles_list.append(None)
        
        return smiles_list
    
    def _tensor_to_rdkit_mol(self, mol_data: Dict[str, torch.Tensor]):
        """Convert tensor representation to RDKit molecule"""
        # This is a simplified conversion - in practice, you'd need
        # a more sophisticated approach to handle all chemical constraints
        
        x = mol_data['x'].cpu().numpy()
        pos = mol_data['pos'].cpu().numpy()
        edge_index = mol_data['edge_index'].cpu().numpy()
        edge_attr = mol_data['edge_attr'].cpu().numpy()
        
        # Create RDKit molecule
        mol = Chem.RWMol()
        
        # Add atoms
        atom_map = {}
        for i, atom_type in enumerate(x):
            atom_idx = mol.AddAtom(Chem.Atom(int(atom_type[0]) + 1))  # +1 for atomic number
            atom_map[i] = atom_idx
        
        # Add bonds
        for i in range(edge_index.shape[1]):
            atom1, atom2 = edge_index[:, i]
            bond_type = edge_attr[i, 0]
            
            if atom1 < atom2:  # Avoid duplicate bonds
                bond_type_rdkit = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                                  Chem.BondType.TRIPLE][min(int(bond_type), 2)]
                mol.AddBond(atom_map[atom1], atom_map[atom2], bond_type_rdkit)
        
        try:
            Chem.SanitizeMol(mol)
            return mol
        except:
            return None