# src/inference/chemical_generator.py - Fixed Generator with Chemical Constraints
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from typing import List, Dict, Any, Optional
import random

class ChemicallyValidGenerator:
    """Chemical-aware molecular generator with valence constraints"""
    
    def __init__(self, model=None, ddpm=None, device='cuda'):
        self.model = model
        self.ddpm = ddpm
        self.device = device
        
        # Chemical knowledge
        self.valence_rules = {
            6: 4,   # Carbon
            7: 3,   # Nitrogen (can be 4 with charge)
            8: 2,   # Oxygen (can be 3 with charge)
            16: 6,  # Sulfur
            9: 1,   # Fluorine
            17: 1,  # Chlorine
            35: 1,  # Bromine
            53: 1   # Iodine
        }
        
        # Common bond lengths (Angstroms)
        self.bond_lengths = {
            (6, 6): 1.54,   # C-C
            (6, 7): 1.47,   # C-N
            (6, 8): 1.43,   # C-O
            (7, 7): 1.45,   # N-N
            (7, 8): 1.40,   # N-O
            (8, 8): 1.48    # O-O
        }
        
        # Atom weights for realistic distribution
        self.atom_weights = {
            6: 0.50,   # Carbon (most common)
            7: 0.20,   # Nitrogen
            8: 0.20,   # Oxygen
            16: 0.05,  # Sulfur
            9: 0.03,   # Fluorine
            17: 0.02   # Chlorine
        }
        
    def generate_molecules(self, num_molecules: int = 1, max_atoms: int = 50, 
                          min_atoms: int = 8, **kwargs) -> Dict[str, Any]:
        """Generate chemically valid molecules"""
        
        generated_molecules = []
        
        for i in range(num_molecules):
            try:
                # Generate molecule with proper size
                num_atoms = random.randint(min_atoms, min(max_atoms, 30))
                
                # Step 1: Generate atom types with chemical distribution
                atom_types = self._generate_atom_types(num_atoms)
                
                # Step 2: Generate 3D positions
                positions = self._generate_3d_positions(num_atoms)
                
                # Step 3: Generate valence-constrained bonds
                edge_index, edge_attr = self._generate_valid_bonds(atom_types, positions)
                
                # Step 4: Create molecule data
                mol_data = {
                    'atom_types': atom_types,
                    'positions': positions,
                    'edge_index': edge_index,
                    'edge_attr': edge_attr,
                    'num_atoms': num_atoms
                }
                
                # Step 5: Validate chemistry
                if self._validate_chemistry(mol_data):
                    generated_molecules.append(mol_data)
                    
            except Exception as e:
                print(f"Error generating molecule {i}: {e}")
                continue
        
        return {'molecules': generated_molecules}
    
    def _generate_atom_types(self, num_atoms: int) -> torch.Tensor:
        """Generate atom types with realistic chemical distribution"""
        
        # Sample atoms based on chemical abundance
        atoms = []
        atom_list = list(self.atom_weights.keys())
        weights = list(self.atom_weights.values())
        
        for _ in range(num_atoms):
            atom_type = np.random.choice(atom_list, p=weights)
            atoms.append(atom_type)
        
        # Ensure at least 60% carbon for organic molecules
        carbon_count = atoms.count(6)
        if carbon_count < num_atoms * 0.6:
            needed_carbons = int(num_atoms * 0.6) - carbon_count
            for i in range(needed_carbons):
                if i < len(atoms):
                    atoms[i] = 6
        
        return torch.tensor(atoms, dtype=torch.long)
    
    def _generate_3d_positions(self, num_atoms: int) -> torch.Tensor:
        """Generate realistic 3D positions"""
        
        if num_atoms <= 1:
            return torch.zeros(num_atoms, 3)
        
        # Start with a reasonable molecular framework
        positions = []
        
        # Place first atom at origin
        positions.append([0.0, 0.0, 0.0])
        
        # Place second atom at bond distance
        if num_atoms > 1:
            positions.append([1.5, 0.0, 0.0])
        
        # Place remaining atoms with realistic geometry
        for i in range(2, num_atoms):
            # Find a position that maintains reasonable bond distances
            max_attempts = 50
            for attempt in range(max_attempts):
                # Random position around existing atoms
                base_idx = random.randint(0, i-1)
                base_pos = np.array(positions[base_idx])
                
                # Random direction with realistic distance
                direction = np.random.randn(3)
                direction = direction / np.linalg.norm(direction)
                distance = random.uniform(1.2, 2.0)  # Realistic bond lengths
                
                new_pos = base_pos + direction * distance
                
                # Check minimum distances to avoid overlaps
                too_close = False
                for existing_pos in positions:
                    if np.linalg.norm(new_pos - existing_pos) < 1.0:
                        too_close = True
                        break
                
                if not too_close:
                    positions.append(new_pos.tolist())
                    break
            else:
                # Fallback: place randomly if can't find good position
                positions.append([
                    random.uniform(-5, 5),
                    random.uniform(-5, 5), 
                    random.uniform(-5, 5)
                ])
        
        return torch.tensor(positions, dtype=torch.float32)
    
    def _generate_valid_bonds(self, atom_types: torch.Tensor, 
                            positions: torch.Tensor) -> tuple:
        """Generate bonds respecting valence constraints"""
        
        num_atoms = len(atom_types)
        if num_atoms <= 1:
            return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 1, dtype=torch.float32)
        
        # Track current valences
        current_valences = {i: 0 for i in range(num_atoms)}
        edges = []
        edge_types = []
        
        # Compute distance matrix
        distances = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0))[0]
        
        # Sort atom pairs by distance (prefer closer atoms)
        atom_pairs = []
        for i in range(num_atoms):
            for j in range(i+1, num_atoms):
                atom_pairs.append((i, j, distances[i, j].item()))
        
        atom_pairs.sort(key=lambda x: x[2])  # Sort by distance
        
        # Add bonds respecting valence limits
        for i, j, dist in atom_pairs:
            atom_i_type = atom_types[i].item()
            atom_j_type = atom_types[j].item()
            
            max_val_i = self.valence_rules.get(atom_i_type, 4)
            max_val_j = self.valence_rules.get(atom_j_type, 4)
            
            # Check if both atoms can form more bonds
            if (current_valences[i] < max_val_i and 
                current_valences[j] < max_val_j and
                dist < 3.0):  # Reasonable bonding distance
                
                # Add bond (always single for simplicity)
                edges.extend([[i, j], [j, i]])
                edge_types.extend([1, 1])  # Single bond
                
                current_valences[i] += 1
                current_valences[j] += 1
                
                # Stop if molecule is well-connected
                if len(edges) >= (num_atoms - 1) * 2:  # At least spanning tree
                    break
        
        # Ensure minimum connectivity (spanning tree)
        if len(edges) < (num_atoms - 1) * 2:
            self._ensure_connectivity(edges, edge_types, num_atoms, current_valences, atom_types)
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_types, dtype=torch.float32).unsqueeze(1)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, 1, dtype=torch.float32)
        
        return edge_index, edge_attr
    
    def _ensure_connectivity(self, edges: List, edge_types: List, num_atoms: int,
                           current_valences: Dict, atom_types: torch.Tensor):
        """Ensure molecule is connected (spanning tree)"""
        
        if num_atoms <= 1:
            return
        
        # Find connected components
        connected = set()
        if edges:
            for edge in edges[::2]:  # Take every other (undirected)
                connected.update(edge)
        
        # Connect disconnected atoms
        for i in range(num_atoms):
            if i not in connected and current_valences[i] < self.valence_rules.get(atom_types[i].item(), 4):
                # Find closest connected atom that can bond
                closest_connected = None
                min_dist = float('inf')
                
                for j in connected:
                    if current_valences[j] < self.valence_rules.get(atom_types[j].item(), 4):
                        if closest_connected is None:
                            closest_connected = j
                            break
                
                if closest_connected is not None:
                    edges.extend([[i, closest_connected], [closest_connected, i]])
                    edge_types.extend([1, 1])
                    current_valences[i] += 1
                    current_valences[closest_connected] += 1
                    connected.add(i)
                elif connected:
                    # Force connection to any connected atom
                    j = next(iter(connected))
                    edges.extend([[i, j], [j, i]])
                    edge_types.extend([1, 1])
                    connected.add(i)
    
    def _validate_chemistry(self, mol_data: Dict) -> bool:
        """Validate chemical correctness"""
        
        atom_types = mol_data['atom_types']
        edge_index = mol_data['edge_index']
        
        if edge_index.size(1) == 0:
            return len(atom_types) == 1  # Single atom is OK
        
        # Check valence constraints
        valence_count = {i: 0 for i in range(len(atom_types))}
        
        for i in range(edge_index.size(1)):
            atom_idx = edge_index[0, i].item()
            if atom_idx in valence_count:
                valence_count[atom_idx] += 1
        
        # Validate each atom's valence
        for atom_idx, valence in valence_count.items():
            atom_type = atom_types[atom_idx].item()
            max_valence = self.valence_rules.get(atom_type, 4)
            
            if valence > max_valence:
                return False
        
        return True
    
    def molecules_to_smiles(self, molecules: List[Dict]) -> List[str]:
        """Convert molecules to chemically valid SMILES"""
        
        smiles_list = []
        
        for mol_data in molecules:
            try:
                smiles = self._mol_data_to_smiles(mol_data)
                smiles_list.append(smiles)
            except Exception as e:
                print(f"SMILES conversion error: {e}")
                smiles_list.append(None)
        
        return smiles_list
    
    def _mol_data_to_smiles(self, mol_data: Dict) -> Optional[str]:
        """Convert molecule data to SMILES with proper chemistry"""
        
        try:
            atom_types = mol_data['atom_types']
            edge_index = mol_data['edge_index']
            edge_attr = mol_data.get('edge_attr', None)
            
            # Create RDKit molecule
            mol = Chem.RWMol()
            
            # Add atoms with correct atomic numbers
            atom_map = {}
            for i, atom_type in enumerate(atom_types):
                atomic_num = int(atom_type.item())
                if atomic_num in self.valence_rules:
                    atom = Chem.Atom(atomic_num)
                    atom_idx = mol.AddAtom(atom)
                    atom_map[i] = atom_idx
                else:
                    # Fallback to carbon for unknown atoms
                    atom = Chem.Atom(6)
                    atom_idx = mol.AddAtom(atom)
                    atom_map[i] = atom_idx
            
            # Add bonds
            added_bonds = set()
            if edge_index.size(1) > 0:
                for i in range(edge_index.size(1)):
                    atom1 = edge_index[0, i].item()
                    atom2 = edge_index[1, i].item()
                    
                    # Skip if atoms not in map or bond already added
                    if atom1 not in atom_map or atom2 not in atom_map:
                        continue
                    
                    bond_key = tuple(sorted([atom1, atom2]))
                    if bond_key in added_bonds:
                        continue
                    
                    # Determine bond type
                    bond_type = Chem.BondType.SINGLE
                    if edge_attr is not None and i < edge_attr.size(0):
                        edge_val = int(edge_attr[i, 0].item())
                        if edge_val == 2:
                            bond_type = Chem.BondType.DOUBLE
                        elif edge_val == 3:
                            bond_type = Chem.BondType.TRIPLE
                    
                    try:
                        mol.AddBond(atom_map[atom1], atom_map[atom2], bond_type)
                        added_bonds.add(bond_key)
                    except Exception as e:
                        # Skip problematic bonds
                        continue
            
            # Convert to mol and sanitize
            mol = mol.GetMol()
            
            # Add hydrogens to satisfy valencies
            mol = Chem.AddHs(mol)
            
            # Sanitize molecule
            Chem.SanitizeMol(mol)
            
            # Generate SMILES
            smiles = Chem.MolToSmiles(mol)
            
            # Validate SMILES
            test_mol = Chem.MolFromSmiles(smiles)
            if test_mol is None:
                return None
            
            return smiles
            
        except Exception as e:
            print(f"Molecule conversion error: {e}")
            return None


# Usage example and integration
def create_chemical_generator(model=None, ddpm=None, device='cuda'):
    """Factory function to create chemical generator"""
    return ChemicallyValidGenerator(model, ddpm, device)


# Test function
def test_chemical_generator():
    """Test the chemical generator"""
    print("Testing Chemical Generator...")
    
    generator = ChemicallyValidGenerator()
    
    # Generate test molecules
    result = generator.generate_molecules(num_molecules=10, max_atoms=20)
    molecules = result['molecules']
    
    print(f"Generated {len(molecules)} molecules")
    
    # Convert to SMILES
    smiles_list = generator.molecules_to_smiles(molecules)
    valid_smiles = [s for s in smiles_list if s is not None]
    
    print(f"Valid SMILES: {len(valid_smiles)}/{len(smiles_list)}")
    
    # Analyze molecules
    if valid_smiles:
        for i, smiles in enumerate(valid_smiles[:5]):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mw = Descriptors.MolWt(mol)
                atoms = mol.GetNumAtoms()
                print(f"  {i+1}: {smiles} (MW: {mw:.1f}, Atoms: {atoms})")
    
    return valid_smiles


if __name__ == "__main__":
    test_chemical_generator()