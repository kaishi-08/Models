# scripts/preprocess_crossdock_generation.py - Generation-focused preprocessing
import os
import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings('ignore')

class GenerationFocusedPreprocessor:

    def __init__(self, data_dir="data/crossdocked_pocket10", output_dir="data/processed", 
                 max_samples=None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        
        # Limited to most common drug-like atoms for stable generation
        self.atom_types = {
            'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 'Cl': 5, 
            'Br': 6, 'I': 7, 'P': 8, 'UNK': 9
        }
        
        # Generation-focused bond types (critical for molecular validity)
        self.bond_types = {
            Chem.BondType.SINGLE: 0,
            Chem.BondType.DOUBLE: 1, 
            Chem.BondType.TRIPLE: 2,
            Chem.BondType.AROMATIC: 3
        }
        
        # Pocket residue types (Cα representation like DiffSBDD)
        self.residue_types = {
            'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
            'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
            'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
            'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
            'UNK': 20
        }
    
    def load_splits(self):
        """Load train/test splits"""
        split_file = Path("data/split_by_name.pt")
        
        if split_file.exists():
            print("Loading existing splits...")
            try:
                splits = torch.load(split_file)
                if 'train' in splits and 'test' in splits:
                    print(f"Found train/test splits: train={len(splits['train'])}, test={len(splits['test'])}")
                    return splits
                else:
                    print("Invalid split format, creating new splits...")
            except Exception as e:
                print(f"Error loading splits: {e}, creating new splits...")
        
        print("🔧 Creating train/test splits from index.pkl...")
        return self.create_train_test_splits()
    
    def create_train_test_splits(self):
        """Create train/test splits"""
        index_file = self.data_dir / "index.pkl"
        if not index_file.exists():
            print(f"index.pkl not found at {index_file}")
            return None
        
        try:
            with open(index_file, 'rb') as f:
                index_data = pickle.load(f)
            
            print(f"📊 Found {len(index_data)} entries in index")
            
            np.random.seed(42)
            indices = list(range(len(index_data)))
            np.random.shuffle(indices)
            
            n_total = len(indices)
            n_train = int(0.8 * n_total)
            
            splits = {
                'train': [index_data[i] for i in indices[:n_train]],
                'test': [index_data[i] for i in indices[n_train:]]
            }
            
            print(f"✅ Created splits: train={len(splits['train'])}, test={len(splits['test'])}")
            
            split_file = Path("data/split_by_name.pt")
            torch.save(splits, split_file)
            print(f"💾 Splits saved to {split_file}")
            
            return splits
            
        except Exception as e:
            print(f"❌ Error creating splits: {e}")
            return None
    
    def process_complex(self, entry):
        """Process complex with generation-focused features"""
        
        # Handle entry formats
        if len(entry) == 2:
            pocket_file, ligand_file = entry
            receptor_file = None
            score = 0.0
        elif len(entry) >= 4:
            pocket_file, ligand_file, receptor_file, score = entry[:4]
        else:
            return None
        
        ligand_path = self.data_dir / ligand_file
        pocket_path = self.data_dir / pocket_file
        
        # Check ligand file
        if not ligand_path.exists():
            parent_dir = ligand_path.parent
            if parent_dir.exists():
                ligand_files = list(parent_dir.glob("*.sdf"))
                if ligand_files:
                    ligand_path = ligand_files[0]
                else:
                    return None
            else:
                return None
        
        # Check pocket file
        if not pocket_path.exists():
            parent_dir = pocket_path.parent
            if parent_dir.exists():
                pocket_files = list(parent_dir.glob("*pocket*.pdb"))
                if pocket_files:
                    pocket_path = pocket_files[0]
                else:
                    pocket_path = None
            else:
                pocket_path = None
        
        # Process ligand with generation-focused features
        try:
            if ligand_path.suffix.lower() == '.sdf':
                supplier = Chem.SDMolSupplier(str(ligand_path))
                mol = None
                for m in supplier:
                    if m is not None:
                        mol = m
                        break
                if mol is None:
                    return None
            elif ligand_path.suffix.lower() in ['.mol', '.mol2']:
                mol = Chem.MolFromMolFile(str(ligand_path))
            else:
                return None
                
            ligand_data = self.mol_to_generation_features(mol)
            if ligand_data is None:
                return None
        except Exception as e:
            return None
        
        # Process pocket with Cα representation 
        pocket_data = None
        if pocket_path and pocket_path.exists():
            pocket_data = self.pdb_to_ca_features(pocket_path)
        
        complex_data = {
            'pocket_file': str(pocket_file),
            'ligand_file': str(ligand_file),
            'receptor_file': str(receptor_file) if receptor_file else None,
            'score': float(score),
            'ligand': ligand_data
        }
        
        if pocket_data:
            complex_data['pocket'] = pocket_data
        
        return complex_data
    
    def mol_to_generation_features(self, mol):
        if mol is None:
            return None
        
        # Generate 3D coordinates if needed
        if mol.GetNumConformers() == 0:
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                return None
        
        conf = mol.GetConformer()
        
        # GENERATION-FOCUSED ATOM FEATURES (minimal but essential)
        atom_features = []
        positions = []
        
        for atom in mol.GetAtoms():
            # Core features for generation (based on DiffSBDD)
            atom_type = self.atom_types.get(atom.GetSymbol(), self.atom_types['UNK'])
            
            # Minimal but essential features for valid molecular generation
            features = [
                atom_type,                       # 0: Atom type (most important)
                atom.GetDegree(),                # 1: Degree (connectivity)
                atom.GetFormalCharge() + 2,      # 2: Formal charge (shifted to be positive)
                int(atom.GetIsAromatic()),       # 3: Aromatic (important for ring systems)
                atom.GetTotalNumHs(),            # 4: Hydrogen count (valence)
                int(atom.IsInRing()),            # 5: In ring (structural constraint)
            ]
            
            atom_features.append(features)
            
            # 3D coordinates 
            pos = conf.GetAtomPosition(atom.GetIdx())
            positions.append([pos.x, pos.y, pos.z])
        
        # GENERATION-FOCUSED BOND FEATURES
        edge_index = []
        edge_features = []
        
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = self.bond_types.get(bond.GetBondType(), 0)
            
            # Minimal bond features for generation
            bond_feature = [
                bond_type,                        # 0: Bond type (essential)
                int(bond.GetIsConjugated()),     # 1: Conjugated (aromaticity)
                int(bond.IsInRing()),            # 2: In ring (constraints)
            ]
            
            # Add both directions (undirected graph)
            edge_index.extend([[start, end], [end, start]])
            edge_features.extend([bond_feature, bond_feature])
        
        # Validate molecular structure
        if not self._is_valid_molecule(mol, atom_features, edge_index):
            return None
        
        return {
            'atom_features': np.array(atom_features, dtype=np.float32),  # [N, 6]
            'positions': np.array(positions, dtype=np.float32),          # [N, 3]
            'edge_index': np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64),
            'edge_features': np.array(edge_features, dtype=np.float32) if edge_features else np.zeros((0, 3), dtype=np.float32),
            'smiles': Chem.MolToSmiles(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds(),
            'molecular_weight': Chem.rdMolDescriptors.CalcExactMolWt(mol),
        }
    
    def pdb_to_ca_features(self, pdb_file):
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('pocket', pdb_file)
            
            ca_positions = []
            ca_features = []
            residue_sequence = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        res_name = residue.get_resname()
                        res_id = residue.get_id()[1]
                        
                        # Get Cα atom (like DiffSBDD)
                        if 'CA' in residue:
                            ca_atom = residue['CA']
                            
                            # Residue-level features (generation-focused)
                            res_type = self.residue_types.get(res_name, self.residue_types['UNK'])
                            
                            features = [
                                res_type,                        # 0: Residue type
                                res_id % 100,                   # 1: Residue position (mod 100)
                                int(self._is_hydrophobic(res_name)),   # 2: Hydrophobic
                                int(self._is_charged(res_name)),       # 3: Charged
                                int(self._is_polar(res_name)),         # 4: Polar
                                int(self._is_aromatic(res_name)),      # 5: Aromatic
                                ca_atom.bfactor / 50.0,               # 6: B-factor (flexibility)
                            ]
                            
                            ca_features.append(features)
                            
                            # Cα coordinates
                            coord = ca_atom.coord
                            ca_positions.append([coord[0], coord[1], coord[2]])
                            
                            # Store sequence
                            residue_sequence.append(res_name)
            
            if len(ca_features) == 0:
                return None
            
            # Generate Cα-Cα connectivity (distance-based)
            positions_array = np.array(ca_positions)
            edge_index, edge_features = self._compute_ca_connectivity(positions_array)
            
            return {
                'atom_features': np.array(ca_features, dtype=np.float32),    # [N_res, 7]
                'positions': np.array(ca_positions, dtype=np.float32),       # [N_res, 3]
                'edge_index': edge_index,                                     # [2, E]
                'edge_features': edge_features,                               # [E, 1]
                'residue_sequence': residue_sequence,
                'num_residues': len(residue_sequence),
            }
            
        except Exception as e:
            return None
    
    def _is_valid_molecule(self, mol, atom_features, edge_index):
        """Validate molecule for generation"""
        # Check basic validity
        if mol.GetNumAtoms() < 5 or mol.GetNumAtoms() > 50:  # Reasonable size
            return False
        
        # Check connectivity
        if len(edge_index) == 0 and mol.GetNumAtoms() > 1:
            return False
        
        # Check for valid atom types
        for features in atom_features:
            if features[0] == self.atom_types['UNK']:  # Too many unknown atoms
                return False
        
        return True
    
    def _is_hydrophobic(self, res_name):
        """Check if residue is hydrophobic"""
        hydrophobic = {'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO'}
        return res_name in hydrophobic
    
    def _is_charged(self, res_name):
        """Check if residue is charged"""
        charged = {'ARG', 'LYS', 'ASP', 'GLU', 'HIS'}
        return res_name in charged
    
    def _is_polar(self, res_name):
        """Check if residue is polar"""
        polar = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS'}
        return res_name in polar
    
    def _is_aromatic(self, res_name):
        """Check if residue is aromatic"""
        aromatic = {'PHE', 'TRP', 'TYR', 'HIS'}
        return res_name in aromatic
    
    def _compute_ca_connectivity(self, positions):
        """Compute Cα-Cα connectivity for pocket representation"""
        if len(positions) <= 1:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 1), dtype=np.float32)
        
        # Distance-based connectivity (like DiffSBDD)
        distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
        
        edge_index = []
        edge_features = []
        
        # Connect residues within 10 Å (standard pocket cutoff)
        cutoff = 10.0
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if distances[i, j] <= cutoff:
                    edge_index.extend([[i, j], [j, i]])
                    edge_features.extend([[distances[i, j]], [distances[i, j]]])
        
        if edge_index:
            return (np.array(edge_index, dtype=np.int64).T, 
                   np.array(edge_features, dtype=np.float32))
        else:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 1), dtype=np.float32)
    
    def process_dataset(self):
        """Process dataset with generation-focused features"""
        print("Processing CrossDock dataset for molecular generation...")        
        # Load splits
        splits = self.load_splits()
        if splits is None:
            print("❌ Failed to load or create splits!")
            return
        
        # Process train and test
        for split_name in ['train', 'test']:
            if split_name not in splits:
                print(f"❌ Missing {split_name} split!")
                continue
                
            entries = splits[split_name]
            print(f"\n📊 Processing {split_name} split ({len(entries)} entries)...")
            
            processed_data = []
            failed_count = 0
            
            # Determine max samples
            if self.max_samples:
                if split_name == 'train':
                    max_for_split = min(self.max_samples, len(entries))
                else:
                    max_for_split = min(self.max_samples // 4, len(entries))
                print(f"⚠️  Limited to {max_for_split} samples for testing")
            else:
                max_for_split = len(entries)
            
            for i, entry in enumerate(tqdm(entries, desc=f"Processing {split_name}")):
                if i >= max_for_split:
                    break
                    
                complex_data = self.process_complex(entry)
                if complex_data:
                    processed_data.append(complex_data)
                else:
                    failed_count += 1
            
            print(f"✅ Processed {len(processed_data)} complexes, failed: {failed_count}")
            
            # Save processed data
            if processed_data:
                output_file = self.output_dir / f"{split_name}_generation.pkl"
                with open(output_file, 'wb') as f:
                    pickle.dump(processed_data, f)
                print(f"💾 Generation features saved to {output_file}")
                
                # Print feature summary
                self._print_feature_summary(processed_data, split_name)
            else:
                print(f"❌ No valid data for {split_name}")
        
        print("\n Generation-focused preprocessing completed!")
    
    def _print_feature_summary(self, processed_data, split_name):
        """Print feature summary"""
        print(f"\n📊 {split_name.upper()} Feature Summary:")
        
        # Ligand statistics
        ligand_atoms = [d['ligand']['num_atoms'] for d in processed_data if 'ligand' in d]
        ligand_bonds = [d['ligand']['num_bonds'] for d in processed_data if 'ligand' in d]
        
        if ligand_atoms:
            print(f"   Ligands: {len(ligand_atoms)} molecules")
            print(f"     Atoms: {np.mean(ligand_atoms):.1f} ± {np.std(ligand_atoms):.1f} (avg ± std)")
            print(f"     Range: {np.min(ligand_atoms)} - {np.max(ligand_atoms)} atoms")
            print(f"     Bonds: {np.mean(ligand_bonds):.1f} ± {np.std(ligand_bonds):.1f} (avg ± std)")
        
        # Pocket statistics
        pocket_residues = [d['pocket']['num_residues'] for d in processed_data if 'pocket' in d]
        
        if pocket_residues:
            print(f"   Pockets: {len(pocket_residues)} pockets")
            print(f"     Residues: {np.mean(pocket_residues):.1f} ± {np.std(pocket_residues):.1f} (avg ± std)")
            print(f"     Range: {np.min(pocket_residues)} - {np.max(pocket_residues)} residues")
        
def main():
    parser = argparse.ArgumentParser(description='Generation-focused CrossDock preprocessing')
    parser.add_argument('--data_dir', type=str, default='data/crossdocked_pocket10',
                       help='Path to crossdocked_pocket10 data')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples per split (for testing)')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not Path(args.data_dir).exists():
        print(f"Data directory not found: {args.data_dir}")
        print("Please make sure crossdocked_pocket10 is in the correct location")
        return
    
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    
    preprocessor = GenerationFocusedPreprocessor(
        data_dir=args.data_dir, 
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )
    
    preprocessor.process_dataset()

if __name__ == "__main__":
    main()