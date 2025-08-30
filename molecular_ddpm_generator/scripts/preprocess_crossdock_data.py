# scripts/preprocess_crossdock_generation.py - DiffSBDD style preprocessing
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

    def __init__(self, data_dir="molecular_ddpm_generator/data/crossdocked_pocket10", output_dir="molecular_ddpm_generator/data/processed", 
                 max_samples=None, force_new_split=False):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        self.force_new_split = force_new_split  # Flag để force tạo split mới
        
        # ========== ONE-HOT ENCODING DICTIONARIES (như DiffSBDD) ==========
        
        # Ligand atom types - ONE-HOT ENCODED
        self.atom_types = {
            'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 'Cl': 5, 
            'Br': 6, 'I': 7, 'P': 8, 'B': 9, 'UNK': 10
        }
        self.atom_nf = len(self.atom_types)  # 11 features
        
        # Pocket residue types - ONE-HOT ENCODED
        self.residue_types = {
            'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
            'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
            'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
            'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
            'UNK': 20
        }
        self.residue_nf = len(self.residue_types)  # 21 features
        
        # Bond types for edge features
        self.bond_types = {
            Chem.BondType.SINGLE: 0,
            Chem.BondType.DOUBLE: 1, 
            Chem.BondType.TRIPLE: 2,
            Chem.BondType.AROMATIC: 3
        }
        
        print(f"Atom features (one-hot): {self.atom_nf}")
        print(f"Residue features (one-hot): {self.residue_nf}")
    
    def load_splits(self, force_new_split=False):
        """Load train/test splits với option tạo split mới 80:20"""
        split_file = Path("data/split_80_20.pt") 
        
        if split_file.exists() and not force_new_split:
            print("Loading existing 80:20 splits...")
            try:
                splits = torch.load(split_file)
                if 'train' in splits and 'test' in splits:
                    print(f"Found 80:20 splits: train={len(splits['train'])}, test={len(splits['test'])}")
                    test_ratio = len(splits['test']) / (len(splits['train']) + len(splits['test']))
                    print(f"Test ratio: {test_ratio:.3f}")
                    return splits
                else:
                    print("Invalid split format, creating new 80:20 splits...")
            except Exception as e:
                print(f"Error loading splits: {e}, creating new 80:20 splits...")
        
        print("Creating NEW 80:20 train/test splits from index.pkl...")
        return self.create_train_test_splits()
    
    def create_train_test_splits(self):
        """Create NEW 80:20 train/test splits"""
        index_file = self.data_dir / "index.pkl"
        if not index_file.exists():
            print(f"index.pkl not found at {index_file}")
            return None
        
        try:
            with open(index_file, 'rb') as f:
                index_data = pickle.load(f)
            
            print(f"📊 Found {len(index_data)} total entries in index")
            
            np.random.seed(42)  # Different seed for different split
            indices = list(range(len(index_data)))
            np.random.shuffle(indices)
            
            n_total = len(indices)
            n_train = int(0.8 * n_total)
            n_test = n_total - n_train
            
            splits = {
                'train': [index_data[i] for i in indices[:n_train]],
                'test': [index_data[i] for i in indices[n_train:]]
            }
            
            print(f"Created NEW 80:20 splits:")
            print(f"Train: {len(splits['train'])} samples ({len(splits['train'])/n_total*100:.1f}%)")
            print(f"Test:  {len(splits['test'])} samples ({len(splits['test'])/n_total*100:.1f}%)")
            
            # Save với tên file mới
            split_file = Path("data/split_80_20.pt")
            torch.save(splits, split_file)
            print(f"New 80:20 splits saved to {split_file}")
            
            return splits
            
        except Exception as e:
            print(f"Error creating splits: {e}")
            return None
    
    def process_complex(self, entry):
        """Process complex with DiffSBDD-style one-hot features"""
        
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
        
        # Process ligand với ONE-HOT features
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
                
            ligand_data = self.mol_to_onehot_features(mol)
            if ligand_data is None:
                return None
        except Exception as e:
            return None
        
        # Process pocket với ONE-HOT residue features 
        pocket_data = None
        if pocket_path and pocket_path.exists():
            pocket_data = self.pdb_to_onehot_features(pocket_path)
        
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
    
    def mol_to_onehot_features(self, mol):
        if mol is None:
            return None
        
        # Remove hydrogens for consistency với DiffSBDD
        mol = Chem.RemoveHs(mol)
        
        # Generate 3D coordinates if needed
        if mol.GetNumConformers() == 0:
            try:
                mol_with_h = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol_with_h, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol_with_h)
                mol = Chem.RemoveHs(mol_with_h)
            except:
                return None
        
        # Size filter
        if mol.GetNumAtoms() < 5 or mol.GetNumAtoms() > 50:
            return None
        
        conf = mol.GetConformer()
        
        atom_features = []
        positions = []
        
        for atom in mol.GetAtoms():
            # Get atom type
            atom_type = atom.GetSymbol()
            atom_type_idx = self.atom_types.get(atom_type, self.atom_types['UNK'])
            
            # ONE-HOT ENCODING for atom type
            atom_onehot = np.zeros(self.atom_nf, dtype=np.float32)
            atom_onehot[atom_type_idx] = 1.0
            
            atom_features.append(atom_onehot)
            
            # 3D coordinates 
            pos = conf.GetAtomPosition(atom.GetIdx())
            positions.append([pos.x, pos.y, pos.z])
        
        edge_index = []
        edge_features = []
        
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = self.bond_types.get(bond.GetBondType(), 0)
            
            # Simple bond features
            bond_feature = [
                bond_type,                        # Bond type ID
                int(bond.GetIsConjugated()),     # Conjugated
                int(bond.IsInRing()),            # In ring
            ]
            
            # Add both directions (undirected graph)
            edge_index.extend([[start, end], [end, start]])
            edge_features.extend([bond_feature, bond_feature])
        
        # Validate
        if not self._is_valid_molecule(mol):
            return None
        
        return {
            'atom_features': np.array(atom_features, dtype=np.float32),  # [N, atom_nf] - ONE-HOT
            'positions': np.array(positions, dtype=np.float32),          # [N, 3]
            'edge_index': np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64),
            'edge_features': np.array(edge_features, dtype=np.float32) if edge_features else np.zeros((0, 3), dtype=np.float32),
            'smiles': Chem.MolToSmiles(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds(),
            'molecular_weight': Chem.rdMolDescriptors.CalcExactMolWt(mol),
        }
    
    def pdb_to_onehot_features(self, pdb_file):
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('pocket', pdb_file)
            
            residue_features = []
            residue_positions = []
            residue_sequence = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        # Skip non-standard residues
                        if residue.id[0] != ' ':
                            continue
                            
                        res_name = residue.get_resname()
                        
                        # Get Cα atom (DiffSBDD uses Cα-only representation)
                        if 'CA' in residue:
                            ca_atom = residue['CA']
                            
                            # Get residue type
                            res_type_idx = self.residue_types.get(res_name, self.residue_types['UNK'])
                            
                            # ========== ONE-HOT RESIDUE FEATURES ==========
                            residue_onehot = np.zeros(self.residue_nf, dtype=np.float32)
                            residue_onehot[res_type_idx] = 1.0
                            
                            residue_features.append(residue_onehot)
                            
                            # Cα coordinates
                            coord = ca_atom.coord
                            residue_positions.append([coord[0], coord[1], coord[2]])
                            
                            # Store sequence
                            residue_sequence.append(res_name)
            
            if len(residue_features) == 0:
                return None
            
            # Generate Cα-Cα connectivity
            positions_array = np.array(residue_positions)
            edge_index, edge_features = self._compute_ca_connectivity(positions_array)
            
            return {
                'atom_features': np.array(residue_features, dtype=np.float32),   # [N_res, residue_nf] - ONE-HOT
                'positions': np.array(residue_positions, dtype=np.float32),      # [N_res, 3]
                'edge_index': edge_index,                                        # [2, E]
                'edge_features': edge_features,                                  # [E, 1]
                'residue_sequence': residue_sequence,
                'num_residues': len(residue_sequence),
            }
            
        except Exception as e:
            return None
    
    def _is_valid_molecule(self, mol):
        """Validate molecule for generation"""
        # Check basic validity
        if mol.GetNumAtoms() < 5 or mol.GetNumAtoms() > 50:  
            return False
        
        # Check for too many unknown atoms
        unknown_count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in self.atom_types:
                unknown_count += 1
        
        if unknown_count > mol.GetNumAtoms() * 0.2:  # Max 20% unknown atoms
            return False
        
        return True
    
    def _compute_ca_connectivity(self, positions):
        """Compute Cα-Cα connectivity for pocket representation"""
        if len(positions) <= 1:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 1), dtype=np.float32)
        
        # Distance-based connectivity (DiffSBDD style)
        distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
        
        edge_index = []
        edge_features = []
        
        # Connect residues within 10 Å (standard pocket cutoff)
        cutoff = 10.0
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if distances[i, j] <= cutoff:
                    edge_index.extend([[i, j], [j, i]])
                    edge_features.extend([[distances[i, j]/cutoff], [distances[i, j]/cutoff]])  # Normalized distance
        
        if edge_index:
            return (np.array(edge_index, dtype=np.int64).T, 
                   np.array(edge_features, dtype=np.float32))
        else:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 1), dtype=np.float32)
    
    def create_size_histogram(self, processed_data):
        """Tạo size histogram cho conditional generation như DiffSBDD"""
        
        print("📊 Creating size histogram for conditional generation...")
        
        size_pairs = []
        
        for data in processed_data:
            if 'ligand' in data and 'pocket' in data:
                n_lig = data['ligand']['num_atoms']
                n_pocket = data['pocket']['num_residues'] 
                size_pairs.append((n_lig, n_pocket))
        
        if not size_pairs:
            return None
        
        # Create 2D histogram
        max_lig = max(pair[0] for pair in size_pairs)
        max_pocket = max(pair[1] for pair in size_pairs)
        
        # Initialize histogram
        histogram = np.zeros((max_lig + 1, max_pocket + 1))
        
        # Fill histogram
        for n_lig, n_pocket in size_pairs:
            histogram[n_lig, n_pocket] += 1
        
        # Normalize by pocket size (conditional P(n_lig | n_pocket))
        for j in range(max_pocket + 1):
            total = histogram[:, j].sum()
            if total > 0:
                histogram[:, j] /= total
        
        print(f"✅ Size histogram shape: {histogram.shape}")
        print(f"   Ligand atoms: 0-{max_lig}")
        print(f"   Pocket residues: 0-{max_pocket}")
        
        return histogram.tolist()  # Convert to list for JSON serialization
    
    def save_dataset_info(self, all_processed_data):
        """Save dataset information như DiffSBDD"""
        
        # Create size histogram
        size_histogram = self.create_size_histogram(all_processed_data)
        
        # Dataset statistics
        dataset_info = {
            'atom_nf': self.atom_nf,           # Number of atom features (one-hot)
            'residue_nf': self.residue_nf,     # Number of residue features (one-hot)
            'atom_types': self.atom_types,
            'residue_types': self.residue_types,
            'bond_types': self.bond_types,
            'size_histogram': size_histogram,
            'total_complexes': len(all_processed_data),
        }
        
        # Save dataset info
        info_file = self.output_dir / 'dataset_info.pkl'
        with open(info_file, 'wb') as f:
            pickle.dump(dataset_info, f)
        
        print(f"💾 Dataset info saved to {info_file}")
        return dataset_info
    
    def process_dataset(self):
        """Process dataset with DiffSBDD-style one-hot features"""
        print("🧬 Processing CrossDock dataset with ONE-HOT encoding (DiffSBDD style)...")        
        
        # Load splits với option force new split
        splits = self.load_splits(force_new_split=self.force_new_split)
        if splits is None:
            print("❌ Failed to load or create splits!")
            return
        
        all_processed_data = []
        
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
                    all_processed_data.append(complex_data)
                else:
                    failed_count += 1
            
            print(f"✅ Processed {len(processed_data)} complexes, failed: {failed_count}")
            
            # Save processed data
            if processed_data:
                output_file = self.output_dir / f"{split_name}.pkl"
                with open(output_file, 'wb') as f:
                    pickle.dump(processed_data, f)
                print(f"💾 ONE-HOT features saved to {output_file}")
                
                # Print feature summary
                self._print_feature_summary(processed_data, split_name)
            else:
                print(f"❌ No valid data for {split_name}")
        
        # Save dataset info với size histogram
        if all_processed_data:
            self.save_dataset_info(all_processed_data)
        
        print("\n✅ DiffSBDD-style preprocessing completed!")
        print("🔥 Features are now ONE-HOT encoded for conditional generation!")
    
    def _print_feature_summary(self, processed_data, split_name):
        """Print feature summary"""
        print(f"\n📊 {split_name.upper()} Feature Summary (ONE-HOT):")
        
        # Ligand statistics
        ligand_atoms = [d['ligand']['num_atoms'] for d in processed_data if 'ligand' in d]
        ligand_bonds = [d['ligand']['num_bonds'] for d in processed_data if 'ligand' in d]
        
        if ligand_atoms:
            print(f"   🧪 Ligands: {len(ligand_atoms)} molecules")
            print(f"     Atoms: {np.mean(ligand_atoms):.1f} ± {np.std(ligand_atoms):.1f}")
            print(f"     Range: {np.min(ligand_atoms)} - {np.max(ligand_atoms)} atoms")
            print(f"     Feature shape: [N_atoms, {self.atom_nf}] (ONE-HOT)")
        
        # Pocket statistics
        pocket_residues = [d['pocket']['num_residues'] for d in processed_data if 'pocket' in d]
        
        if pocket_residues:
            print(f"   🧬 Pockets: {len(pocket_residues)} pockets")
            print(f"     Residues: {np.mean(pocket_residues):.1f} ± {np.std(pocket_residues):.1f}")
            print(f"     Range: {np.min(pocket_residues)} - {np.max(pocket_residues)} residues")
            print(f"     Feature shape: [N_residues, {self.residue_nf}] (ONE-HOT)")
        
def main():
    parser = argparse.ArgumentParser(description='DiffSBDD-style CrossDock preprocessing with ONE-HOT encoding')
    parser.add_argument('--data_dir', type=str, default='data/crossdocked_pocket10',
                       help='Path to crossdocked_pocket10 data')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples per split (for testing)')
    parser.add_argument('--force_new_split', action='store_true',
                       help='Force create new 80:20 split even if exists')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not Path(args.data_dir).exists():
        print(f"Data directory not found: {args.data_dir}")
        print("Please make sure crossdocked_pocket10 is in the correct location")
        return
    
    print(f"🧬 DiffSBDD-style preprocessing with ONE-HOT encoding")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    if args.force_new_split:
        print(f"🔄 Force creating NEW 80:20 split")
    
    preprocessor = GenerationFocusedPreprocessor(
        data_dir=args.data_dir, 
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        force_new_split=args.force_new_split
    )
    
    preprocessor.process_dataset()

if __name__ == "__main__":
    main()