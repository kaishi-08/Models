# scripts/preprocess_crossdock_data.py - Skip molecules with valence errors
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

# Suppress RDKit warnings completely
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class GenerationFocusedPreprocessor:

    def __init__(self, data_dir="data/crossdocked_pocket10", output_dir="data/processed", 
                 max_samples=None, force_new_split=False):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        self.force_new_split = force_new_split
        
        # Ligand atom types - ONE-HOT ENCODED
        self.atom_types = {
            'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 'Cl': 5, 
            'Br': 6, 'I': 7, 'P': 8, 'B': 9, 'UNK': 10
        }
        self.atom_nf = len(self.atom_types)
        
        # Pocket residue types - ONE-HOT ENCODED
        self.residue_types = {
            'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
            'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
            'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
            'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
            'UNK': 20
        }
        self.residue_nf = len(self.residue_types)
        
        # Bond types for edge features
        self.bond_types = {
            Chem.BondType.SINGLE: 0,
            Chem.BondType.DOUBLE: 1, 
            Chem.BondType.TRIPLE: 2,
            Chem.BondType.AROMATIC: 3
        }
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'valence_errors': 0,
            'sanitization_errors': 0,
            'coord_errors': 0,
            'size_filtered': 0,
            'successful': 0
        }
        
        print(f"Atom features: {self.atom_nf}")
        print(f"Residue features: {self.residue_nf}")
    
    def load_splits(self, force_new_split=False):
        split_file = Path("data/split_80_20.pt") 
        
        if split_file.exists() and not force_new_split:
            print("Loading existing splits...")
            try:
                splits = torch.load(split_file)
                if 'train' in splits and 'test' in splits:
                    print(f"Found splits: train={len(splits['train'])}, test={len(splits['test'])}")
                    return splits
            except Exception as e:
                print(f"Error loading splits: {e}")
        
        print("Creating new splits...")
        return self.create_train_test_splits()
    
    def create_train_test_splits(self):
        index_file = self.data_dir / "index.pkl"
        if not index_file.exists():
            print(f"index.pkl not found at {index_file}")
            return None
        
        try:
            with open(index_file, 'rb') as f:
                index_data = pickle.load(f)
            
            print(f"Found {len(index_data)} entries")
            
            np.random.seed(42)
            indices = list(range(len(index_data)))
            np.random.shuffle(indices)
            
            n_total = len(indices)
            n_train = int(0.8 * n_total)
            
            splits = {
                'train': [index_data[i] for i in indices[:n_train]],
                'test': [index_data[i] for i in indices[n_train:]]
            }
            split_file = Path("data/split_80_20.pt")
            torch.save(splits, split_file)
            print(f"Splits saved: train={len(splits['train'])}, test={len(splits['test'])}")
            
            return splits
            
        except Exception as e:
            print(f"Error creating splits: {e}")
            return None
    
    def process_complex(self, entry):
        """Process complex - skip invalid molecules"""
        try:
            self.stats['total_processed'] += 1
            
            if len(entry) == 2:
                pocket_file, ligand_file = entry
                receptor_file = None
                score = 0.0
            elif len(entry) == 4:
                pocket_file, ligand_file, receptor_file, score = entry
            else:
                return None
            
            ligand_path = self.data_dir / ligand_file
            pocket_path = self.data_dir / pocket_file
            
            if not ligand_path.exists():
                return None
            
            # Process ligand - skip if any errors
            ligand_data = self.process_molecule_safe(ligand_path)
            if ligand_data is None:
                return None
            
            # Process pocket
            pocket_data = None
            if pocket_path.exists():
                pocket_data = self.process_pocket_safe(pocket_path)
            
            complex_data = {
                'pocket_file': str(pocket_file),
                'ligand_file': str(ligand_file), 
                'receptor_file': str(receptor_file) if receptor_file else None,
                'score': float(score),
                'ligand': ligand_data
            }
            
            if pocket_data:
                complex_data['pocket'] = pocket_data
            
            self.stats['successful'] += 1
            return complex_data
        
        except:
            return None
    
    def process_molecule_safe(self, ligand_path):
        """Process molecule with strict validation - return None if any issues"""
        try:
            # Read molecule
            if ligand_path.suffix.lower() != '.sdf':
                return None
                
            supplier = Chem.SDMolSupplier(str(ligand_path), sanitize=False, removeHs=False)
            mol = None
            for m in supplier:
                if m is not None:
                    mol = m
                    break
            
            if mol is None:
                return None
            
            # Test sanitization - skip molecule if ANY errors
            try:
                Chem.SanitizeMol(mol)
            except Chem.AtomValenceException:
                self.stats['valence_errors'] += 1
                return None  # Skip molecules with valence errors
            except Chem.KekulizeException:
                self.stats['sanitization_errors'] += 1
                return None  # Skip kekulization errors
            except Exception:
                self.stats['sanitization_errors'] += 1
                return None  # Skip any other sanitization errors
            
            # Remove hydrogens
            mol = Chem.RemoveHs(mol)
            
            # Size filter
            if mol.GetNumAtoms() < 5 or mol.GetNumAtoms() > 50:
                self.stats['size_filtered'] += 1
                return None
            
            # Generate 3D coordinates
            if mol.GetNumConformers() == 0:
                try:
                    mol_with_h = Chem.AddHs(mol)
                    if AllChem.EmbedMolecule(mol_with_h, randomSeed=42) == -1:
                        self.stats['coord_errors'] += 1
                        return None
                    AllChem.MMFFOptimizeMolecule(mol_with_h)
                    mol = Chem.RemoveHs(mol_with_h)
                except:
                    self.stats['coord_errors'] += 1
                    return None
            
            # Get conformer
            conf = mol.GetConformer()
            
            # Extract features
            atom_features = []
            positions = []
            
            for atom in mol.GetAtoms():
                atom_type = atom.GetSymbol()
                atom_type_idx = self.atom_types.get(atom_type, self.atom_types['UNK'])
                
                # ONE-HOT encoding
                atom_onehot = np.zeros(self.atom_nf, dtype=np.float32)
                atom_onehot[atom_type_idx] = 1.0
                atom_features.append(atom_onehot)
                
                # 3D coordinates
                pos = conf.GetAtomPosition(atom.GetIdx())
                positions.append([pos.x, pos.y, pos.z])
            
            # Extract bonds
            edge_index = []
            edge_features = []
            
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bond_type = self.bond_types.get(bond.GetBondType(), 0)
                
                bond_feature = [
                    bond_type,
                    int(bond.GetIsConjugated()),
                    int(bond.IsInRing()),
                ]
                
                edge_index.extend([[start, end], [end, start]])
                edge_features.extend([bond_feature, bond_feature])
            
            return {
                'atom_features': np.array(atom_features, dtype=np.float32),
                'positions': np.array(positions, dtype=np.float32),
                'edge_index': np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64),
                'edge_features': np.array(edge_features, dtype=np.float32) if edge_features else np.zeros((0, 3), dtype=np.float32),
                'smiles': Chem.MolToSmiles(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'molecular_weight': Chem.rdMolDescriptors.CalcExactMolWt(mol),
            }
            
        except:
            return None
    
    def process_pocket_safe(self, pdb_file):
        """Process pocket safely"""
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('pocket', pdb_file)
            
            residue_features = []
            residue_positions = []
            residue_sequence = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] != ' ':
                            continue
                            
                        res_name = residue.get_resname()
                        
                        if 'CA' in residue:
                            ca_atom = residue['CA']
                            
                            res_type_idx = self.residue_types.get(res_name, self.residue_types['UNK'])
                            
                            residue_onehot = np.zeros(self.residue_nf, dtype=np.float32)
                            residue_onehot[res_type_idx] = 1.0
                            
                            residue_features.append(residue_onehot)
                            
                            coord = ca_atom.coord
                            residue_positions.append([coord[0], coord[1], coord[2]])
                            
                            residue_sequence.append(res_name)
            
            if len(residue_features) == 0:
                return None
            
            positions_array = np.array(residue_positions)
            edge_index, edge_features = self._compute_ca_connectivity(positions_array)
            
            return {
                'atom_features': np.array(residue_features, dtype=np.float32),
                'positions': np.array(residue_positions, dtype=np.float32),
                'edge_index': edge_index,
                'edge_features': edge_features,
                'residue_sequence': residue_sequence,
                'num_residues': len(residue_sequence),
            }
            
        except:
            return None
    
    def _compute_ca_connectivity(self, positions):
        """Compute CA-CA connectivity"""
        if len(positions) <= 1:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 1), dtype=np.float32)
        
        distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
        
        edge_index = []
        edge_features = []
        cutoff = 10.0
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if distances[i, j] <= cutoff:
                    edge_index.extend([[i, j], [j, i]])
                    edge_features.extend([[distances[i, j]/cutoff], [distances[i, j]/cutoff]])
        
        if edge_index:
            return (np.array(edge_index, dtype=np.int64).T, 
                   np.array(edge_features, dtype=np.float32))
        else:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 1), dtype=np.float32)
    
    def create_size_histogram(self, processed_data):
        """Create size histogram for conditional generation"""
        print("Creating size histogram...")
        
        size_pairs = []
        for data in processed_data:
            if 'ligand' in data and 'pocket' in data:
                n_lig = data['ligand']['num_atoms']
                n_pocket = data['pocket']['num_residues'] 
                size_pairs.append((n_lig, n_pocket))
        
        if not size_pairs:
            return None
        
        max_lig = max(pair[0] for pair in size_pairs)
        max_pocket = max(pair[1] for pair in size_pairs)
        
        histogram = np.zeros((max_lig + 1, max_pocket + 1))
        
        for n_lig, n_pocket in size_pairs:
            histogram[n_lig, n_pocket] += 1
        
        # Normalize by pocket size
        for j in range(max_pocket + 1):
            total = histogram[:, j].sum()
            if total > 0:
                histogram[:, j] /= total
        
        return histogram.tolist()
    
    def save_dataset_info(self, all_processed_data):
        """Save dataset information"""
        size_histogram = self.create_size_histogram(all_processed_data)
        
        dataset_info = {
            'atom_nf': self.atom_nf,
            'residue_nf': self.residue_nf,
            'atom_types': self.atom_types,
            'residue_types': self.residue_types,
            'bond_types': self.bond_types,
            'size_histogram': size_histogram,
            'total_complexes': len(all_processed_data),
            'processing_stats': self.stats
        }
        
        info_file = self.output_dir / 'dataset_info.pkl'
        with open(info_file, 'wb') as f:
            pickle.dump(dataset_info, f)
        
        print(f"Dataset info saved to {info_file}")
        return dataset_info
    
    def print_processing_stats(self):
        """Print processing statistics"""
        total = self.stats['total_processed']
        if total > 0:
            print(f"\nProcessing Statistics:")
            print(f"  Total processed: {total}")
            print(f"  Successful: {self.stats['successful']} ({self.stats['successful']/total*100:.1f}%)")
            print(f"  Valence errors: {self.stats['valence_errors']} ({self.stats['valence_errors']/total*100:.1f}%)")
            print(f"  Sanitization errors: {self.stats['sanitization_errors']} ({self.stats['sanitization_errors']/total*100:.1f}%)")
            print(f"  Coordinate errors: {self.stats['coord_errors']} ({self.stats['coord_errors']/total*100:.1f}%)")
            print(f"  Size filtered: {self.stats['size_filtered']} ({self.stats['size_filtered']/total*100:.1f}%)")
    
    def process_dataset(self):
        """Process dataset - skip all invalid molecules"""
        print("Processing CrossDock dataset (skipping invalid molecules)...")
        
        splits = self.load_splits(force_new_split=self.force_new_split)
        if splits is None:
            print("Failed to load splits!")
            return
        
        all_processed_data = []
        
        for split_name in ['train', 'test']:
            if split_name not in splits:
                continue
                
            entries = splits[split_name]
            print(f"\nProcessing {split_name} ({len(entries)} entries)...")
            
            processed_data = []
            
            if self.max_samples:
                if split_name == 'train':
                    max_for_split = min(self.max_samples, len(entries))
                else:
                    max_for_split = min(self.max_samples // 4, len(entries))
                print(f"Limited to {max_for_split} samples")
            else:
                max_for_split = len(entries)
            
            # Reset stats for this split
            split_stats = self.stats.copy()
            
            for i, entry in enumerate(tqdm(entries, desc=f"Processing {split_name}")):
                if i >= max_for_split:
                    break
                    
                complex_data = self.process_complex(entry)
                if complex_data:
                    processed_data.append(complex_data)
                    all_processed_data.append(complex_data)
            
            # Calculate split stats
            split_processed = self.stats['total_processed'] - split_stats['total_processed']
            split_successful = self.stats['successful'] - split_stats['successful']
            
            print(f"Success: {split_successful}/{split_processed} ({split_successful/split_processed*100:.1f}% if split_processed > 0 else 0)")
            
            if processed_data:
                output_file = self.output_dir / f"{split_name}.pkl"
                with open(output_file, 'wb') as f:
                    pickle.dump(processed_data, f)
                print(f"Saved to {output_file}")
                
                self._print_feature_summary(processed_data, split_name)
        
        if all_processed_data:
            self.save_dataset_info(all_processed_data)
        
        self.print_processing_stats()
        print("\nPreprocessing completed! Only valid molecules included.")
    
    def _print_feature_summary(self, processed_data, split_name):
        """Print feature summary"""
        print(f"\n{split_name.upper()} Summary:")
        
        ligand_atoms = [d['ligand']['num_atoms'] for d in processed_data if 'ligand' in d]
        if ligand_atoms:
            print(f"   Ligands: {len(ligand_atoms)} molecules")
            print(f"   Atoms: {np.mean(ligand_atoms):.1f} ± {np.std(ligand_atoms):.1f}")
            print(f"   Range: {np.min(ligand_atoms)}-{np.max(ligand_atoms)}")
        
        pocket_residues = [d['pocket']['num_residues'] for d in processed_data if 'pocket' in d]
        if pocket_residues:
            print(f"   Pockets: {len(pocket_residues)} pockets") 
            print(f"   Residues: {np.mean(pocket_residues):.1f} ± {np.std(pocket_residues):.1f}")
            print(f"   Range: {np.min(pocket_residues)}-{np.max(pocket_residues)}")

def main():
    parser = argparse.ArgumentParser(description='Clean CrossDock preprocessing - skip invalid molecules')
    parser.add_argument('--data_dir', type=str, default='data/crossdocked_pocket10')
    parser.add_argument('--output_dir', type=str, default='data/processed') 
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--force_new_split', action='store_true')
    
    args = parser.parse_args()
    
    if not Path(args.data_dir).exists():
        print(f"Data directory not found: {args.data_dir}")
        return
    
    print(f"Starting preprocessing (skip invalid molecules)...")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    
    preprocessor = GenerationFocusedPreprocessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        force_new_split=args.force_new_split
    )
    
    try:
        preprocessor.process_dataset()
        print("All done!")
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()