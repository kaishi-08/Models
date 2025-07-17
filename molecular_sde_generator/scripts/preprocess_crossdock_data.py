# scripts/preprocess_crossdock_data.py - Preprocessing CrossDock data

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.append('src')

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from Bio.PDB import PDBParser
    from Bio.PDB.DSSP import DSSP
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install rdkit-pypi biopython")
    sys.exit(1)

from utils.molecular_utils import MolecularFeaturizer

class CrossDockPreprocessor:
    """Preprocessor for CrossDock2020 dataset"""
    
    def __init__(self, raw_data_path: str = "data/raw/crossdock2020",
                 processed_data_path: str = "data/processed",
                 config: Dict = None):
        
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.config = config or self._get_default_config()
        
        # Create output directory
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        self.featurizer = MolecularFeaturizer()
        
    def _get_default_config(self) -> Dict:
        """Default preprocessing configuration"""
        return {
            'max_atoms': 50,
            'min_atoms': 5,
            'max_pocket_atoms': 500,
            'pocket_radius': 10.0,
            'pocket_cutoff': 5.0,
            'remove_hydrogens': False,
            'add_hydrogens': False,
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        }
    
    def load_index(self) -> Dict:
        """Load CrossDock index file"""
        index_file = self.raw_data_path / "index.pkl"
        
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)
        
        print(f"✅ Loaded index with {len(index_data)} entries")
        return index_data
    
    def process_ligand(self, ligand_file: str) -> Optional[Dict]:
        """Process ligand SDF file"""
        try:
            # Read SDF file
            supplier = Chem.SDMolSupplier(ligand_file)
            mol = next(supplier)
            
            if mol is None:
                return None
            
            # Remove/add hydrogens based on config
            if self.config['remove_hydrogens']:
                mol = Chem.RemoveHs(mol)
            elif self.config['add_hydrogens']:
                mol = Chem.AddHs(mol)
            
            # Check atom count constraints
            num_atoms = mol.GetNumAtoms()
            if num_atoms < self.config['min_atoms'] or num_atoms > self.config['max_atoms']:
                return None
            
            # Convert to graph representation
            graph_data = self.featurizer.mol_to_graph(mol, add_hydrogens=False)
            
            # Get SMILES
            smiles = Chem.MolToSmiles(mol)
            
            return {
                'atom_features': graph_data['atom_features'],
                'positions': graph_data['positions'],
                'edge_index': graph_data['edge_index'],
                'edge_features': graph_data['edge_features'],
                'smiles': smiles,
                'num_atoms': num_atoms
            }
            
        except Exception as e:
            print(f"❌ Error processing ligand {ligand_file}: {e}")
            return None
    
    def process_pocket(self, pocket_file: str, ligand_pos: np.ndarray) -> Optional[Dict]:
        """Process protein pocket PDB file"""
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('pocket', pocket_file)
            
            # Extract atoms within pocket radius
            pocket_atoms = []
            pocket_positions = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            atom_pos = atom.coord
                            
                            # Check if within pocket radius of any ligand atom
                            min_dist = np.min(np.linalg.norm(
                                atom_pos - ligand_pos, axis=1
                            ))
                            
                            if min_dist <= self.config['pocket_radius']:
                                # Atom features: [atomic_num, residue_type, is_backbone, ...]
                                residue_name = residue.resname
                                atom_name = atom.name
                                element = atom.element
                                
                                # Map residue to number
                                residue_mapping = {
                                    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
                                    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
                                    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
                                    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
                                }
                                
                                # Atomic number mapping
                                atomic_nums = {'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15, 'H': 1}
                                atomic_num = atomic_nums.get(element, 6)
                                
                                # Is backbone atom
                                is_backbone = 1 if atom_name in ['N', 'CA', 'C', 'O'] else 0
                                
                                residue_type = residue_mapping.get(residue_name, 20)
                                
                                # Feature vector
                                features = [
                                    atomic_num,           # 0: atomic number
                                    residue_type,         # 1: residue type
                                    is_backbone,          # 2: is backbone
                                    residue.id[1],        # 3: residue number
                                    ord(chain.id) - ord('A'), # 4: chain ID
                                    min_dist,             # 5: distance to ligand
                                    0,                    # 6: placeholder
                                    0                     # 7: placeholder
                                ]
                                
                                pocket_atoms.append(features)
                                pocket_positions.append(atom_pos)
            
            if len(pocket_atoms) == 0 or len(pocket_atoms) > self.config['max_pocket_atoms']:
                return None
            
            # Convert to numpy arrays
            pocket_atoms = np.array(pocket_atoms, dtype=np.float32)
            pocket_positions = np.array(pocket_positions, dtype=np.float32)
            
            # Create pocket graph (simple distance-based)
            edge_index, edge_features = self._create_pocket_graph(pocket_positions)
            
            return {
                'atom_features': pocket_atoms,
                'positions': pocket_positions,
                'edge_index': edge_index,
                'edge_features': edge_features,
                'num_atoms': len(pocket_atoms)
            }
            
        except Exception as e:
            print(f"❌ Error processing pocket {pocket_file}: {e}")
            return None
    
    def _create_pocket_graph(self, positions: np.ndarray) -> tuple:
        """Create graph connectivity for pocket"""
        from scipy.spatial.distance import cdist
        
        # Compute pairwise distances
        distances = cdist(positions, positions)
        
        # Create edges based on cutoff
        edge_index = []
        edge_features = []
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = distances[i, j]
                if dist < self.config['pocket_cutoff']:
                    edge_index.extend([[i, j], [j, i]])
                    edge_features.extend([dist, dist])
        
        edge_index = np.array(edge_index).T if edge_index else np.zeros((2, 0))
        edge_features = np.array(edge_features).reshape(-1, 1) if edge_features else np.zeros((0, 1))
        
        return edge_index.astype(np.int64), edge_features.astype(np.float32)
    
    def process_complex(self, complex_id: str, ligand_file: str, pocket_file: str) -> Optional[Dict]:
        """Process ligand-pocket complex"""
        
        # Process ligand
        ligand_data = self.process_ligand(ligand_file)
        if ligand_data is None:
            return None
        
        # Process pocket
        pocket_data = self.process_pocket(pocket_file, ligand_data['positions'])
        if pocket_data is None:
            return None
        
        return {
            'complex_id': complex_id,
            'ligand': ligand_data,
            'pocket': pocket_data
        }
    
    def process_all_complexes(self, index_data: Dict) -> List[Dict]:
        """Process all complexes in the dataset"""
        processed_complexes = []
        failed_count = 0
        
        print("🔄 Processing complexes...")
        
        for complex_id, complex_info in tqdm(index_data.items()):
            try:
                # Construct file paths (adjust based on your data structure)
                # This is a simplified example - adjust paths based on actual CrossDock structure
                ligand_file = self.raw_data_path / f"{complex_id}_ligand.sdf"
                pocket_file = self.raw_data_path / f"{complex_id}_pocket.pdb"
                
                # Check if files exist
                if not ligand_file.exists() or not pocket_file.exists():
                    # Try alternative naming conventions
                    ligand_file = self.raw_data_path / complex_id / "ligand.sdf"
                    pocket_file = self.raw_data_path / complex_id / "pocket.pdb"
                    
                    if not ligand_file.exists() or not pocket_file.exists():
                        failed_count += 1
                        continue
                
                # Process complex
                complex_data = self.process_complex(complex_id, str(ligand_file), str(pocket_file))
                
                if complex_data is not None:
                    processed_complexes.append(complex_data)
                else:
                    failed_count += 1
                    
            except Exception as e:
                print(f"❌ Error processing complex {complex_id}: {e}")
                failed_count += 1
                continue
        
        print(f"✅ Processed {len(processed_complexes)} complexes")
        print(f"❌ Failed: {failed_count} complexes")
        
        return processed_complexes
    
    def split_data(self, complexes: List[Dict]) -> Dict[str, List[Dict]]:
        """Split data into train/val/test sets"""
        np.random.seed(42)  # For reproducibility
        
        # Shuffle complexes
        indices = np.random.permutation(len(complexes))
        
        # Calculate split sizes
        train_size = int(len(complexes) * self.config['train_split'])
        val_size = int(len(complexes) * self.config['val_split'])
        
        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        splits = {
            'train': [complexes[i] for i in train_indices],
            'val': [complexes[i] for i in val_indices],
            'test': [complexes[i] for i in test_indices]
        }
        
        print(f"📊 Data splits:")
        for split_name, split_data in splits.items():
            print(f"   {split_name}: {len(split_data)} complexes")
        
        return splits
    
    def save_processed_data(self, splits: Dict[str, List[Dict]]):
        """Save processed data to pickle files"""
        print("💾 Saving processed data...")
        
        for split_name, split_data in splits.items():
            output_file = self.processed_data_path / f"{split_name}.pkl"
            
            with open(output_file, 'wb') as f:
                pickle.dump(split_data, f)
            
            print(f"✅ Saved {split_name}: {len(split_data)} complexes to {output_file}")
    
    def generate_statistics(self, splits: Dict[str, List[Dict]]):
        """Generate dataset statistics"""
        print("📈 Generating statistics...")
        
        all_complexes = []
        for split_data in splits.values():
            all_complexes.extend(split_data)
        
        # Molecular statistics
        atom_counts = []
        bond_counts = []
        pocket_atom_counts = []
        
        for complex_data in all_complexes:
            ligand = complex_data['ligand']
            pocket = complex_data['pocket']
            
            atom_counts.append(ligand['num_atoms'])
            bond_counts.append(ligand['edge_index'].shape[1] // 2)  # Undirected
            pocket_atom_counts.append(pocket['num_atoms'])
        
        stats = {
            'total_complexes': len(all_complexes),
            'atom_count_stats': {
                'mean': float(np.mean(atom_counts)),
                'std': float(np.std(atom_counts)),
                'min': int(np.min(atom_counts)),
                'max': int(np.max(atom_counts))
            },
            'bond_count_stats': {
                'mean': float(np.mean(bond_counts)),
                'std': float(np.std(bond_counts)),
                'min': int(np.min(bond_counts)),
                'max': int(np.max(bond_counts))
            },
            'pocket_atom_stats': {
                'mean': float(np.mean(pocket_atom_counts)),
                'std': float(np.std(pocket_atom_counts)),
                'min': int(np.min(pocket_atom_counts)),
                'max': int(np.max(pocket_atom_counts))
            },
            'splits': {name: len(data) for name, data in splits.items()}
        }
        
        # Save statistics
        with open(self.processed_data_path / "dataset_stats.json", 'w') as f:
            import json
            json.dump(stats, f, indent=2)
        
        print("✅ Statistics saved to dataset_stats.json")
        
        # Print summary
        print("\n📊 Dataset Summary:")
        print(f"   Total complexes: {stats['total_complexes']}")
        print(f"   Atoms per molecule: {stats['atom_count_stats']['mean']:.1f} ± {stats['atom_count_stats']['std']:.1f}")
        print(f"   Bonds per molecule: {stats['bond_count_stats']['mean']:.1f} ± {stats['bond_count_stats']['std']:.1f}")
        print(f"   Pocket atoms: {stats['pocket_atom_stats']['mean']:.1f} ± {stats['pocket_atom_stats']['std']:.1f}")
    
    def run_preprocessing(self):
        """Run complete preprocessing pipeline"""
        print("🔬 CrossDock Data Preprocessing")
        print("=" * 50)
        
        # Load index
        index_data = self.load_index()
        
        # Process complexes
        processed_complexes = self.process_all_complexes(index_data)
        
        if len(processed_complexes) == 0:
            print("❌ No complexes were processed successfully!")
            return False
        
        # Split data
        splits = self.split_data(processed_complexes)
        
        # Save processed data
        self.save_processed_data(splits)
        
        # Generate statistics
        self.generate_statistics(splits)
        
        print("\n🎉 Preprocessing completed successfully!")
        return True

def main():
    parser = argparse.ArgumentParser(description='Preprocess CrossDock2020 dataset')
    parser.add_argument('--raw_path', type=str, default='data/raw/crossdock2020',
                       help='Path to raw CrossDock data')
    parser.add_argument('--output_path', type=str, default='data/processed',
                       help='Output path for processed data')
    parser.add_argument('--config', type=str, help='Config file for preprocessing')
    parser.add_argument('--max_atoms', type=int, default=50,
                       help='Maximum atoms per molecule')
    parser.add_argument('--max_pocket_atoms', type=int, default=500,
                       help='Maximum pocket atoms')
    parser.add_argument('--pocket_radius', type=float, default=10.0,
                       help='Pocket radius around ligand')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
    else:
        config = {
            'max_atoms': args.max_atoms,
            'min_atoms': 5,
            'max_pocket_atoms': args.max_pocket_atoms,
            'pocket_radius': args.pocket_radius,
            'pocket_cutoff': 5.0,
            'remove_hydrogens': False,
            'add_hydrogens': False,
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        }
    
    # Create preprocessor and run
    preprocessor = CrossDockPreprocessor(
        raw_data_path=args.raw_path,
        processed_data_path=args.output_path,
        config=config
    )
    
    success = preprocessor.run_preprocessing()
    
    if success:
        print("\n✅ Ready for training!")
        print("Next steps:")
        print("   1. python scripts/check_data_flow.py  # Validate data")
        print("   2. python scripts/train_enhanced.py   # Start training")
    else:
        print("\n❌ Preprocessing failed!")

if __name__ == "__main__":
    main()