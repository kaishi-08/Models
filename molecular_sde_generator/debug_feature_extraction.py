#!/usr/bin/env python3
# debug_feature_extraction.py - Debug exactly where feature extraction fails

import pickle
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
import numpy as np
import traceback
import sys

# Add src to path for imports
sys.path.append('src')

def debug_original_preprocessing():
    """Debug the exact same logic as the original preprocessing"""
    print("🔍 Testing original preprocessing logic...")
    
    # Import the original class
    try:
        import sys
        sys.path.append('scripts')
        from scripts.preprocess_crossdock_data import CrossDockPreprocessor
        
        # Create preprocessor
        preprocessor = CrossDockPreprocessor(
            data_dir="data/crossdocked_pocket10",
            output_dir="data/processed_debug",
            max_samples=5  # Test with just 5 samples
        )
        
        # Load index
        index_file = Path("data/crossdocked_pocket10/index.pkl")
        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)
        
        print(f"✅ Original preprocessor created")
        
        # Test first 5 entries with detailed debugging
        for i, entry in enumerate(index_data[:5]):
            print(f"\n📋 Testing entry {i+1} with original logic:")
            print(f"   Entry: {entry}")
            
            try:
                result = preprocessor.process_complex(entry)
                if result:
                    print(f"   ✅ SUCCESS: Processed successfully")
                    print(f"      Ligand atoms: {len(result['ligand']['atom_features'])}")
                    if 'pocket' in result:
                        print(f"      Pocket atoms: {len(result['pocket']['atom_features'])}")
                    else:
                        print(f"      No pocket data")
                else:
                    print(f"   ❌ FAILED: process_complex returned None")
                    
            except Exception as e:
                print(f"   ❌ EXCEPTION: {e}")
                traceback.print_exc()
                
    except Exception as e:
        print(f"❌ Error importing original preprocessor: {e}")
        traceback.print_exc()

def test_mol_to_features():
    """Test mol_to_features function specifically"""
    print("\n🧪 Testing mol_to_features function...")
    
    # Load a test molecule
    index_file = Path("data/crossdocked_pocket10/index.pkl")
    with open(index_file, 'rb') as f:
        index_data = pickle.load(f)
    
    entry = index_data[0]
    ligand_file = entry[1]
    data_dir = Path("data/crossdocked_pocket10")
    ligand_path = data_dir / ligand_file
    
    # Load molecule
    supplier = Chem.SDMolSupplier(str(ligand_path))
    mol = None
    for m in supplier:
        if m is not None:
            mol = m
            break
    
    if mol is None:
        print("❌ Could not load test molecule")
        return
    
    print(f"✅ Test molecule loaded: {mol.GetNumAtoms()} atoms")
    
    # Test the exact feature extraction logic
    atom_types = {
        'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5, 
        'Cl': 6, 'Br': 7, 'I': 8, 'H': 9, 'UNK': 10
    }
    
    bond_types = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1, 
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3
    }
    
    try:
        # Step 1: Check conformer
        print("   Step 1: Checking conformer...")
        if mol.GetNumConformers() == 0:
            print("   Generating conformer...")
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        
        conf = mol.GetConformer()
        print(f"   ✅ Conformer ready")
        
        # Step 2: Extract atom features
        print("   Step 2: Extracting atom features...")
        atom_features = []
        positions = []
        
        for atom in mol.GetAtoms():
            # Atom type
            atom_type = atom_types.get(atom.GetSymbol(), atom_types['UNK'])
            
            # Additional features
            features = [
                atom_type,
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetTotalNumHs(),
                int(atom.IsInRing()),
                atom.GetMass() / 100.0  # Normalize
            ]
            atom_features.append(features)
            
            # 3D position
            pos = conf.GetAtomPosition(atom.GetIdx())
            positions.append([pos.x, pos.y, pos.z])
        
        print(f"   ✅ Atom features extracted: {len(atom_features)} atoms")
        
        # Step 3: Extract bond features
        print("   Step 3: Extracting bond features...")
        edge_index = []
        edge_features = []
        
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = bond_types.get(bond.GetBondType(), 0)
            
            # Add both directions
            edge_index.extend([[start, end], [end, start]])
            edge_features.extend([[bond_type], [bond_type]])
        
        print(f"   ✅ Bond features extracted: {len(edge_index)} edges")
        
        # Step 4: Create arrays
        print("   Step 4: Creating numpy arrays...")
        
        atom_features_array = np.array(atom_features, dtype=np.float32)
        positions_array = np.array(positions, dtype=np.float32)
        edge_index_array = np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64)
        edge_features_array = np.array(edge_features, dtype=np.float32) if edge_features else np.zeros((0, 1), dtype=np.float32)
        
        print(f"   ✅ Arrays created:")
        print(f"      Atom features: {atom_features_array.shape}")
        print(f"      Positions: {positions_array.shape}")
        print(f"      Edge index: {edge_index_array.shape}")
        print(f"      Edge features: {edge_features_array.shape}")
        
        # Step 5: SMILES
        print("   Step 5: Creating SMILES...")
        smiles = Chem.MolToSmiles(mol)
        print(f"   ✅ SMILES: {smiles}")
        
        # Step 6: Final data structure
        ligand_data = {
            'atom_features': atom_features_array,
            'positions': positions_array,
            'edge_index': edge_index_array,
            'edge_features': edge_features_array,
            'smiles': smiles
        }
        
        print(f"   ✅ mol_to_features would succeed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error in mol_to_features: {e}")
        traceback.print_exc()
        return False

def test_pdb_to_features():
    """Test pdb_to_features function specifically"""
    print("\n🧪 Testing pdb_to_features function...")
    
    # Load a test pocket
    index_file = Path("data/crossdocked_pocket10/index.pkl")
    with open(index_file, 'rb') as f:
        index_data = pickle.load(f)
    
    entry = index_data[0]
    pocket_file = entry[0]
    data_dir = Path("data/crossdocked_pocket10")
    pocket_path = data_dir / pocket_file
    
    print(f"   Testing pocket: {pocket_path}")
    
    atom_types = {
        'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5, 
        'Cl': 6, 'Br': 7, 'I': 8, 'H': 9, 'UNK': 10
    }
    
    amino_acids = {
        'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
        'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
        'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
        'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
        'UNK': 20
    }
    
    try:
        # Step 1: Load PDB
        print("   Step 1: Loading PDB...")
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('pocket', pocket_path)
        print(f"   ✅ PDB loaded")
        
        # Step 2: Extract features
        print("   Step 2: Extracting atom features...")
        atom_features = []
        positions = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    res_name = residue.get_resname()
                    res_id = residue.get_id()[1]
                    
                    for atom in residue:
                        # Basic atom features
                        element = atom.element.strip() if atom.element else 'C'
                        atom_type = atom_types.get(element, atom_types['UNK'])
                        res_type = amino_acids.get(res_name, amino_acids['UNK'])
                        
                        features = [
                            atom_type,
                            res_type, 
                            res_id % 1000,  # Normalize residue ID
                            int(atom.name.startswith('C')),  # Is carbon backbone
                            int(atom.name.startswith('N')),  # Is nitrogen backbone
                            int(atom.name.startswith('O')),  # Is oxygen backbone
                            0.0,  # Placeholder for surface accessibility
                            0.0   # Placeholder for charge
                        ]
                        
                        atom_features.append(features)
                        
                        # 3D coordinates
                        coord = atom.coord
                        positions.append([coord[0], coord[1], coord[2]])
        
        print(f"   ✅ Features extracted: {len(atom_features)} atoms")
        
        # Step 3: Connectivity
        print("   Step 3: Computing connectivity...")
        positions_array = np.array(positions)
        
        # Limit connectivity computation to avoid memory issues
        max_atoms = min(len(positions), 1000)  # Limit to 1000 atoms
        distances = np.linalg.norm(positions_array[:max_atoms, None] - positions_array[None, :max_atoms], axis=2)
        
        edge_index = []
        edge_features = []
        
        # Connect atoms within 5Å
        for i in range(max_atoms):
            for j in range(i + 1, max_atoms):
                if distances[i, j] < 5.0:
                    edge_index.extend([[i, j], [j, i]])
                    edge_features.extend([[distances[i, j]], [distances[i, j]]])
        
        print(f"   ✅ Connectivity computed: {len(edge_index)} edges")
        
        # Step 4: Create arrays
        print("   Step 4: Creating arrays...")
        
        atom_features_array = np.array(atom_features, dtype=np.float32)
        positions_array = np.array(positions, dtype=np.float32)
        edge_index_array = np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64)
        edge_features_array = np.array(edge_features, dtype=np.float32) if edge_features else np.zeros((0, 1), dtype=np.float32)
        
        print(f"   ✅ Arrays created:")
        print(f"      Atom features: {atom_features_array.shape}")
        print(f"      Positions: {positions_array.shape}")
        print(f"      Edge index: {edge_index_array.shape}")
        print(f"      Edge features: {edge_features_array.shape}")
        
        print(f"   ✅ pdb_to_features would succeed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error in pdb_to_features: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Debugging feature extraction specifically...")
    
    # Test individual functions
    mol_success = test_mol_to_features()
    pdb_success = test_pdb_to_features()
    
    if mol_success and pdb_success:
        print("\n✅ Individual functions work - testing original preprocessing...")
        debug_original_preprocessing()
    else:
        print("\n❌ Individual functions failed - need to fix these first")