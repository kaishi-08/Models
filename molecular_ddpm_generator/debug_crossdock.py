# debug_crossdock.py - Standalone debug script for CrossDock data
import os
import pickle
import sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings('ignore')

def debug_crossdock_data(data_dir="data/crossdocked_pocket10", max_entries=5):
    """Debug CrossDock data structure and file accessibility"""
    
    print("🐛 ===== CROSSDOCK DEBUG REPORT =====")
    print(f"🐛 Data directory: {data_dir}")
    
    data_path = Path(data_dir)
    
    # 1. Check data directory
    print(f"\n🔍 1. DATA DIRECTORY CHECK")
    print(f"   Directory exists: {data_path.exists()}")
    if data_path.exists():
        print(f"   Directory contents: {list(data_path.iterdir())[:10]}")
    else:
        print("   ❌ Data directory not found!")
        return
    
    # 2. Check index.pkl
    print(f"\n🔍 2. INDEX FILE CHECK")
    index_file = data_path / "index.pkl"
    print(f"   index.pkl exists: {index_file.exists()}")
    
    if not index_file.exists():
        print("   ❌ index.pkl not found!")
        return
    
    try:
        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)
        
        print(f"   ✅ index.pkl loaded successfully")
        print(f"   Total entries: {len(index_data)}")
        print(f"   First entry: {index_data[0]}")
        print(f"   Entry type: {type(index_data[0])}")
        print(f"   Entry length: {len(index_data[0])}")
        
        if len(index_data[0]) >= 2:
            print(f"   Pocket file (0): {index_data[0][0]}")
            print(f"   Ligand file (1): {index_data[0][1]}")
            if len(index_data[0]) >= 3:
                print(f"   Receptor file (2): {index_data[0][2]}")
            if len(index_data[0]) >= 4:
                print(f"   Score (3): {index_data[0][3]}")
    
    except Exception as e:
        print(f"   ❌ Error loading index.pkl: {e}")
        return
    
    # 3. Test file accessibility
    print(f"\n🔍 3. FILE ACCESSIBILITY CHECK")
    
    for i in range(min(max_entries, len(index_data))):
        entry = index_data[i]
        print(f"\n   --- Entry {i+1} ---")
        print(f"   Raw entry: {entry}")
        
        if len(entry) >= 2:
            pocket_file = entry[0]
            ligand_file = entry[1]
            
            # Check ligand file
            ligand_path = data_path / ligand_file
            print(f"   Ligand path: {ligand_path}")
            print(f"   Ligand exists: {ligand_path.exists()}")
            
            if not ligand_path.exists():
                # Try parent directory
                parent = ligand_path.parent
                print(f"   Parent dir: {parent}")
                print(f"   Parent exists: {parent.exists()}")
                if parent.exists():
                    sdf_files = list(parent.glob("*.sdf"))
                    print(f"   SDF files in parent: {sdf_files}")
            
            # Check pocket file
            pocket_path = data_path / pocket_file
            print(f"   Pocket path: {pocket_path}")
            print(f"   Pocket exists: {pocket_path.exists()}")
            
            if not pocket_path.exists():
                parent = pocket_path.parent
                print(f"   Parent dir: {parent}")
                print(f"   Parent exists: {parent.exists()}")
                if parent.exists():
                    pdb_files = list(parent.glob("*pocket*.pdb"))
                    print(f"   Pocket PDB files: {pdb_files}")
    
    # 4. Test molecule reading
    print(f"\n🔍 4. MOLECULE READING TEST")
    
    for i in range(min(3, len(index_data))):
        entry = index_data[i]
        if len(entry) >= 2:
            ligand_file = entry[1]
            ligand_path = data_path / ligand_file
            
            print(f"\n   --- Testing molecule {i+1} ---")
            print(f"   File: {ligand_file}")
            
            # Find actual file
            actual_file = None
            if ligand_path.exists():
                actual_file = ligand_path
            else:
                parent = ligand_path.parent
                if parent.exists():
                    sdf_files = list(parent.glob("*.sdf"))
                    if sdf_files:
                        actual_file = sdf_files[0]
            
            if actual_file:
                print(f"   Using file: {actual_file}")
                print(f"   File size: {actual_file.stat().st_size} bytes")
                
                # Test RDKit reading
                try:
                    if actual_file.suffix.lower() == '.sdf':
                        supplier = Chem.SDMolSupplier(str(actual_file), sanitize=False)
                        mol = None
                        for m in supplier:
                            if m is not None:
                                mol = m
                                break
                        
                        if mol:
                            print(f"   ✅ RDKit read successful")
                            print(f"   Atoms: {mol.GetNumAtoms()}")
                            print(f"   Bonds: {mol.GetNumBonds()}")
                            print(f"   SMILES: {Chem.MolToSmiles(mol)}")
                        else:
                            print(f"   ❌ RDKit could not read molecule")
                    
                except Exception as e:
                    print(f"   ❌ RDKit error: {e}")
            else:
                print(f"   ❌ No actual file found")
    
    # 5. Test pocket reading
    print(f"\n🔍 5. POCKET READING TEST")
    
    for i in range(min(2, len(index_data))):
        entry = index_data[i]
        if len(entry) >= 1:
            pocket_file = entry[0]
            pocket_path = data_path / pocket_file
            
            print(f"\n   --- Testing pocket {i+1} ---")
            print(f"   File: {pocket_file}")
            
            # Find actual file
            actual_file = None
            if pocket_path.exists():
                actual_file = pocket_path
            else:
                parent = pocket_path.parent
                if parent.exists():
                    pdb_files = list(parent.glob("*pocket*.pdb"))
                    if pdb_files:
                        actual_file = pdb_files[0]
            
            if actual_file:
                print(f"   Using file: {actual_file}")
                print(f"   File size: {actual_file.stat().st_size} bytes")
                
                # Test BioPython reading
                try:
                    parser = PDBParser(QUIET=True)
                    structure = parser.get_structure('pocket', actual_file)
                    
                    residue_count = 0
                    ca_count = 0
                    
                    for model in structure:
                        for chain in model:
                            for residue in chain:
                                if residue.id[0] == ' ':  # Standard residue
                                    residue_count += 1
                                    if 'CA' in residue:
                                        ca_count += 1
                    
                    print(f"   ✅ BioPython read successful")
                    print(f"   Residues: {residue_count}")
                    print(f"   CA atoms: {ca_count}")
                    
                except Exception as e:
                    print(f"   ❌ BioPython error: {e}")
            else:
                print(f"   ❌ No actual pocket file found")
    
    # 6. Directory structure analysis
    print(f"\n🔍 6. DIRECTORY STRUCTURE ANALYSIS")
    
    try:
        def analyze_directory(path, max_depth=3, current_depth=0):
            if current_depth > max_depth:
                return
            
            indent = "   " * current_depth
            if path.is_dir():
                print(f"{indent}{path.name}/")
                try:
                    children = list(path.iterdir())[:5]  # Limit to first 5
                    for child in children:
                        analyze_directory(child, max_depth, current_depth + 1)
                    if len(list(path.iterdir())) > 5:
                        print(f"{indent}   ... ({len(list(path.iterdir()))} total)")
                except PermissionError:
                    print(f"{indent}   [Permission Denied]")
            else:
                size = path.stat().st_size if path.exists() else 0
                print(f"{indent}{path.name} ({size} bytes)")
        
        print("   Directory structure (first few levels):")
        analyze_directory(data_path, max_depth=2)
        
    except Exception as e:
        print(f"   Error analyzing directory: {e}")
    
    print(f"\n🐛 ===== DEBUG REPORT COMPLETE =====")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug CrossDock data structure')
    parser.add_argument('--data_dir', type=str, default='data/crossdocked_pocket10',
                       help='Path to CrossDock data directory')
    parser.add_argument('--max_entries', type=int, default=5,
                       help='Maximum entries to test')
    
    args = parser.parse_args()
    
    debug_crossdock_data(args.data_dir, args.max_entries)

if __name__ == "__main__":
    main()