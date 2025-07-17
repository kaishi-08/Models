# debug_preprocessing_detailed.py - Debug chi tiết preprocessing
import sys
import os
import pickle
import traceback
from pathlib import Path

# Add src to path
sys.path.append('src')
sys.path.append('scripts')

def debug_splits():
    """Debug split data structure"""
    print("🔍 Debugging splits...")
    
    split_file = Path("data/split_by_name.pt")
    if split_file.exists():
        import torch
        splits = torch.load(split_file)
        
        print(f"✅ Splits loaded")
        print(f"   Keys: {list(splits.keys())}")
        
        for split_name, entries in splits.items():
            print(f"\n📊 {split_name} split:")
            print(f"   Total entries: {len(entries)}")
            print(f"   Entry type: {type(entries[0]) if entries else 'N/A'}")
            
            # Show first few entries
            for i, entry in enumerate(entries[:3]):
                print(f"   Entry {i}: {entry}")
                print(f"   Entry type: {type(entry)}")
                print(f"   Entry length: {len(entry) if hasattr(entry, '__len__') else 'N/A'}")
    else:
        print("❌ No split file found")

def debug_preprocessing_step_by_step():
    """Debug preprocessing từng bước"""
    print("\n🔍 Debugging preprocessing step by step...")
    
    try:
        from scripts.preprocess_crossdock_data import CrossDockPreprocessor
        
        # Tạo preprocessor
        preprocessor = CrossDockPreprocessor(
            data_dir="data/crossdocked_pocket10",
            output_dir="data/debug_test",
            max_samples=5
        )
        
        print("✅ Preprocessor created")
        
        # Load splits
        splits = preprocessor.load_splits()
        if not splits:
            print("❌ No splits loaded")
            return
        
        print(f"✅ Splits loaded: {list(splits.keys())}")
        
        # Test với train split
        train_entries = splits['train']
        print(f"\n🧪 Testing train entries (first 5):")
        
        for i, entry in enumerate(train_entries[:5]):
            print(f"\n--- Entry {i+1} ---")
            print(f"Entry: {entry}")
            print(f"Type: {type(entry)}")
            print(f"Length: {len(entry) if hasattr(entry, '__len__') else 'N/A'}")
            
            # Thử process entry này
            try:
                result = preprocessor.process_complex(entry)
                if result:
                    print(f"✅ SUCCESS: Entry processed")
                    print(f"   Keys: {list(result.keys())}")
                    if 'ligand' in result:
                        ligand = result['ligand']
                        print(f"   Ligand atoms: {len(ligand.get('atom_features', []))}")
                        print(f"   Ligand SMILES: {ligand.get('smiles', 'N/A')}")
                    if 'pocket' in result:
                        pocket = result['pocket']
                        print(f"   Pocket atoms: {len(pocket.get('atom_features', []))}")
                else:
                    print(f"❌ FAILED: process_complex returned None")
                    
                    # Debug tại sao fail
                    debug_single_entry_failure(preprocessor, entry)
                    
            except Exception as e:
                print(f"❌ EXCEPTION: {e}")
                traceback.print_exc()
                
                # Debug tại sao exception
                debug_single_entry_failure(preprocessor, entry)
    
    except Exception as e:
        print(f"❌ Error creating preprocessor: {e}")
        traceback.print_exc()

def debug_single_entry_failure(preprocessor, entry):
    """Debug tại sao một entry fail"""
    print(f"🔍 Debugging why entry failed...")
    
    if len(entry) < 4:
        print(f"   ❌ Entry too short: {len(entry)} < 4")
        return
    
    pocket_file, ligand_file, receptor_file, score = entry[:4]
    data_dir = Path("data/crossdocked_pocket10")
    
    # Check ligand file
    ligand_path = data_dir / ligand_file
    print(f"   Ligand file: {ligand_file}")
    print(f"   Full path: {ligand_path}")
    print(f"   Exists: {ligand_path.exists()}")
    
    if not ligand_path.exists():
        parent_dir = ligand_path.parent
        print(f"   Parent dir: {parent_dir}")
        print(f"   Parent exists: {parent_dir.exists()}")
        
        if parent_dir.exists():
            files = list(parent_dir.glob("*"))
            print(f"   Files in parent dir: {[f.name for f in files[:10]]}")
            
            sdf_files = list(parent_dir.glob("*.sdf"))
            print(f"   SDF files: {[f.name for f in sdf_files]}")
    else:
        # Try to load ligand
        try:
            from rdkit import Chem
            supplier = Chem.SDMolSupplier(str(ligand_path))
            mol = None
            for m in supplier:
                if m is not None:
                    mol = m
                    break
            
            if mol:
                print(f"   ✅ Ligand loaded: {mol.GetNumAtoms()} atoms")
            else:
                print(f"   ❌ Ligand could not be loaded from SDF")
        except Exception as e:
            print(f"   ❌ Ligand loading error: {e}")
    
    # Check pocket file
    pocket_path = data_dir / pocket_file
    print(f"   Pocket file: {pocket_file}")
    print(f"   Full path: {pocket_path}")
    print(f"   Exists: {pocket_path.exists()}")
    
    if not pocket_path.exists():
        parent_dir = pocket_path.parent
        if parent_dir.exists():
            pdb_files = list(parent_dir.glob("*.pdb"))
            print(f"   PDB files: {[f.name for f in pdb_files]}")
    else:
        # Try to load pocket
        try:
            from Bio.PDB import PDBParser
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('pocket', pocket_path)
            atom_count = sum(1 for model in structure for chain in model 
                           for residue in chain for atom in residue)
            print(f"   ✅ Pocket loaded: {atom_count} atoms")
        except Exception as e:
            print(f"   ❌ Pocket loading error: {e}")

def debug_file_structure():
    """Debug cấu trúc files"""
    print("\n🔍 Debugging file structure...")
    
    data_dir = Path("data/crossdocked_pocket10")
    
    # List first few subdirectories
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    print(f"📁 Found {len(subdirs)} subdirectories")
    
    for i, subdir in enumerate(subdirs[:3]):
        print(f"\n📁 Directory {i+1}: {subdir.name}")
        
        # List files in this directory
        files = list(subdir.glob("*"))
        print(f"   Files: {len(files)}")
        
        sdf_files = list(subdir.glob("*.sdf"))
        pdb_files = list(subdir.glob("*.pdb"))
        
        print(f"   SDF files: {len(sdf_files)}")
        print(f"   PDB files: {len(pdb_files)}")
        
        # Show some file names
        if sdf_files:
            print(f"   Sample SDF: {sdf_files[0].name}")
        if pdb_files:
            print(f"   Sample PDB: {pdb_files[0].name}")

def main():
    print("🔬 Detailed Preprocessing Debug")
    print("=" * 50)
    
    # Step 1: Debug splits
    debug_splits()
    
    # Step 2: Debug file structure
    debug_file_structure()
    
    # Step 3: Debug preprocessing step by step
    debug_preprocessing_step_by_step()

if __name__ == "__main__":
    main()