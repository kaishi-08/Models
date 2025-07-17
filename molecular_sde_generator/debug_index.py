#!/usr/bin/env python3
# debug_index.py - Debug script để kiểm tra index.pkl

import pickle
import os
from pathlib import Path

def debug_index():
    print("🔍 Debugging CrossDock index.pkl...")
    
    # Kiểm tra file index.pkl
    index_file = Path("data/crossdocked_pocket10/index.pkl")
    
    if not index_file.exists():
        print(f"❌ Index file not found: {index_file}")
        return
    
    print(f"✅ Index file found: {index_file}")
    
    # Load index
    try:
        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)
        
        print(f"✅ Index loaded successfully")
        print(f"   Type: {type(index_data)}")
        print(f"   Length: {len(index_data)}")
        
        # Kiểm tra format của entries
        print("\n🔍 Analyzing index entries...")
        
        # Lấy 5 entries đầu tiên
        sample_entries = index_data[:5] if isinstance(index_data, list) else list(index_data.items())[:5]
        
        for i, entry in enumerate(sample_entries):
            print(f"\n--- Entry {i+1} ---")
            print(f"Type: {type(entry)}")
            print(f"Content: {entry}")
            
            if isinstance(entry, (list, tuple)) and len(entry) >= 4:
                pocket_file, ligand_file, receptor_file, score = entry[:4]
                print(f"   Pocket file: {pocket_file}")
                print(f"   Ligand file: {ligand_file}")
                print(f"   Receptor file: {receptor_file}")
                print(f"   Score: {score}")
                
                # Kiểm tra file có tồn tại không
                base_dir = Path("data/crossdocked_pocket10")
                pocket_path = base_dir / pocket_file
                ligand_path = base_dir / ligand_file
                
                print(f"   Pocket exists: {pocket_path.exists()}")
                print(f"   Ligand exists: {ligand_path.exists()}")
                
                if not pocket_path.exists():
                    print(f"   ❌ Pocket file not found: {pocket_path}")
                    # Thử tìm file tương tự
                    parent_dir = pocket_path.parent
                    if parent_dir.exists():
                        print(f"   Parent directory exists: {parent_dir}")
                        files = list(parent_dir.glob("*.pdb"))
                        print(f"   PDB files in directory: {len(files)}")
                        if files:
                            print(f"   Sample PDB files: {[f.name for f in files[:3]]}")
                    else:
                        print(f"   ❌ Parent directory not found: {parent_dir}")
                
                if not ligand_path.exists():
                    print(f"   ❌ Ligand file not found: {ligand_path}")
                    # Thử tìm file tương tự
                    parent_dir = ligand_path.parent
                    if parent_dir.exists():
                        print(f"   Parent directory exists: {parent_dir}")
                        files = list(parent_dir.glob("*.sdf"))
                        print(f"   SDF files in directory: {len(files)}")
                        if files:
                            print(f"   Sample SDF files: {[f.name for f in files[:3]]}")
                    else:
                        print(f"   ❌ Parent directory not found: {parent_dir}")
            
            if i >= 2:  # Chỉ kiểm tra 3 entries đầu tiên
                break
                
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        import traceback
        traceback.print_exc()

def check_directory_structure():
    print("\n🔍 Checking directory structure...")
    
    base_dir = Path("data/crossdocked_pocket10")
    
    if not base_dir.exists():
        print(f"❌ Base directory not found: {base_dir}")
        return
    
    print(f"✅ Base directory exists: {base_dir}")
    
    # Liệt kê các subdirectories
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    files = [f for f in base_dir.iterdir() if f.is_file()]
    
    print(f"   Subdirectories: {len(subdirs)}")
    print(f"   Files: {len(files)}")
    
    # Show some subdirectories
    print("\n📁 Sample subdirectories:")
    for i, subdir in enumerate(subdirs[:10]):
        print(f"   {i+1}. {subdir.name}")
        
        # Check files in subdirectory
        subdir_files = list(subdir.glob("*"))
        print(f"      Files: {len(subdir_files)}")
        
        # Show file types
        extensions = {}
        for f in subdir_files:
            ext = f.suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1
        print(f"      Extensions: {extensions}")
        
        if i >= 4:  # Chỉ show 5 subdirectories
            break

if __name__ == "__main__":
    debug_index()
    check_directory_structure()