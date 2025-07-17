#!/usr/bin/env python3
# debug_data_structure.py - Debug CrossDock data structure

import pickle
import os
from pathlib import Path

def debug_data_structure():
    print("🔍 Debugging CrossDock data structure...")
    
    # Check main directories
    data_dir = Path("data/crossdocked_pocket10")
    print(f"📁 Data directory: {data_dir}")
    print(f"   Exists: {data_dir.exists()}")
    
    if not data_dir.exists():
        print("❌ Data directory not found!")
        return
    
    # Check index.pkl
    index_file = data_dir / "index.pkl"
    print(f"📄 Index file: {index_file}")
    print(f"   Exists: {index_file.exists()}")
    
    if not index_file.exists():
        print("❌ Index file not found!")
        return
    
    # Load and examine index
    try:
        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)
        
        print(f"✅ Index loaded successfully")
        print(f"   Total entries: {len(index_data)}")
        print(f"   Data type: {type(index_data)}")
        
        # Check first few entries
        print("\n📋 First 5 entries:")
        for i, entry in enumerate(index_data[:5]):
            print(f"   {i+1}. {entry}")
            print(f"      Type: {type(entry)}")
            print(f"      Length: {len(entry) if hasattr(entry, '__len__') else 'N/A'}")
        
        # Check if files exist for first entry
        if len(index_data) > 0:
            print(f"\n🔍 Checking files for first entry...")
            entry = index_data[0]
            
            if len(entry) >= 4:
                pocket_file, ligand_file, receptor_file, score = entry[:4]
                
                print(f"   Pocket file: {pocket_file}")
                pocket_path = data_dir / pocket_file
                print(f"   Full path: {pocket_path}")
                print(f"   Exists: {pocket_path.exists()}")
                
                print(f"   Ligand file: {ligand_file}")
                ligand_path = data_dir / ligand_file
                print(f"   Full path: {ligand_path}")
                print(f"   Exists: {ligand_path.exists()}")
                
                print(f"   Score: {score}")
                
                # Check parent directories
                pocket_parent = pocket_path.parent
                ligand_parent = ligand_path.parent
                
                print(f"\n📁 Directory structure:")
                print(f"   Pocket parent: {pocket_parent}")
                print(f"   Exists: {pocket_parent.exists()}")
                if pocket_parent.exists():
                    files = list(pocket_parent.glob("*"))[:5]
                    print(f"   Files in pocket dir: {[f.name for f in files]}")
                
                print(f"   Ligand parent: {ligand_parent}")
                print(f"   Exists: {ligand_parent.exists()}")
                if ligand_parent.exists():
                    files = list(ligand_parent.glob("*"))[:5]
                    print(f"   Files in ligand dir: {[f.name for f in files]}")
            
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        import traceback
        traceback.print_exc()
    
    # Check overall directory structure
    print(f"\n📁 Overall directory structure:")
    try:
        subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
        files = [f for f in data_dir.iterdir() if f.is_file()]
        
        print(f"   Subdirectories ({len(subdirs)}):")
        for d in subdirs[:10]:  # Show first 10
            print(f"      {d.name}")
        
        print(f"   Files ({len(files)}):")
        for f in files[:10]:  # Show first 10
            print(f"      {f.name}")
            
    except Exception as e:
        print(f"❌ Error listing directory: {e}")

if __name__ == "__main__":
    debug_data_structure()