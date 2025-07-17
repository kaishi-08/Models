# check_data_structure.py - Fixed for crossdocked_pocket10
import os
import pickle
import torch
from pathlib import Path

def check_data_structure():
    print("🔍 Checking crossdocked_pocket10 data structure...")
    
    # Check main data directory
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ data/ directory not found!")
        return
    
    print(f"✅ Found data directory: {data_dir}")
    
    # Check crossdocked_pocket10 - FIXED PATH
    crossdock_dir = data_dir / "crossdocked_pocket10"
    if crossdock_dir.exists():
        print(f"✅ Found crossdocked_pocket10: {crossdock_dir}")
        
        # List subdirectories
        subdirs = [d for d in crossdock_dir.iterdir() if d.is_dir()]
        files = [f for f in crossdock_dir.iterdir() if f.is_file()]
        
        print(f"   Subdirectories: {len(subdirs)}")
        print(f"   Files: {len(files)}")
        
        # Show sample subdirectories
        for i, subdir in enumerate(subdirs[:5]):
            print(f"     {i+1}. {subdir.name}")
        if len(subdirs) > 5:
            print(f"     ... and {len(subdirs) - 5} more")
        
        # Show sample files
        for i, file in enumerate(files[:5]):
            print(f"     {i+1}. {file.name}")
        
        # Check index.pkl
        index_file = crossdock_dir / "index.pkl"
        if index_file.exists():
            print(f"✅ Found index.pkl")
            try:
                with open(index_file, 'rb') as f:
                    index_data = pickle.load(f)
                print(f"   Index entries: {len(index_data)}")
                print(f"   Index type: {type(index_data)}")
                
                # Show sample entries
                if isinstance(index_data, list) and len(index_data) > 0:
                    print("   Sample entries:")
                    for i, entry in enumerate(index_data[:3]):
                        print(f"      {i+1}. {entry}")
                elif isinstance(index_data, dict):
                    print("   Sample keys:", list(index_data.keys())[:3])
                    
            except Exception as e:
                print(f"❌ Error reading index.pkl: {e}")
        else:
            print("❌ index.pkl not found")
    else:
        print("❌ crossdocked_pocket10/ not found")
        return
    
    # Check split_by_name.pt
    split_file = data_dir / "split_by_name.pt"
    if split_file.exists():
        print(f"✅ Found split_by_name.pt")
        try:
            split_data = torch.load(split_file)
            print(f"   Split data type: {type(split_data)}")
            if isinstance(split_data, dict):
                print(f"   Split keys: {list(split_data.keys())}")
                for key, value in split_data.items():
                    if hasattr(value, '__len__'):
                        print(f"     {key}: {len(value)} entries")
        except Exception as e:
            print(f"❌ Error reading split_by_name.pt: {e}")
    else:
        print("❌ split_by_name.pt not found in data/")
    
    # Check processed data
    processed_dir = data_dir / "processed"
    if processed_dir.exists():
        print(f"✅ Found processed directory")
        processed_files = list(processed_dir.glob("*.pkl"))
        print(f"   Processed files: {len(processed_files)}")
        for file in processed_files:
            print(f"     - {file.name}")
            
            # Check file content
            try:
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                print(f"       {len(data)} samples")
                if data:
                    sample = data[0]
                    print(f"       Sample keys: {list(sample.keys())}")
            except Exception as e:
                print(f"       Error reading {file.name}: {e}")
    else:
        print("❌ processed/ directory not found - need to create")
    
    print("\n" + "="*50)

def check_sample_files():
    """Check if sample ligand and pocket files exist"""
    print("🔍 Checking sample files...")
    
    try:
        # Load index
        index_file = Path("data/crossdocked_pocket10/index.pkl")
        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)
        
        base_dir = Path("data/crossdocked_pocket10")
        
        # Check first 5 entries
        for i, entry in enumerate(index_data[:5]):
            print(f"\nEntry {i+1}: {entry}")
            
            if len(entry) >= 4:
                pocket_file, ligand_file, receptor_file, score = entry[:4]
                
                ligand_path = base_dir / ligand_file
                pocket_path = base_dir / pocket_file
                
                print(f"   Ligand: {ligand_file} - {'✅' if ligand_path.exists() else '❌'}")
                print(f"   Pocket: {pocket_file} - {'✅' if pocket_path.exists() else '❌'}")
                print(f"   Score: {score}")
                
                if ligand_path.exists() and pocket_path.exists():
                    print(f"   Both files exist for entry {i+1}")
                    return True
            else:
                print(f"   Invalid entry format: {entry}")
        
        print("\n❌ No valid file pairs found in first 5 entries")
        return False
        
    except Exception as e:
        print(f"❌ Error checking sample files: {e}")
        return False

if __name__ == "__main__":
    check_data_structure()
    check_sample_files()