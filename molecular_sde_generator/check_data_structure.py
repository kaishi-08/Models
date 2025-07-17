# check_data_structure.py - Kiểm tra cấu trúc data hiện tại
import os
import pickle
import torch
from pathlib import Path

def check_data_structure():
    print("🔍 Checking data structure...")
    
    # Check main data directory
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ data/ directory not found!")
        return
    
    print(f"✅ Found data directory: {data_dir}")
    
    # Check crossdocked_pocket10
    crossdock_dir = data_dir / "crossdocked_pocket10"
    if crossdock_dir.exists():
        print(f"✅ Found crossdocked_pocket10: {crossdock_dir}")
        
        # List subdirectories
        subdirs = [d for d in crossdock_dir.iterdir() if d.is_dir()]
        print(f"   Subdirectories: {len(subdirs)}")
        for i, subdir in enumerate(subdirs[:5]):  # Show first 5
            print(f"     {i+1}. {subdir.name}")
        if len(subdirs) > 5:
            print(f"     ... and {len(subdirs) - 5} more")
        
        # Check index.pkl
        index_file = crossdock_dir / "index.pkl"
        if index_file.exists():
            print(f"✅ Found index.pkl")
            try:
                with open(index_file, 'rb') as f:
                    index_data = pickle.load(f)
                print(f"   Index entries: {len(index_data)}")
                
                # Show sample entries
                if hasattr(index_data, 'keys'):
                    print("   Sample keys:", list(index_data.keys())[:3])
                elif isinstance(index_data, list):
                    print("   Sample entries:", index_data[:2])
                else:
                    print(f"   Index type: {type(index_data)}")
                    
            except Exception as e:
                print(f"❌ Error reading index.pkl: {e}")
        else:
            print("❌ index.pkl not found")
    else:
        print("❌ crossdocked_pocket10/ not found")
    
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
        print("❌ split_by_name.pt not found")
    
    # Check processed data
    processed_dir = data_dir / "processed"
    if processed_dir.exists():
        print(f"✅ Found processed directory")
        processed_files = list(processed_dir.glob("*.pkl"))
        print(f"   Processed files: {len(processed_files)}")
        for file in processed_files:
            print(f"     - {file.name}")
    else:
        print("❌ processed/ directory not found - need to create")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    check_data_structure()