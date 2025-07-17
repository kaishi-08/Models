# test_preprocessing.py - Test preprocessing với sample nhỏ
import os
import pickle
import torch
from pathlib import Path
import sys

def test_index_loading():
    """Test loading index.pkl"""
    print("🔍 Testing index loading...")
    
    index_file = Path("data/crossdocked_pocket10/index.pkl")
    if not index_file.exists():
        print("❌ index.pkl not found!")
        return False
    
    try:
        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)
        
        print(f"✅ Loaded {len(index_data)} entries")
        print(f"   Type: {type(index_data)}")
        
        # Sample entries
        for i, entry in enumerate(index_data[:3]):
            print(f"   Entry {i}: {entry}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        return False

def test_split_loading():
    """Test loading splits"""
    print("\n🔍 Testing split loading...")
    
    split_file = Path("data/split_by_name.pt")
    if not split_file.exists():
        print("❌ split_by_name.pt not found!")
        return False
    
    try:
        splits = torch.load(split_file)
        print(f"✅ Loaded splits: {list(splits.keys())}")
        
        for split_name, data in splits.items():
            print(f"   {split_name}: {len(data)} entries")
            if len(data) > 0:
                print(f"     Sample: {data[0]}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading splits: {e}")
        return False

def test_file_existence():
    """Test if some sample files exist"""
    print("\n🔍 Testing file existence...")
    
    # Load index to check actual files
    try:
        with open("data/crossdocked_pocket10/index.pkl", 'rb') as f:
            index_data = pickle.load(f)
        
        base_dir = Path("data/crossdocked_pocket10")
        found_count = 0
        missing_count = 0
        
        # Check first 10 entries
        for i, entry in enumerate(index_data[:10]):
            pocket_file, ligand_file = entry[0], entry[1]
            
            ligand_path = base_dir / ligand_file
            pocket_path = base_dir / pocket_file
            
            if ligand_path.exists() and pocket_path.exists():
                found_count += 1
                if i == 0:
                    print(f"✅ Sample files found:")
                    print(f"     Ligand: {ligand_file}")
                    print(f"     Pocket: {pocket_file}")
            else:
                missing_count += 1
                if missing_count == 1:
                    print(f"❌ Missing files:")
                    print(f"     Ligand: {ligand_file} (exists: {ligand_path.exists()})")
                    print(f"     Pocket: {pocket_file} (exists: {pocket_path.exists()})")
        
        print(f"\n📊 File check (first 10): Found={found_count}, Missing={missing_count}")
        return found_count > 0
        
    except Exception as e:
        print(f"❌ Error checking files: {e}")
        return False

def test_single_preprocessing():
    """Test preprocessing on single entry"""
    print("\n🧪 Testing single entry preprocessing...")
    
    try:
        # Import the preprocessor
        sys.path.append('.')
        from scripts.preprocess_crossdock_data import CrossDockPreprocessor
        
        preprocessor = CrossDockPreprocessor()
        
        # Load one entry
        with open("data/crossdocked_pocket10/index.pkl", 'rb') as f:
            index_data = pickle.load(f)
        
        # Try first few entries until we find one that works
        for i, entry in enumerate(index_data[:5]):
            print(f"   Trying entry {i}: {entry[1]}")
            
            result = preprocessor.process_complex(entry)
            if result:
                print(f"✅ Successfully processed entry {i}")
                print(f"   Ligand atoms: {result['ligand']['atom_features'].shape[0]}")
                print(f"   Ligand bonds: {result['ligand']['edge_index'].shape[1]}")
                if 'pocket' in result:
                    print(f"   Pocket atoms: {result['pocket']['atom_features'].shape[0]}")
                else:
                    print("   No pocket data")
                return True
            else:
                print(f"❌ Failed to process entry {i}")
        
        print("❌ All test entries failed")
        return False
        
    except Exception as e:
        print(f"❌ Error in preprocessing test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 Testing CrossDock Preprocessing")
    print("=" * 50)
    
    tests = [
        test_index_loading,
        test_split_loading, 
        test_file_existence,
        test_single_preprocessing
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    test_names = [
        "Index Loading",
        "Split Loading", 
        "File Existence",
        "Single Preprocessing"
    ]
    
    for name, result in zip(test_names, results):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<20}: {status}")
    
    if all(results):
        print("\n🎉 All tests passed! Ready for full preprocessing.")
        print("\nRun full preprocessing:")
        print("python preprocess_crossdock_data.py --max_samples 1000")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before full preprocessing.")

if __name__ == "__main__":
    main()