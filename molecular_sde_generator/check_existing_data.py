# check_existing_data.py - Kiểm tra và validate data hiện có

import os
import pickle
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm

def analyze_data_structure():
    """Phân tích cấu trúc data hiện có"""
    print("🔍 Analyzing existing data structure...")
    
    data_dir = Path("data")
    
    # Scan toàn bộ data directory
    print(f"\n📁 Data directory structure:")
    for root, dirs, files in os.walk(data_dir):
        level = root.replace(str(data_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        
        # Show first few files and total count
        if len(files) > 10:
            for file in files[:5]:
                print(f"{subindent}{file}")
            print(f"{subindent}... and {len(files)-5} more files")
        else:
            for file in files:
                print(f"{subindent}{file}")
    
    return data_dir

def check_crossdock_data():
    """Kiểm tra CrossDock data format"""
    print("\n🧬 Checking CrossDock data format...")
    
    # Common paths to check
    possible_paths = [
        "data/crossdocked_pocket10",
        "data/raw/crossdock2020", 
        "data/crossdock2020",
        "data/crossdock",
        "data/"
    ]
    
    found_data = []
    
    for path in possible_paths:
        if Path(path).exists():
            print(f"✅ Found directory: {path}")
            
            # Check for common CrossDock files
            path_obj = Path(path)
            
            # Look for index files
            index_files = list(path_obj.glob("**/index.pkl")) + list(path_obj.glob("**/index.csv"))
            for idx_file in index_files:
                found_data.append({"type": "index", "path": idx_file})
                print(f"   📄 Index file: {idx_file}")
            
            # Look for test_list files (CrossDock format)
            test_files = list(path_obj.glob("**/test_list.tsv"))
            for test_file in test_files:
                found_data.append({"type": "test_list", "path": test_file})
                print(f"   📄 Test list: {test_file}")
            
            # Look for PDB files
            pdb_files = list(path_obj.glob("**/*.pdb"))[:5]  # First 5 only
            if pdb_files:
                print(f"   🧪 Found {len(list(path_obj.glob('**/*.pdb')))} PDB files (showing first 5):")
                for pdb in pdb_files:
                    print(f"      {pdb}")
            
            # Look for SDF files  
            sdf_files = list(path_obj.glob("**/*.sdf"))[:5]  # First 5 only
            if sdf_files:
                print(f"   💊 Found {len(list(path_obj.glob('**/*.sdf')))} SDF files (showing first 5):")
                for sdf in sdf_files:
                    print(f"      {sdf}")
                    
    return found_data

def check_processed_data():
    """Kiểm tra processed data"""
    print("\n⚙️  Checking processed data...")
    
    processed_dir = Path("data/processed")
    
    if not processed_dir.exists():
        print("❌ No processed data directory found")
        return False
    
    expected_files = ["train.pkl", "val.pkl", "test.pkl"]
    found_files = []
    
    for file_name in expected_files:
        file_path = processed_dir / file_name
        if file_path.exists():
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"✅ {file_name}: {len(data)} samples")
                found_files.append(file_name)
                
                # Analyze first sample
                if len(data) > 0:
                    sample = data[0]
                    print(f"   Sample keys: {list(sample.keys())}")
                    
                    if 'ligand' in sample:
                        ligand = sample['ligand']
                        print(f"   Ligand atoms: {ligand['atom_features'].shape[0] if 'atom_features' in ligand else 'N/A'}")
                        print(f"   Ligand bonds: {ligand['edge_index'].shape[1] if 'edge_index' in ligand else 'N/A'}")
                    
                    if 'pocket' in sample:
                        pocket = sample['pocket']
                        print(f"   Pocket atoms: {pocket['atom_features'].shape[0] if 'atom_features' in pocket else 'N/A'}")
                        
            except Exception as e:
                print(f"❌ Error loading {file_name}: {e}")
        else:
            print(f"❌ Missing: {file_name}")
    
    return len(found_files) == len(expected_files)

def analyze_crossdock_format(data_info):
    """Phân tích format cụ thể của CrossDock data"""
    print("\n🔬 Analyzing CrossDock format...")
    
    for item in data_info:
        if item["type"] == "index":
            try:
                with open(item["path"], 'rb') as f:
                    index_data = pickle.load(f)
                
                print(f"📄 Index file analysis ({item['path']}):")
                print(f"   Total entries: {len(index_data)}")
                
                if len(index_data) > 0:
                    sample = index_data[0]
                    print(f"   Sample keys: {list(sample.keys()) if isinstance(sample, dict) else 'Not a dict'}")
                    
                    if isinstance(sample, dict):
                        for key, value in sample.items():
                            if isinstance(value, str) and len(value) < 100:
                                print(f"   {key}: {value}")
                
            except Exception as e:
                print(f"❌ Error analyzing index: {e}")
                
        elif item["type"] == "test_list":
            try:
                df = pd.read_csv(item["path"], sep='\t')
                print(f"📄 Test list analysis ({item['path']}):")
                print(f"   Columns: {list(df.columns)}")
                print(f"   Rows: {len(df)}")
                print(f"   Sample data:")
                print(df.head(3).to_string(index=False))
                
            except Exception as e:
                print(f"❌ Error analyzing test list: {e}")

def suggest_preprocessing_approach(data_info):
    """Đề xuất cách preprocess dựa trên data format"""
    print("\n💡 Preprocessing suggestions...")
    
    has_processed = check_processed_data()
    
    if has_processed:
        print("✅ Processed data already exists!")
        print("   You can proceed directly to training:")
        print("   python scripts/train_enhanced.py --quick_test")
        return
    
    print("❌ No processed data found. Need to preprocess raw data.")
    
    # Analyze what we have
    has_index = any(item["type"] == "index" for item in data_info)
    has_test_list = any(item["type"] == "test_list" for item in data_info)
    
    if has_index:
        print("\n📋 Approach 1: Use existing index.pkl")
        print("   - Modify preprocess_crossdock_data.py to use your index file")
        print("   - Update data path in the script")
        
    elif has_test_list:
        print("\n📋 Approach 2: Use test_list.tsv format")
        print("   - This looks like standard CrossDock format")
        print("   - Need to adapt preprocessing script for this format")
        
    else:
        print("\n📋 Approach 3: Scan directory structure")
        print("   - Auto-discover protein-ligand pairs")
        print("   - Match .pdb and .sdf files")

def create_custom_preprocessor():
    """Tạo custom preprocessor dựa trên data format"""
    print("\n🛠️  Creating custom preprocessor...")
    
    # Scan for actual data structure
    data_formats = {
        "pdb_files": len(list(Path("data").glob("**/*.pdb"))),
        "sdf_files": len(list(Path("data").glob("**/*.sdf"))),
        "mol2_files": len(list(Path("data").glob("**/*.mol2"))),
        "has_index": len(list(Path("data").glob("**/index.pkl"))) > 0,
        "has_test_list": len(list(Path("data").glob("**/test_list.tsv"))) > 0
    }
    
    print("📊 Data format summary:")
    for format_type, count in data_formats.items():
        if isinstance(count, bool):
            print(f"   {format_type}: {'Yes' if count else 'No'}")
        else:
            print(f"   {format_type}: {count}")
    
    # Create appropriate preprocessor
    if data_formats["has_test_list"]:
        create_test_list_preprocessor()
    elif data_formats["has_index"]:
        print("   Use existing preprocess_crossdock_data.py with minor modifications")
    else:
        create_directory_scanner_preprocessor()

def create_test_list_preprocessor():
    """Tạo preprocessor cho format test_list.tsv"""
    print("\n📝 Creating test_list.tsv preprocessor...")
    
    preprocessor_code = '''
# preprocess_test_list_format.py
import pandas as pd
import os
from pathlib import Path
import pickle
from tqdm import tqdm

def process_test_list_format():
    """Process CrossDock data from test_list.tsv format"""
    
    # Find test_list.tsv
    test_list_files = list(Path("data").glob("**/test_list.tsv"))
    if not test_list_files:
        print("No test_list.tsv found!")
        return
    
    test_list_file = test_list_files[0]
    print(f"Using test list: {test_list_file}")
    
    # Read test list
    df = pd.read_csv(test_list_file, sep='\\t')
    print(f"Found {len(df)} entries")
    
    # Process each entry
    processed_data = []
    base_dir = test_list_file.parent
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Adjust these column names based on your actual test_list.tsv
            protein_file = base_dir / row['protein_path']  # Adjust column name
            ligand_file = base_dir / row['ligand_path']    # Adjust column name
            
            if protein_file.exists() and ligand_file.exists():
                complex_data = process_complex_pair(protein_file, ligand_file)
                if complex_data:
                    processed_data.append(complex_data)
        
        except Exception as e:
            continue
    
    # Save processed data
    save_processed_data(processed_data)

def process_complex_pair(protein_file, ligand_file):
    """Process a protein-ligand pair"""
    # Use your existing CrossDockDataProcessor logic here
    pass

def save_processed_data(data):
    """Save to train/val/test splits"""
    # Your existing save logic
    pass

if __name__ == "__main__":
    process_test_list_format()
'''
    
    with open("preprocess_test_list_format.py", "w") as f:
        f.write(preprocessor_code)
    
    print("   ✅ Created preprocess_test_list_format.py")
    print("   📝 You need to customize column names in the script")

def create_directory_scanner_preprocessor():
    """Tạo preprocessor scan directory structure"""
    print("\n📝 Creating directory scanner preprocessor...")
    
    print("   This approach will:")
    print("   1. Scan for all .pdb and .sdf/.mol2 files")
    print("   2. Try to match protein-ligand pairs by naming")
    print("   3. Process matched pairs")
    
    # You can implement this if needed

def main():
    print("🔬 Data Analysis for Molecular SDE Generator")
    print("=" * 50)
    
    # Analyze data structure
    data_dir = analyze_data_structure()
    
    # Check for CrossDock specific formats
    data_info = check_crossdock_data()
    
    # Analyze specific formats
    if data_info:
        analyze_crossdock_format(data_info)
    
    # Check processed data
    processed_exists = check_processed_data()
    
    # Suggest preprocessing approach
    suggest_preprocessing_approach(data_info)
    
    # Create custom preprocessor if needed
    if not processed_exists:
        create_custom_preprocessor()
    
    print("\n" + "=" * 50)
    print("📋 NEXT STEPS:")
    print("=" * 50)
    
    if processed_exists:
        print("✅ Data ready for training!")
        print("   Run: python setup_and_validate.py")
        print("   Then: python scripts/train_enhanced.py --quick_test")
    else:
        print("❌ Need to preprocess data first")
        print("   1. Check the suggested preprocessing approach above")
        print("   2. Modify preprocess script for your data format") 
        print("   3. Run preprocessing")
        print("   4. Then run training")

if __name__ == "__main__":
    main()
    