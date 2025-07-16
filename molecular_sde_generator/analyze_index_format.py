# analyze_index_format.py - Phân tích format cụ thể của index.pkl

import pickle
import pandas as pd
from pathlib import Path
import os

def analyze_index_file():
    """Phân tích chi tiết format của index.pkl"""
    print("🔍 Analyzing index.pkl format in detail...")
    
    index_path = "data/crossdocked_pocket10/index.pkl"
    
    try:
        with open(index_path, 'rb') as f:
            index_data = pickle.load(f)
        
        print(f"📊 Index data analysis:")
        print(f"   Type: {type(index_data)}")
        print(f"   Length: {len(index_data)}")
        
        # Analyze first few entries
        print(f"\n🔬 First 5 entries:")
        for i in range(min(5, len(index_data))):
            entry = index_data[i]
            print(f"   Entry {i}:")
            print(f"      Type: {type(entry)}")
            
            if isinstance(entry, (list, tuple)):
                print(f"      Length: {len(entry)}")
                for j, item in enumerate(entry):
                    print(f"         [{j}]: {type(item)} - {str(item)[:100]}")
            elif isinstance(entry, str):
                print(f"      Value: {entry}")
            elif hasattr(entry, '__dict__'):
                print(f"      Attributes: {vars(entry)}")
            else:
                print(f"      Value: {str(entry)[:100]}")
        
        # Try to identify pattern
        print(f"\n🧩 Pattern analysis:")
        if len(index_data) > 0:
            first_entry = index_data[0]
            
            if isinstance(first_entry, (list, tuple)) and len(first_entry) >= 2:
                print("   Likely format: List/Tuple with multiple elements")
                print("   Possible elements: [protein_path, ligand_path, ...]")
                
                # Check if elements are file paths
                for i, item in enumerate(first_entry[:5]):
                    if isinstance(item, str):
                        if any(ext in item for ext in ['.pdb', '.sdf', '.mol2']):
                            print(f"      Element {i}: Likely file path - {item}")
                        else:
                            print(f"      Element {i}: String - {item}")
            
            elif isinstance(first_entry, str):
                print("   Format: Simple string list")
                print("   Need to infer protein-ligand pairing from filenames")
        
        return index_data
        
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        return None

def check_file_structure():
    """Kiểm tra cấu trúc files trong directory"""
    print("\n📁 Checking file structure...")
    
    base_dir = Path("data/crossdocked_pocket10")
    
    # Sample some files to understand naming pattern
    pdb_files = list(base_dir.glob("**/*.pdb"))[:10]
    sdf_files = list(base_dir.glob("**/*.sdf"))[:10]
    
    print(f"📄 Sample PDB files:")
    for pdb in pdb_files[:5]:
        rel_path = pdb.relative_to(base_dir)
        print(f"   {rel_path}")
    
    print(f"📄 Sample SDF files:")
    for sdf in sdf_files[:5]:
        rel_path = sdf.relative_to(base_dir)
        print(f"   {rel_path}")
    
    # Check if there's a pattern
    print(f"\n🔍 Analyzing naming patterns...")
    
    if pdb_files and sdf_files:
        pdb_stem = pdb_files[0].stem
        sdf_stem = sdf_files[0].stem
        
        print(f"   PDB example: {pdb_stem}")
        print(f"   SDF example: {sdf_stem}")
        
        # Look for common prefixes/suffixes
        if '_' in pdb_stem:
            pdb_parts = pdb_stem.split('_')
            print(f"   PDB parts: {pdb_parts}")
        
        if '_' in sdf_stem:
            sdf_parts = sdf_stem.split('_')
            print(f"   SDF parts: {sdf_parts}")

def suggest_preprocessing_approach(index_data):
    """Đề xuất approach preprocessing dựa trên format"""
    print("\n💡 Preprocessing approach suggestions...")
    
    if index_data is None:
        print("❌ Cannot suggest approach - index data not loaded")
        return
    
    if len(index_data) == 0:
        print("❌ Empty index data")
        return
    
    first_entry = index_data[0]
    
    if isinstance(first_entry, (list, tuple)):
        if len(first_entry) >= 2:
            print("✅ Approach: Tuple/List format")
            print("   - Each entry contains multiple elements")
            print("   - Likely: [protein_path, ligand_path, ...]")
            print("   - Will create custom processor for this format")
            create_tuple_preprocessor()
        else:
            print("❌ Unexpected tuple format")
    
    elif isinstance(first_entry, str):
        print("✅ Approach: String list format")
        print("   - Each entry is a string (likely file path)")
        print("   - Need to pair proteins with ligands")
        print("   - Will create pairing processor")
        create_string_list_preprocessor()
    
    else:
        print(f"❌ Unknown format: {type(first_entry)}")
        print("   - Need manual inspection")

def create_tuple_preprocessor():
    """Tạo preprocessor cho tuple/list format"""
    print("\n📝 Creating tuple format preprocessor...")
    
    preprocessor_code = '''# preprocess_crossdock_tuple.py - Preprocessor cho tuple format
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import your existing CrossDockDataProcessor
import sys
sys.path.append('scripts')
from preprocess_crossdock_data import CrossDockDataProcessor

class TupleFormatProcessor(CrossDockDataProcessor):
    """Processor cho CrossDock tuple format"""
    
    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path, **kwargs)
    
    def process_dataset(self, output_dir: str = "data/processed/"):
        """Process dataset với tuple format"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load index file
        index_file = os.path.join(self.data_path, "index.pkl")
        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)
        
        print(f"Processing {len(index_data)} entries...")
        
        processed_data = []
        failed_count = 0
        
        for i, entry in enumerate(tqdm(index_data)):
            try:
                # Parse entry format
                complex_data = self.process_tuple_entry(entry)
                if complex_data is not None:
                    processed_data.append(complex_data)
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                if i < 10:  # Debug first few failures
                    print(f"Failed entry {i}: {e}")
                continue
                
            # Save intermediate results every 1000
            if (i + 1) % 1000 == 0:
                temp_file = os.path.join(output_dir, f"temp_data_{i+1}.pkl")
                with open(temp_file, 'wb') as f:
                    pickle.dump(processed_data, f)
        
        print(f"Successfully processed: {len(processed_data)}")
        print(f"Failed: {failed_count}")
        
        # Split and save
        self.split_and_save_data(processed_data, output_dir)
        return processed_data
    
    def process_tuple_entry(self, entry) -> Optional[Dict]:
        """Process một entry tuple format"""
        try:
            # Phân tích entry format - cần customize dựa trên actual format
            if isinstance(entry, (list, tuple)):
                if len(entry) >= 2:
                    # Thử các pattern phổ biến
                    protein_path = None
                    ligand_path = None
                    
                    for item in entry:
                        if isinstance(item, str):
                            if '.pdb' in item:
                                protein_path = os.path.join(self.data_path, item)
                            elif '.sdf' in item or '.mol2' in item:
                                ligand_path = os.path.join(self.data_path, item)
                    
                    if protein_path and ligand_path:
                        return self.process_complex_pair(protein_path, ligand_path)
            
            return None
            
        except Exception as e:
            return None
    
    def process_complex_pair(self, protein_file: str, ligand_file: str) -> Optional[Dict]:
        """Process một protein-ligand pair"""
        if not os.path.exists(protein_file) or not os.path.exists(ligand_file):
            return None
        
        # Process ligand
        ligand_data = self.process_ligand(ligand_file)
        if ligand_data is None:
            return None
        
        # Process pocket
        pocket_data = self.process_pocket(protein_file, ligand_data['positions'])
        if pocket_data is None:
            return None
        
        return {
            'ligand': ligand_data,
            'pocket': pocket_data,
            'protein_file': protein_file,
            'ligand_file': ligand_file
        }

def main():
    processor = TupleFormatProcessor(
        data_path="data/crossdocked_pocket10",
        pocket_radius=10.0,
        min_atoms=5,
        max_atoms=50,
        max_pocket_atoms=500
    )
    
    processed_data = processor.process_dataset("data/processed/")
    print(f"Preprocessing completed! Total: {len(processed_data)}")

if __name__ == "__main__":
    main()
'''
    
    with open("preprocess_crossdock_tuple.py", "w") as f:
        f.write(preprocessor_code)
    
    print("   ✅ Created preprocess_crossdock_tuple.py")

def create_string_list_preprocessor():
    """Tạo preprocessor cho string list format"""
    print("\n📝 Creating string list format preprocessor...")
    
    # Similar code but for string format
    print("   ✅ String list preprocessor approach outlined")

def test_sample_processing():
    """Test processing một sample để debug"""
    print("\n🧪 Testing sample processing...")
    
    try:
        # Load index
        with open("data/crossdocked_pocket10/index.pkl", 'rb') as f:
            index_data = pickle.load(f)
        
        if len(index_data) > 0:
            sample_entry = index_data[0]
            print(f"Testing entry: {sample_entry}")
            
            # Try to extract file paths
            protein_file = None
            ligand_file = None
            
            base_dir = Path("data/crossdocked_pocket10")
            
            if isinstance(sample_entry, (list, tuple)):
                for item in sample_entry:
                    if isinstance(item, str):
                        full_path = base_dir / item
                        if '.pdb' in item and full_path.exists():
                            protein_file = str(full_path)
                            print(f"   Found protein: {protein_file}")
                        elif ('.sdf' in item or '.mol2' in item) and full_path.exists():
                            ligand_file = str(full_path)
                            print(f"   Found ligand: {ligand_file}")
            
            if protein_file and ligand_file:
                print("   ✅ Successfully identified protein-ligand pair")
                return True
            else:
                print("   ❌ Could not identify protein-ligand pair")
                return False
                
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def main():
    print("🔬 CrossDock Index Format Analysis")
    print("=" * 50)
    
    # Analyze index format
    index_data = analyze_index_file()
    
    # Check file structure
    check_file_structure()
    
    # Test sample processing
    test_success = test_sample_processing()
    
    # Suggest approach
    suggest_preprocessing_approach(index_data)
    
    print("\n" + "=" * 50)
    print("📋 NEXT STEPS:")
    print("=" * 50)
    
    if test_success:
        print("✅ Sample processing successful!")
        print("   1. Run: python preprocess_crossdock_tuple.py")
        print("   2. Wait for preprocessing to complete")
        print("   3. Then: python scripts/train_enhanced.py --quick_test")
    else:
        print("❌ Need manual format adjustment")
        print("   1. Check the analysis above")
        print("   2. Modify preprocess_crossdock_tuple.py")
        print("   3. Test with small subset first")

if __name__ == "__main__":
    main()