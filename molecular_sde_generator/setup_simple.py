# setup_simple.py - Simple setup and validation
import subprocess
import sys
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch', 'torch_geometric', 'rdkit', 'numpy', 
        'scipy', 'pandas', 'biopython', 'e3nn', 'tqdm', 'pyyaml'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"   ❌ {package}")
    
    if missing:
        print(f"\n❌ Missing packages: {missing}")
        print("Install with:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True

def check_data():
    """Check if data is available"""
    print("\n🔍 Checking data...")
    
    # Check raw data
    data_dir = Path("data/crossdocked_pocket10")
    index_file = data_dir / "index.pkl"
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    if not index_file.exists():
        print(f"❌ Index file not found: {index_file}")
        return False
    
    # Check data size
    try:
        import pickle
        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)
        data_size = len(index_data)
        print(f"✅ Found data directory: {data_dir}")
        print(f"✅ Found index file with {data_size} entries")
        
        # Check if we have subdirectories
        subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
        print(f"✅ Found {len(subdirs)} subdirectories")
        
    except Exception as e:
        print(f"❌ Error reading index file: {e}")
        return False
    
    # Check processed data
    processed_dir = Path("data/processed")
    train_file = processed_dir / "train.pkl"
    val_file = processed_dir / "val.pkl"
    
    if train_file.exists() and val_file.exists():
        print(f"✅ Found processed data: {processed_dir}")
        
        # Check processed data size
        try:
            import pickle
            with open(train_file, 'rb') as f:
                train_data = pickle.load(f)
            with open(val_file, 'rb') as f:
                val_data = pickle.load(f)
            
            print(f"   Train samples: {len(train_data)}")
            print(f"   Val samples: {len(val_data)}")
            
        except Exception as e:
            print(f"⚠️  Error checking processed data: {e}")
        
        return True
    else:
        print(f"❌ Processed data not found: {processed_dir}")
        print("   Need to run preprocessing")
        return False

def run_preprocessing():
    """Run preprocessing"""
    print("\n🔄 Running preprocessing...")
    
    try:
        # First check if we have a lot of data
        index_file = Path("data/crossdocked_pocket10/index.pkl")
        if index_file.exists():
            import pickle
            with open(index_file, 'rb') as f:
                index_data = pickle.load(f)
            data_size = len(index_data)
            print(f"   Found {data_size} entries in dataset")
            
            # Use different max_samples based on data size
            if data_size > 100000:
                max_samples = 5000  # Use 5k samples for large dataset
                print(f"   Using {max_samples} samples for large dataset")
            else:
                max_samples = 1000  # Use 1k samples for smaller dataset
        else:
            max_samples = 1000
        
        result = subprocess.run([
            sys.executable, 'preprocess_crossdock_data.py',
            '--max_samples', str(max_samples)
        ], capture_output=True, text=True, timeout=7200)  # 2 hours timeout
        
        if result.returncode == 0:
            print("✅ Preprocessing completed")
            return True
        else:
            print(f"❌ Preprocessing failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Preprocessing timed out")
        return False
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return False

def run_test_training():
    """Run test training"""
    print("\n🧪 Running test training...")
    
    try:
        result = subprocess.run([
            sys.executable, 'train_simple.py',
            '--test',
            '--epochs', '2'
        ], capture_output=False, text=True, timeout=1800)  # 30 minutes
        
        if result.returncode == 0:
            print("✅ Test training completed")
            return True
        else:
            print("❌ Test training failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test training timed out")
        return False
    except Exception as e:
        print(f"❌ Test training error: {e}")
        return False

def main():
    print("🔬 CrossDock Molecular SDE - Simple Setup")
    print("=" * 50)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        return
    
    # Step 2: Check data
    data_ready = check_data()
    
    # Step 3: Run preprocessing if needed
    if not data_ready:
        print("\n📋 Data not ready, running preprocessing...")
        if not run_preprocessing():
            print("❌ Setup failed at preprocessing step")
            return
    
    # Step 4: Run test training
    print("\n📋 Running test training...")
    if not run_test_training():
        print("❌ Test training failed")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run full training:")
    print("   python train_simple.py")
    print("2. Run with more epochs:")
    print("   python train_simple.py --epochs 50")
    print("3. Run with specific GPU:")
    print("   python train_simple.py --gpu 0")

if __name__ == "__main__":
    main()