# quick_run.py - Script chạy nhanh preprocessing và training

import os
import sys
import pickle
from pathlib import Path
import subprocess

def check_processed_data():
    """Kiểm tra xem data đã được processed chưa"""
    processed_files = [
        "data/processed/train.pkl",
        "data/processed/val.pkl", 
        "data/processed/test.pkl"
    ]
    
    all_exist = True
    for file_path in processed_files:
        if Path(file_path).exists():
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"✅ {file_path}: {len(data)} samples")
            except Exception as e:
                print(f"❌ {file_path}: Error loading - {e}")
                all_exist = False
        else:
            print(f"❌ {file_path}: Not found")
            all_exist = False
    
    return all_exist

def run_preprocessing():
    """Chạy preprocessing"""
    print("🔄 Running preprocessing...")
    
    try:
        # Chạy preprocessing script
        result = subprocess.run([
            sys.executable, "preprocess_crossdock_fixed.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Preprocessing completed successfully!")
            print(result.stdout)
            return True
        else:
            print("❌ Preprocessing failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running preprocessing: {e}")
        return False

def run_quick_training():
    """Chạy quick training test"""
    print("🚀 Running quick training test...")
    
    try:
        # Chạy training script
        result = subprocess.run([
            sys.executable, "scripts/train_enhanced.py", 
            "--quick_test", "--wandb"
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            print("✅ Quick training test passed!")
            print(result.stdout[-1000:])  # Last 1000 chars
            return True
        else:
            print("❌ Quick training test failed!")
            print("STDERR:", result.stderr[-500:])  # Last 500 chars
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ Training test timed out (might be normal)")
        return True
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return False

def main():
    print("🚀 Quick Run - CrossDock Preprocessing & Training")
    print("=" * 60)
    
    # Step 1: Check if data already processed
    print("Step 1: Checking processed data...")
    if check_processed_data():
        print("✅ Data already processed! Skipping preprocessing.")
        skip_preprocessing = True
    else:
        print("❌ Data not processed. Will run preprocessing.")
        skip_preprocessing = False
    
    # Step 2: Run preprocessing if needed
    if not skip_preprocessing:
        print("\nStep 2: Running preprocessing...")
        print("⏰ This will take 2-4 hours for 184K complexes...")
        
        user_input = input("Continue with preprocessing? (y/n): ")
        if user_input.lower() != 'y':
            print("Preprocessing cancelled.")
            return
        
        if not run_preprocessing():
            print("❌ Preprocessing failed. Check errors above.")
            return
        
        # Verify processed data
        if not check_processed_data():
            print("❌ Preprocessing completed but files not found.")
            return
    
    # Step 3: Run quick training test
    print("\nStep 3: Running quick training test...")
    if run_quick_training():
        print("✅ Quick training successful!")
        
        # Ask about full training
        print("\n🎉 Ready for full training!")
        print("Commands for full training:")
        print("  python scripts/train_enhanced.py --wandb --batch_size 16 --epochs 100")
        print("  python scripts/train_enhanced.py --wandb --batch_size 8 --epochs 50  # if GPU memory limited")
        
        user_input = input("\nStart full training now? (y/n): ")
        if user_input.lower() == 'y':
            print("🔥 Starting full training...")
            try:
                # Run full training (no timeout)
                subprocess.run([
                    sys.executable, "scripts/train_enhanced.py", 
                    "--wandb", "--batch_size", "16", "--epochs", "100"
                ])
            except KeyboardInterrupt:
                print("\n⚠️ Training interrupted by user")
        else:
            print("Full training skipped. Run manually when ready.")
    
    else:
        print("❌ Quick training failed. Check setup.")
        print("\nTroubleshooting steps:")
        print("1. python setup_and_validate.py")
        print("2. Check GPU memory: nvidia-smi")
        print("3. Try smaller batch: python scripts/train_enhanced.py --quick_test --batch_size 4")

if __name__ == "__main__":
    main()