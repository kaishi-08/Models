# run_complete_pipeline.py - Master script chạy toàn bộ pipeline

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
import yaml

class MolecularSDEPipeline:
    """Master pipeline để chạy toàn bộ workflow"""
    
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        
    def check_dependencies(self):
        """Kiểm tra dependencies"""
        print("🔍 Checking dependencies...")
        
        required_packages = [
            'torch', 'torch_geometric', 'rdkit', 'numpy', 
            'scipy', 'pandas', 'biopython', 'e3nn', 'wandb', 'tqdm'
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
        
        print("✅ All dependencies satisfied")
        return True
    
    def setup_directories(self):
        """Setup project directories"""
        print("\n📁 Setting up directories...")
        
        directories = [
            "data/raw/crossdock2020",
            "data/processed",
            "models",
            "logs", 
            "training_outputs",
            "generated_molecules",
            "evaluation_results",
            "data_analysis"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {directory}")
        
        return True
    
    def download_data(self):
        """Download CrossDock data"""
        if self.args.skip_download:
            print("\n⏭️  Skipping data download")
            return True
        
        print("\n📥 Downloading CrossDock data...")
        
        # Check if data already exists
        index_file = Path("data/raw/crossdock2020/index.pkl")
        if index_file.exists():
            print("   ✅ Data already exists, skipping download")
            return True
        
        try:
            result = subprocess.run([
                sys.executable, 'scripts/download_crossdock.py',
                '--setup_dirs'
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                print("   ✅ Data download completed")
                return True
            else:
                print(f"   ❌ Download failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("   ⚠️  Download timed out")
            return False
        except Exception as e:
            print(f"   ❌ Download error: {e}")
            return False
    
    def preprocess_data(self):
        """Preprocess data"""
        if self.args.skip_preprocessing:
            print("\n⏭️  Skipping data preprocessing")
            return True
        
        print("\n🔄 Preprocessing data...")
        
        # Check if processed data exists
        train_file = Path("data/processed/train.pkl")
        if train_file.exists() and not self.args.force_preprocess:
            print("   ✅ Processed data exists, skipping preprocessing")
            print("   Use --force_preprocess to reprocess")
            return True
        
        try:
            cmd = [sys.executable, 'scripts/preprocess_crossdock_data.py']
            
            if self.args.max_atoms:
                cmd.extend(['--max_atoms', str(self.args.max_atoms)])
            if self.args.pocket_radius:
                cmd.extend(['--pocket_radius', str(self.args.pocket_radius)])
            
            result = subprocess.run(cmd, capture_output=False, text=True, timeout=7200)  # 2 hours
            
            if result.returncode == 0:
                print("   ✅ Data preprocessing completed")
                return True
            else:
                print("   ❌ Preprocessing failed")
                return False
                
        except subprocess.TimeoutExpired:
            print("   ⚠️  Preprocessing timed out")
            return False
        except Exception as e:
            print(f"   ❌ Preprocessing error: {e}")
            return False
    
    def validate_data(self):
        """Validate data flow"""
        print("\n🔍 Validating data flow...")
        
        try:
            result = subprocess.run([
                sys.executable, 'scripts/check_data_flow.py'
            ], capture_output=False, text=True, timeout=1800)  # 30 minutes
            
            if result.returncode == 0:
                print("   ✅ Data validation passed")
                return True
            else:
                print("   ❌ Data validation failed")
                return False
                
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            return False
    
    def run_training(self):
        """Run training"""
        print("\n🚀 Starting training...")
        
        # Choose config based on arguments
        if self.args.quick_test:
            config_file = 'config/simple_training_config.yaml'
        else:
            config_file = self.args.config
        
        try:
            cmd = [sys.executable, 'scripts/train_comprehensive.py', 
                   '--config', config_file]
            
            if self.args.wandb:
                cmd.append('--wandb')
            if self.args.debug:
                cmd.append('--debug')
            if self.args.quick_test:
                cmd.append('--quick_test')
            if self.args.gpu is not None:
                cmd.extend(['--gpu', str(self.args.gpu)])
            if self.args.batch_size:
                cmd.extend(['--batch_size', str(self.args.batch_size)])
            if self.args.lr:
                cmd.extend(['--lr', str(self.args.lr)])
            if self.args.epochs:
                cmd.extend(['--epochs', str(self.args.epochs)])
            
            print(f"   Command: {' '.join(cmd)}")
            
            # Run training
            result = subprocess.run(cmd, capture_output=False, text=True)
            
            if result.returncode == 0:
                print("   ✅ Training completed successfully")
                return True
            else:
                print("   ❌ Training failed")
                return False
                
        except KeyboardInterrupt:
            print("   ⚠️  Training interrupted by user")
            return False
        except Exception as e:
            print(f"   ❌ Training error: {e}")
            return False
    
    def evaluate_model(self):
        """Evaluate trained model"""
        if self.args.skip_evaluation:
            print("\n⏭️  Skipping model evaluation")
            return True
        
        print("\n📊 Evaluating model...")
        
        # Find latest checkpoint
        checkpoint_dirs = list(Path("training_outputs").glob("*/checkpoints"))
        if not checkpoint_dirs:
            print("   ❌ No checkpoints found")
            return False
        
        latest_checkpoint = None
        for checkpoint_dir in checkpoint_dirs:
            checkpoint_files = list(checkpoint_dir.glob("best_model*.pth"))
            if checkpoint_files:
                latest_checkpoint = checkpoint_files[0]
                break
        
        if not latest_checkpoint:
            print("   ❌ No checkpoint files found")
            return False
        
        try:
            result = subprocess.run([
                sys.executable, 'scripts/evaluate_model.py',
                '--checkpoint', str(latest_checkpoint),
                '--num_samples', str(self.args.eval_samples)
            ], capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                print("   ✅ Model evaluation completed")
                print(result.stdout)
                return True
            else:
                print(f"   ❌ Evaluation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"   ❌ Evaluation error: {e}")
            return False
    
    def generate_molecules(self):
        """Generate sample molecules"""
        if self.args.skip_generation:
            print("\n⏭️  Skipping molecule generation")
            return True
        
        print("\n🧪 Generating molecules...")
        
        # Find latest checkpoint
        checkpoint_dirs = list(Path("training_outputs").glob("*/checkpoints"))
        if not checkpoint_dirs:
            print("   ❌ No checkpoints found")
            return False
        
        latest_checkpoint = None
        for checkpoint_dir in checkpoint_dirs:
            checkpoint_files = list(checkpoint_dir.glob("best_model*.pth"))
            if checkpoint_files:
                latest_checkpoint = checkpoint_files[0]
                break
        
        if not latest_checkpoint:
            print("   ❌ No checkpoint files found")
            return False
        
        try:
            result = subprocess.run([
                sys.executable, 'scripts/generate_molecules.py',
                '--checkpoint', str(latest_checkpoint),
                '--num_molecules', str(self.args.gen_molecules)
            ], capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                print("   ✅ Molecule generation completed")
                return True
            else:
                print(f"   ❌ Generation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"   ❌ Generation error: {e}")
            return False
    
    def generate_report(self):
        """Generate final report"""
        print("\n📝 Generating final report...")
        
        total_time = time.time() - self.start_time
        
        report = []
        report.append("# Molecular SDE Generator Pipeline Report")
        report.append("=" * 50)
        report.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total time: {total_time / 3600:.2f} hours")
        report.append("")
        
        # Check outputs
        report.append("## Generated Outputs")
        
        output_dirs = [
            ("Processed Data", "data/processed"),
            ("Training Outputs", "training_outputs"),
            ("Generated Molecules", "generated_molecules"),
            ("Evaluation Results", "evaluation_results"),
            ("Data Analysis", "data_analysis")
        ]
        
        for name, dir_path in output_dirs:
            if Path(dir_path).exists():
                files = list(Path(dir_path).rglob("*"))
                report.append(f"- {name}: {len(files)} files")
            else:
                report.append(f"- {name}: Directory not found")
        
        report.append("")
        report.append("## Next Steps")
        report.append("1. Review training outputs in training_outputs/")
        report.append("2. Check generated molecules in generated_molecules/")
        report.append("3. Analyze evaluation results in evaluation_results/")
        report.append("4. Use model for further experiments")
        
        # Save report
        report_path = Path("pipeline_report.txt")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"   ✅ Report saved to {report_path}")
        
        # Print summary
        print("\n" + "\n".join(report))
    
    def run_complete_pipeline(self):
        """Run complete pipeline"""
        print("🔬 Molecular SDE Generator - Complete Pipeline")
        print("=" * 60)
        print(f"Arguments: {vars(self.args)}")
        
        pipeline_steps = [
            ("Check Dependencies", self.check_dependencies, True),
            ("Setup Directories", self.setup_directories, True),
            ("Download Data", self.download_data, not self.args.skip_download),
            ("Preprocess Data", self.preprocess_data, not self.args.skip_preprocessing),
            ("Validate Data", self.validate_data, True),
            ("Run Training", self.run_training, not self.args.skip_training),
            ("Evaluate Model", self.evaluate_model, not self.args.skip_evaluation),
            ("Generate Molecules", self.generate_molecules, not self.args.skip_generation),
            ("Generate Report", self.generate_report, True)
        ]
        
        results = {}
        
        for step_name, step_func, should_run in pipeline_steps:
            if not should_run:
                print(f"\n⏭️  Skipping {step_name}")
                results[step_name] = "Skipped"
                continue
            
            print(f"\n{'='*20} {step_name} {'='*20}")
            try:
                result = step_func()
                results[step_name] = "Success" if result else "Failed"
                
                if not result and not self.args.continue_on_failure:
                    print(f"❌ {step_name} failed, stopping pipeline")
                    break
                    
            except KeyboardInterrupt:
                print(f"\n⚠️  Pipeline interrupted by user at {step_name}")
                results[step_name] = "Interrupted"
                break
            except Exception as e:
                print(f"❌ {step_name} failed with exception: {e}")
                results[step_name] = "Error"
                if not self.args.continue_on_failure:
                    break
        
        # Final summary
        print(f"\n{'='*60}")
        print("📋 PIPELINE SUMMARY")
        print("=" * 60)
        
        for step_name, result in results.items():
            status_emoji = {
                "Success": "✅",
                "Failed": "❌", 
                "Skipped": "⏭️",
                "Interrupted": "⚠️",
                "Error": "💥"
            }
            emoji = status_emoji.get(result, "❓")
            print(f"{emoji} {step_name}: {result}")
        
        total_time = time.time() - self.start_time
        print(f"\n⏱️  Total pipeline time: {total_time / 3600:.2f} hours")
        
        success_count = sum(1 for r in results.values() if r == "Success")
        total_count = len([r for r in results.values() if r != "Skipped"])
        
        if success_count == total_count:
            print("\n🎉 Pipeline completed successfully!")
        else:
            print(f"\n⚠️  Pipeline completed with {total_count - success_count} failures")

def main():
    parser = argparse.ArgumentParser(description='Molecular SDE Generator - Complete Pipeline')
    
    # Data arguments
    parser.add_argument('--skip_download', action='store_true',
                       help='Skip data download')
    parser.add_argument('--skip_preprocessing', action='store_true', 
                       help='Skip data preprocessing')
    parser.add_argument('--force_preprocess', action='store_true',
                       help='Force data preprocessing even if exists')
    parser.add_argument('--max_atoms', type=int, default=50,
                       help='Maximum atoms per molecule')
    parser.add_argument('--pocket_radius', type=float, default=10.0,
                       help='Pocket radius around ligand')
    
    # Training arguments
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Training config file')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable wandb logging')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode')
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test mode')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device ID')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs')
    
    # Evaluation arguments
    parser.add_argument('--skip_evaluation', action='store_true',
                       help='Skip model evaluation')
    parser.add_argument('--eval_samples', type=int, default=1000,
                       help='Number of samples for evaluation')
    
    # Generation arguments
    parser.add_argument('--skip_generation', action='store_true',
                       help='Skip molecule generation')
    parser.add_argument('--gen_molecules', type=int, default=100,
                       help='Number of molecules to generate')
    
    # Pipeline control
    parser.add_argument('--continue_on_failure', action='store_true',
                       help='Continue pipeline even if steps fail')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = MolecularSDEPipeline(args)
    pipeline.run_complete_pipeline()

if __name__ == "__main__":
    main()