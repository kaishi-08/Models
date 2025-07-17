# scripts/check_data_flow.py - Script kiểm tra flow data chi tiết

import os
import sys
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

# Add src to path
sys.path.append('src')

from src.data.molecular_dataset import CrossDockMolecularDataset, collate_crossdock_data
from src.data.data_loaders import CrossDockDataLoader
from src.utils.molecular_utils import MolecularMetrics
from rdkit import Chem

class DataFlowChecker:
    """Comprehensive data flow checker"""
    
    def __init__(self):
        self.results = {}
        
    def check_raw_data_structure(self):
        """Kiểm tra cấu trúc raw data"""
        print("🔍 Checking raw data structure...")
        
        raw_path = Path("data/raw/crossdock2020")
        
        # Check if raw data exists
        if not raw_path.exists():
            print("❌ Raw data directory not found!")
            print("   Expected: data/raw/crossdock2020/")
            print("   Run: python scripts/download_crossdock.py")
            return False
        
        # Check index file
        index_file = raw_path / "index.pkl"
        if index_file.exists():
            try:
                with open(index_file, 'rb') as f:
                    index_data = pickle.load(f)
                print(f"✅ Index file found: {len(index_data)} entries")
                
                # Sample some entries
                sample_keys = list(index_data.keys())[:5]
                print("   Sample entries:")
                for key in sample_keys:
                    print(f"      {key}: {type(index_data[key])}")
                
                self.results['raw_index_count'] = len(index_data)
                return True
                
            except Exception as e:
                print(f"❌ Error loading index: {e}")
                return False
        else:
            print("❌ Index file not found!")
            return False
    
    def check_processed_data(self):
        """Kiểm tra processed data"""
        print("\n📊 Checking processed data...")
        
        processed_files = [
            "data/processed/train.pkl",
            "data/processed/val.pkl", 
            "data/processed/test.pkl"
        ]
        
        all_good = True
        total_samples = 0
        
        for file_path in processed_files:
            if Path(file_path).exists():
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    print(f"✅ {Path(file_path).name}: {len(data)} samples")
                    total_samples += len(data)
                    
                    # Check sample structure
                    if data:
                        sample = data[0]
                        print(f"   Sample keys: {list(sample.keys())}")
                        
                        if 'ligand' in sample:
                            ligand = sample['ligand']
                            print(f"   Ligand features: {list(ligand.keys())}")
                            if 'atom_features' in ligand:
                                print(f"   Atoms: {len(ligand['atom_features'])}")
                            if 'positions' in ligand:
                                print(f"   Positions shape: {np.array(ligand['positions']).shape}")
                        
                        if 'pocket' in sample:
                            pocket = sample['pocket']
                            print(f"   Pocket features: {list(pocket.keys())}")
                            if 'atom_features' in pocket:
                                print(f"   Pocket atoms: {len(pocket['atom_features'])}")
                    
                except Exception as e:
                    print(f"❌ Error loading {file_path}: {e}")
                    all_good = False
            else:
                print(f"❌ {file_path} not found")
                all_good = False
        
        if all_good:
            print(f"✅ Total processed samples: {total_samples}")
            self.results['processed_samples'] = total_samples
        
        return all_good
    
    def test_dataset_loading(self):
        """Test dataset loading"""
        print("\n🔄 Testing dataset loading...")
        
        try:
            # Test train dataset
            train_dataset = CrossDockMolecularDataset(
                data_path="data/processed/train.pkl",
                include_pocket=True,
                max_atoms=50,
                augment=False
            )
            
            print(f"✅ Train dataset loaded: {len(train_dataset)} samples")
            
            # Test loading a sample
            sample = train_dataset[0]
            print(f"   Sample type: {type(sample)}")
            print(f"   Sample attributes: {[attr for attr in dir(sample) if not attr.startswith('_')]}")
            
            if hasattr(sample, 'x'):
                print(f"   Atom features shape: {sample.x.shape}")
            if hasattr(sample, 'pos'):
                print(f"   Positions shape: {sample.pos.shape}")
            if hasattr(sample, 'edge_index'):
                print(f"   Edge index shape: {sample.edge_index.shape}")
            if hasattr(sample, 'pocket_x'):
                print(f"   Pocket features shape: {sample.pocket_x.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Dataset loading error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_dataloader(self):
        """Test dataloader với batching"""
        print("\n📦 Testing dataloader with batching...")
        
        try:
            # Create simple config
            config = {
                'data': {
                    'train_path': 'data/processed/train.pkl',
                    'val_path': 'data/processed/val.pkl',
                    'batch_size': 4,
                    'num_workers': 1,
                    'shuffle': True,
                    'pin_memory': False
                },
                'include_pocket': True,
                'max_atoms': 50,
                'augment': False
            }
            
            # Create dataloader
            train_loader = CrossDockDataLoader.create_train_loader(config)
            print(f"✅ Dataloader created: {len(train_loader)} batches")
            
            # Test loading batches
            for i, batch in enumerate(train_loader):
                if batch is None:
                    print(f"   ⚠️  Batch {i} is None")
                    continue
                
                print(f"   Batch {i}:")
                print(f"      Molecules: {batch.batch.max().item() + 1}")
                print(f"      Total atoms: {batch.x.shape[0]}")
                print(f"      Total bonds: {batch.edge_index.shape[1]}")
                
                if hasattr(batch, 'pocket_x'):
                    print(f"      Pocket atoms: {batch.pocket_x.shape[0]}")
                    print(f"      ✅ Pocket data available")
                else:
                    print(f"      ❌ No pocket data")
                
                # Test batch structure
                expected_attrs = ['x', 'pos', 'edge_index', 'edge_attr', 'batch']
                for attr in expected_attrs:
                    if hasattr(batch, attr):
                        tensor = getattr(batch, attr)
                        print(f"      {attr}: {tensor.shape}")
                    else:
                        print(f"      ❌ Missing {attr}")
                
                if i >= 2:  # Test first 3 batches
                    break
            
            return True
            
        except Exception as e:
            print(f"❌ Dataloader error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def analyze_data_statistics(self):
        """Phân tích thống kê data"""
        print("\n📈 Analyzing data statistics...")
        
        try:
            with open("data/processed/train.pkl", 'rb') as f:
                train_data = pickle.load(f)
            
            # Molecular statistics
            atom_counts = []
            bond_counts = []
            pocket_atom_counts = []
            smiles_list = []
            
            for sample in tqdm(train_data[:1000], desc="Analyzing samples"):  # Analyze first 1000
                if 'ligand' in sample:
                    ligand = sample['ligand']
                    
                    if 'atom_features' in ligand:
                        atom_counts.append(len(ligand['atom_features']))
                    
                    if 'edge_index' in ligand:
                        edge_index = ligand['edge_index']
                        if len(edge_index) > 0:
                            bond_counts.append(len(edge_index[0]) // 2)  # Undirected bonds
                        else:
                            bond_counts.append(0)
                    
                    if 'smiles' in ligand:
                        smiles_list.append(ligand['smiles'])
                
                if 'pocket' in sample:
                    pocket = sample['pocket']
                    if 'atom_features' in pocket:
                        pocket_atom_counts.append(len(pocket['atom_features']))
            
            # Print statistics
            print(f"✅ Analyzed {len(atom_counts)} molecules")
            
            if atom_counts:
                print(f"   Atom count - Mean: {np.mean(atom_counts):.1f}, "
                      f"Std: {np.std(atom_counts):.1f}, "
                      f"Range: {min(atom_counts)}-{max(atom_counts)}")
            
            if bond_counts:
                print(f"   Bond count - Mean: {np.mean(bond_counts):.1f}, "
                      f"Std: {np.std(bond_counts):.1f}, "
                      f"Range: {min(bond_counts)}-{max(bond_counts)}")
            
            if pocket_atom_counts:
                print(f"   Pocket atoms - Mean: {np.mean(pocket_atom_counts):.1f}, "
                      f"Std: {np.std(pocket_atom_counts):.1f}, "
                      f"Range: {min(pocket_atom_counts)}-{max(pocket_atom_counts)}")
            
            # Analyze molecular properties
            if smiles_list:
                print(f"   Valid SMILES: {len([s for s in smiles_list if s])}/{len(smiles_list)}")
                
                valid_smiles = [s for s in smiles_list if s][:100]  # Sample 100
                if valid_smiles:
                    validity = MolecularMetrics.compute_validity(valid_smiles)
                    uniqueness = MolecularMetrics.compute_uniqueness(valid_smiles)
                    print(f"   Sample validity: {validity:.3f}")
                    print(f"   Sample uniqueness: {uniqueness:.3f}")
            
            # Store results for plotting
            self.results['atom_counts'] = atom_counts
            self.results['bond_counts'] = bond_counts
            self.results['pocket_atom_counts'] = pocket_atom_counts
            self.results['smiles_sample'] = smiles_list[:100]
            
            return True
            
        except Exception as e:
            print(f"❌ Statistics analysis error: {e}")
            return False
    
    def create_data_visualizations(self):
        """Tạo visualization cho data"""
        print("\n📊 Creating data visualizations...")
        
        if not self.results:
            print("❌ No results to visualize")
            return False
        
        try:
            # Create output directory
            os.makedirs("data_analysis", exist_ok=True)
            
            # Plot distributions
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Atom count distribution
            if 'atom_counts' in self.results:
                axes[0, 0].hist(self.results['atom_counts'], bins=30, alpha=0.7)
                axes[0, 0].set_title('Atom Count Distribution')
                axes[0, 0].set_xlabel('Number of Atoms')
                axes[0, 0].set_ylabel('Frequency')
            
            # Bond count distribution
            if 'bond_counts' in self.results:
                axes[0, 1].hist(self.results['bond_counts'], bins=30, alpha=0.7)
                axes[0, 1].set_title('Bond Count Distribution')
                axes[0, 1].set_xlabel('Number of Bonds')
                axes[0, 1].set_ylabel('Frequency')
            
            # Pocket atom distribution
            if 'pocket_atom_counts' in self.results:
                axes[1, 0].hist(self.results['pocket_atom_counts'], bins=30, alpha=0.7)
                axes[1, 0].set_title('Pocket Atom Count Distribution')
                axes[1, 0].set_xlabel('Number of Pocket Atoms')
                axes[1, 0].set_ylabel('Frequency')
            
            # Atom vs Bond scatter
            if 'atom_counts' in self.results and 'bond_counts' in self.results:
                axes[1, 1].scatter(self.results['atom_counts'], self.results['bond_counts'], alpha=0.5)
                axes[1, 1].set_title('Atoms vs Bonds')
                axes[1, 1].set_xlabel('Number of Atoms')
                axes[1, 1].set_ylabel('Number of Bonds')
            
            plt.tight_layout()
            plt.savefig('data_analysis/data_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ Visualizations saved to data_analysis/")
            return True
            
        except Exception as e:
            print(f"❌ Visualization error: {e}")
            return False
    
    def test_model_compatibility(self):
        """Test model compatibility với data"""
        print("\n🧠 Testing model compatibility...")
        
        try:
            # Import model
            from src.models.joint_2d_3d_model import Joint2D3DMolecularModel
            
            # Create model
            model = Joint2D3DMolecularModel(
                atom_types=11,  # Based on CrossDock
                bond_types=4,
                hidden_dim=64,
                pocket_dim=128,
                num_layers=2,
                max_radius=10.0
            )
            
            # Load a batch
            config = {
                'data': {
                    'train_path': 'data/processed/train.pkl',
                    'batch_size': 2,
                    'num_workers': 1,
                    'shuffle': False,
                    'pin_memory': False
                },
                'include_pocket': True,
                'max_atoms': 50
            }
            
            train_loader = CrossDockDataLoader.create_train_loader(config)
            batch = next(iter(train_loader))
            
            if batch is None:
                print("❌ No valid batch for testing")
                return False
            
            print(f"✅ Test batch loaded:")
            print(f"   Molecules: {batch.batch.max().item() + 1}")
            print(f"   Atoms: {batch.x.shape[0]}")
            print(f"   Bonds: {batch.edge_index.shape[1]}")
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                outputs = model(
                    x=batch.x,
                    pos=batch.pos,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch,
                    pocket_x=getattr(batch, 'pocket_x', None),
                    pocket_pos=getattr(batch, 'pocket_pos', None),
                    pocket_edge_index=getattr(batch, 'pocket_edge_index', None),
                    pocket_batch=getattr(batch, 'pocket_batch', None)
                )
            
            print(f"✅ Forward pass successful:")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: {value.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model compatibility error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_report(self):
        """Tạo báo cáo tổng hợp"""
        print("\n📝 Generating comprehensive report...")
        
        report = []
        report.append("# Data Flow Check Report")
        report.append("=" * 50)
        report.append("")
        
        if 'raw_index_count' in self.results:
            report.append(f"Raw data entries: {self.results['raw_index_count']}")
        
        if 'processed_samples' in self.results:
            report.append(f"Processed samples: {self.results['processed_samples']}")
        
        if 'atom_counts' in self.results:
            atoms = self.results['atom_counts']
            report.append(f"Atom count statistics:")
            report.append(f"  Mean: {np.mean(atoms):.1f}")
            report.append(f"  Std: {np.std(atoms):.1f}")
            report.append(f"  Range: {min(atoms)}-{max(atoms)}")
        
        if 'smiles_sample' in self.results:
            smiles = [s for s in self.results['smiles_sample'] if s]
            if smiles:
                validity = MolecularMetrics.compute_validity(smiles)
                uniqueness = MolecularMetrics.compute_uniqueness(smiles)
                report.append(f"Molecular quality:")
                report.append(f"  Validity: {validity:.3f}")
                report.append(f"  Uniqueness: {uniqueness:.3f}")
        
        # Save report
        with open("data_analysis/data_check_report.txt", "w") as f:
            f.write("\n".join(report))
        
        print("✅ Report saved to data_analysis/data_check_report.txt")

def main():
    print("🔬 Molecular SDE Data Flow Checker")
    print("=" * 50)
    
    checker = DataFlowChecker()
    
    # Run all checks
    checks = [
        ("Raw Data Structure", checker.check_raw_data_structure),
        ("Processed Data", checker.check_processed_data),
        ("Dataset Loading", checker.test_dataset_loading),
        ("DataLoader", checker.test_dataloader),
        ("Data Statistics", checker.analyze_data_statistics),
        ("Visualizations", checker.create_data_visualizations),
        ("Model Compatibility", checker.test_model_compatibility)
    ]
    
    results = {}
    for name, check_func in checks:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = False
    
    # Generate report
    checker.generate_report()
    
    # Summary
    print(f"\n{'='*50}")
    print("📋 SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Ready for training!")
        print("\nNext steps:")
        print("   1. python scripts/train_model.py --config config/simple_training_config.yaml")
        print("   2. python scripts/train_enhanced.py --wandb")
    else:
        print("\n⚠️  Some checks failed. Please fix issues before training.")
        print("\nTroubleshooting:")
        print("   1. Check if data is downloaded: python scripts/download_crossdock.py")
        print("   2. Run preprocessing: python scripts/preprocess_crossdock_data.py")
        print("   3. Check dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    main()