# scripts/evaluate_trained_model.py - Đánh giá model sau training
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
import sys
import yaml

# Add paths
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.models.joint_2d_3d_model import create_joint2d3d_model
from src.models.ddpm_diffusion import MolecularDDPM, MolecularDDPMModel
from src.data.data_loaders import CrossDockDataLoader
from src.inference.conditional_generator import DDPMMolecularGenerator

class ModelEvaluator:
    """Đánh giá comprehensive cho trained DDPM model"""
    
    def __init__(self, model_path, config_path):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        self.model, self.ddpm = self._load_trained_model()
        self.generator = DDPMMolecularGenerator(self.model.base_model, self.ddpm, self.device)
        
    def _load_trained_model(self):
        """Load trained model from checkpoint"""
        print(f"📥 Loading model from {self.model_path}")
        
        # Create model architecture
        base_model = create_joint2d3d_model(
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            conditioning_type=self.config['model'].get('conditioning_type', 'add')
        )
        
        ddpm = MolecularDDPM(
            num_timesteps=self.config['ddpm']['num_timesteps'],
            beta_schedule=self.config['ddpm'].get('beta_schedule', 'cosine'),
            beta_start=self.config['ddpm']['beta_start'],
            beta_end=self.config['ddpm']['beta_end']
        )
        
        model = MolecularDDPMModel(base_model, ddpm)
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"   Training epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"   Training loss: {checkpoint.get('loss', 'unknown')}")
        
        return model, ddpm
    
    def evaluate_generation_quality(self, num_molecules=100, max_atoms=50):
        """🧪 Đánh giá chất lượng generation"""
        print(f"\n🧪 EVALUATING GENERATION QUALITY")
        print("=" * 50)
        
        print(f"🔬 Generating {num_molecules} molecules...")
        
        # Generate molecules
        generated_molecules = []
        valid_smiles = []
        
        for i in range(num_molecules):
            try:
                result = self.generator.generate_molecules(
                    num_molecules=1,
                    max_atoms=max_atoms,
                    guidance_scale=1.0
                )
                
                if result['molecules']:
                    mol_data = result['molecules'][0]
                    smiles = self.generator.molecules_to_smiles([mol_data])
                    
                    if smiles[0] and smiles[0] != 'None':
                        generated_molecules.append(mol_data)
                        valid_smiles.append(smiles[0])
                        
                if (i + 1) % 10 == 0:
                    print(f"   Progress: {i + 1}/{num_molecules} ({len(valid_smiles)} valid)")
                    
            except Exception as e:
                print(f"   Generation error for molecule {i}: {e}")
                continue
        
        print(f"\n📊 Generation Results:")
        print(f"   Generated: {len(generated_molecules)} molecules")
        print(f"   Valid SMILES: {len(valid_smiles)} ({len(valid_smiles)/num_molecules*100:.1f}%)")
        
        if len(valid_smiles) > 0:
            # Analyze generated molecules
            self._analyze_generated_molecules(valid_smiles)
        
        return valid_smiles
    
    def _analyze_generated_molecules(self, smiles_list):
        """Phân tích chi tiết các molecules được generate"""
        print(f"\n🔍 MOLECULAR ANALYSIS")
        print("-" * 30)
        
        properties = {
            'molecular_weights': [],
            'logp_values': [],
            'num_atoms': [],
            'num_bonds': [],
            'num_rings': [],
            'hbd_count': [],
            'hba_count': []
        }
        
        for smiles in smiles_list[:50]:  # Analyze first 50
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                properties['molecular_weights'].append(Descriptors.MolWt(mol))
                properties['logp_values'].append(Crippen.MolLogP(mol))
                properties['num_atoms'].append(mol.GetNumAtoms())
                properties['num_bonds'].append(mol.GetNumBonds())
                properties['num_rings'].append(Descriptors.RingCount(mol))
                properties['hbd_count'].append(Descriptors.NumHDonors(mol))
                properties['hba_count'].append(Descriptors.NumHAcceptors(mol))
        
        # Print statistics
        for prop_name, values in properties.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                print(f"   {prop_name}: {mean_val:.2f} ± {std_val:.2f} [{min_val:.1f}-{max_val:.1f}]")
        
        # Drug-likeness analysis
        self._analyze_drug_likeness(smiles_list)
        
        # Save some example molecules
        self._save_example_molecules(smiles_list[:10])
    
    def _analyze_drug_likeness(self, smiles_list):
        """Phân tích drug-likeness (Lipinski's Rule of Five)"""
        print(f"\n💊 DRUG-LIKENESS ANALYSIS")
        print("-" * 30)
        
        lipinski_violations = []
        qed_scores = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Lipinski's Rule of Five
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                
                violations = sum([
                    mw > 500,
                    logp > 5,
                    hbd > 5,
                    hba > 10
                ])
                lipinski_violations.append(violations)
                
                # QED score
                try:
                    qed = Descriptors.qed(mol)
                    qed_scores.append(qed)
                except:
                    pass
        
        if lipinski_violations:
            violation_counts = np.bincount(lipinski_violations)
            print(f"   Lipinski violations:")
            for i, count in enumerate(violation_counts):
                if count > 0:
                    percentage = count / len(lipinski_violations) * 100
                    print(f"     {i} violations: {count} molecules ({percentage:.1f}%)")
        
        if qed_scores:
            mean_qed = np.mean(qed_scores)
            print(f"   Average QED score: {mean_qed:.3f}")
            good_qed = sum(1 for qed in qed_scores if qed > 0.5)
            print(f"   Molecules with QED > 0.5: {good_qed}/{len(qed_scores)} ({good_qed/len(qed_scores)*100:.1f}%)")
    
    def _save_example_molecules(self, smiles_list):
        """Save example molecules"""
        print(f"\n💾 Saving example molecules...")
        
        output_dir = Path("generated_molecules")
        output_dir.mkdir(exist_ok=True)
        
        # Save SMILES
        with open(output_dir / "generated_smiles.txt", 'w') as f:
            for i, smiles in enumerate(smiles_list):
                f.write(f"{i+1:03d}: {smiles}\n")
        
        print(f"   ✅ Saved {len(smiles_list)} example SMILES to {output_dir}/generated_smiles.txt")
    
    def evaluate_reconstruction_loss(self):
        """📐 Đánh giá reconstruction loss trên test set"""
        print(f"\n📐 EVALUATING RECONSTRUCTION LOSS")
        print("=" * 50)
        
        # Create test dataloader
        test_loader = CrossDockDataLoader.create_test_loader(self.config)
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= 50:  # Test on 50 batches
                    break
                
                if batch is None:
                    continue
                
                batch = batch.to(self.device)
                
                try:
                    # DDPM loss computation
                    loss, loss_dict = self.ddpm.compute_loss(
                        model=self.model,
                        x0=batch.pos,
                        atom_features=batch.x,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        batch=batch.batch,
                        pocket_x=getattr(batch, 'pocket_x', None),
                        pocket_pos=getattr(batch, 'pocket_pos', None),
                        pocket_edge_index=getattr(batch, 'pocket_edge_index', None),
                        pocket_batch=getattr(batch, 'pocket_batch', None)
                    )
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if (batch_idx + 1) % 10 == 0:
                        avg_loss = total_loss / num_batches
                        print(f"   Batch {batch_idx + 1}/50: Avg loss = {avg_loss:.4f}")
                        
                except Exception as e:
                    print(f"   Error in batch {batch_idx}: {e}")
                    continue
        
        if num_batches > 0:
            final_avg_loss = total_loss / num_batches
            print(f"\n📊 Final Results:")
            print(f"   Average reconstruction loss: {final_avg_loss:.4f}")
            print(f"   Evaluated on {num_batches} batches")
        else:
            print("❌ No valid batches for evaluation")
    
    def generate_report(self):
        """📄 Tạo báo cáo tổng hợp"""
        print(f"\n📄 GENERATING COMPREHENSIVE REPORT")
        print("=" * 50)
        
        report_path = Path("evaluation_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# DDPM Molecular Generator - Evaluation Report\n\n")
            f.write(f"**Model Path:** {self.model_path}\n")
            f.write(f"**Config:** {self.config_path}\n")
            f.write(f"**Evaluation Date:** {torch.datetime.now()}\n\n")
            
            f.write("## Model Architecture\n")
            f.write(f"- Hidden Dimension: {self.config['model']['hidden_dim']}\n")
            f.write(f"- Number of Layers: {self.config['model']['num_layers']}\n")
            f.write(f"- DDPM Timesteps: {self.config['ddpm']['num_timesteps']}\n")
            f.write(f"- Beta Schedule: {self.config['ddpm'].get('beta_schedule', 'cosine')}\n\n")
            
            f.write("## Next Steps\n")
            f.write("1. **Molecular Generation:** Use the generator for practical applications\n")
            f.write("2. **Fine-tuning:** Improve generation quality with specific datasets\n")
            f.write("3. **Property Optimization:** Add property-guided generation\n")
            f.write("4. **Deployment:** Create API/service for molecule generation\n")
        
        print(f"✅ Report saved to {report_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained DDPM model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='config/optimized_ddpm_config.yaml',
                       help='Path to config file')
    parser.add_argument('--num_molecules', type=int, default=100,
                       help='Number of molecules to generate for evaluation')
    
    args = parser.parse_args()
    
    print("🎯 DDPM MODEL EVALUATION")
    print("=" * 60)
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model_path, args.config)
    
    # Run evaluations
    print("🚀 Starting comprehensive evaluation...")
    
    # 1. Generation quality
    valid_smiles = evaluator.evaluate_generation_quality(args.num_molecules)
    
    # 2. Reconstruction loss
    evaluator.evaluate_reconstruction_loss()
    
    # 3. Generate report
    evaluator.generate_report()
    
    print(f"\n🎉 EVALUATION COMPLETED!")
    print(f"   Generated {len(valid_smiles)} valid molecules")
    print(f"   Check 'generated_molecules/' folder for results")
    print(f"   Read 'evaluation_report.md' for detailed analysis")

if __name__ == "__main__":
    main()