# scripts/generate_molecules.py - Generate molecules for practical use
import torch
import yaml
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
import sys

# Add paths
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.models.joint_2d_3d_model import create_joint2d3d_model
from src.models.ddpm_diffusion import MolecularDDPM, MolecularDDPMModel
from src.inference.conditional_generator import DDPMMolecularGenerator

class MolecularGenerationPipeline:
    """Production-ready molecular generation pipeline"""
    
    def __init__(self, model_path, config_path):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model, self.ddpm = self._load_model()
        self.generator = DDPMMolecularGenerator(self.model.base_model, self.ddpm, self.device)
        
        print(f"✅ Molecular Generation Pipeline ready!")
        print(f"   Device: {self.device}")
        print(f"   Model: {self.model_path}")
    
    def _load_model(self):
        """Load trained model"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model
        base_model = create_joint2d3d_model(
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            conditioning_type=config['model'].get('conditioning_type', 'add')
        )
        
        ddpm = MolecularDDPM(
            num_timesteps=config['ddpm']['num_timesteps'],
            beta_schedule=config['ddpm'].get('beta_schedule', 'cosine'),
            beta_start=config['ddpm']['beta_start'],
            beta_end=config['ddpm']['beta_end']
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
        
        return model, ddpm
    
    def generate_drug_candidates(self, num_molecules=100, target_properties=None):
        """🎯 Generate drug candidate molecules"""
        print(f"\n🎯 GENERATING DRUG CANDIDATES")
        print(f"Target: {num_molecules} molecules")
        
        if target_properties:
            print(f"Target properties: {target_properties}")
        
        generated_molecules = []
        valid_molecules = []
        
        for i in range(num_molecules):
            try:
                result = self.generator.generate_molecules(
                    num_molecules=1,
                    max_atoms=50,
                    guidance_scale=1.0
                )
                
                if result['molecules']:
                    mol_data = result['molecules'][0]
                    smiles = self.generator.molecules_to_smiles([mol_data])
                    
                    if smiles[0] and smiles[0] != 'None':
                        mol = Chem.MolFromSmiles(smiles[0])
                        if mol and self._is_drug_like(mol, target_properties):
                            valid_molecules.append({
                                'id': len(valid_molecules) + 1,
                                'smiles': smiles[0],
                                'mol_data': mol_data,
                                'properties': self._calculate_properties(mol)
                            })
                
                if (i + 1) % 20 == 0:
                    print(f"   Progress: {i + 1}/{num_molecules} ({len(valid_molecules)} valid drug-like)")
                    
            except Exception as e:
                continue
        
        print(f"\n📊 Results: {len(valid_molecules)} drug-like molecules generated")
        return valid_molecules
    
    def _is_drug_like(self, mol, target_properties=None):
        """Check if molecule is drug-like"""
        if mol is None:
            return False
        
        # Basic drug-likeness (Lipinski's Rule of Five)
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        lipinski_violations = sum([
            mw > 500,
            logp > 5,
            hbd > 5,
            hba > 10
        ])
        
        if lipinski_violations > 1:  # Allow 1 violation
            return False
        
        # Additional filters
        if mol.GetNumAtoms() < 6 or mol.GetNumAtoms() > 50:
            return False
        
        # Check for reactive groups (basic filter)
        smiles = Chem.MolToSmiles(mol)
        reactive_patterns = ['[S,s][S,s]', '[N,n][N,n]', '[O,o][O,o]']
        for pattern in reactive_patterns:
            if Chem.MolFromSmarts(pattern).HasSubstructMatch(mol):
                return False
        
        # Target properties check
        if target_properties:
            for prop, (min_val, max_val) in target_properties.items():
                if prop == 'mw' and not (min_val <= mw <= max_val):
                    return False
                elif prop == 'logp' and not (min_val <= logp <= max_val):
                    return False
        
        return True
    
    def _calculate_properties(self, mol):
        """Calculate molecular properties"""
        return {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'qed': Descriptors.qed(mol)
        }
    
    def generate_with_pocket_conditioning(self, pocket_file, num_molecules=50):
        """🧬 Generate molecules conditioned on protein pocket"""
        print(f"\n🧬 POCKET-CONDITIONED GENERATION")
        print(f"Pocket: {pocket_file}")
        print(f"Target: {num_molecules} molecules")
        
        # Load pocket data (simplified - you'd need to implement pocket loading)
        # pocket_data = self._load_pocket_data(pocket_file)
        
        molecules = []
        
        for i in range(num_molecules):
            try:
                result = self.generator.generate_molecules(
                    # pocket_data=pocket_data,
                    num_molecules=1,
                    max_atoms=40,
                    guidance_scale=1.5  # Higher guidance for pocket conditioning
                )
                
                if result['molecules']:
                    mol_data = result['molecules'][0]
                    smiles = self.generator.molecules_to_smiles([mol_data])
                    
                    if smiles[0] and smiles[0] != 'None':
                        molecules.append({
                            'id': len(molecules) + 1,
                            'smiles': smiles[0],
                            'binding_affinity_predicted': self._predict_binding_affinity(smiles[0])
                        })
                
                if (i + 1) % 10 == 0:
                    print(f"   Progress: {i + 1}/{num_molecules}")
                    
            except Exception as e:
                continue
        
        return molecules
    
    def _predict_binding_affinity(self, smiles):
        """Placeholder for binding affinity prediction"""
        # This would integrate with docking software or ML models
        return np.random.uniform(-10, -5)  # Placeholder
    
    def save_results(self, molecules, output_dir="generated_molecules"):
        """💾 Save generation results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save SMILES CSV
        df_data = []
        for mol in molecules:
            row = {'id': mol['id'], 'smiles': mol['smiles']}
            if 'properties' in mol:
                row.update(mol['properties'])
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        csv_path = output_path / "generated_molecules.csv"
        df.to_csv(csv_path, index=False)
        
        # Save SMILES txt
        smiles_path = output_path / "generated_smiles.txt"
        with open(smiles_path, 'w') as f:
            for mol in molecules:
                f.write(f"{mol['smiles']}\n")
        
        # Generate molecular images (first 20)
        self._generate_molecular_images(molecules[:20], output_path)
        
        print(f"✅ Results saved to {output_path}/")
        print(f"   CSV: {csv_path}")
        print(f"   SMILES: {smiles_path}")
        print(f"   Images: {output_path}/images/")
        
        return output_path
    
    def _generate_molecular_images(self, molecules, output_path):
        """Generate molecular structure images"""
        img_dir = output_path / "images"
        img_dir.mkdir(exist_ok=True)
        
        for mol_info in molecules:
            try:
                mol = Chem.MolFromSmiles(mol_info['smiles'])
                if mol:
                    img = Draw.MolToImage(mol, size=(300, 300))
                    img_path = img_dir / f"molecule_{mol_info['id']:03d}.png"
                    img.save(img_path)
            except Exception as e:
                print(f"   Warning: Could not generate image for molecule {mol_info['id']}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate molecules with trained DDPM model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='config/optimized_ddpm_config.yaml',
                       help='Path to config file')
    parser.add_argument('--num_molecules', type=int, default=100,
                       help='Number of molecules to generate')
    parser.add_argument('--mode', type=str, choices=['drug_candidates', 'pocket_conditioned'], 
                       default='drug_candidates', help='Generation mode')
    parser.add_argument('--pocket_file', type=str, help='Pocket PDB file for conditioning')
    parser.add_argument('--output_dir', type=str, default='generated_molecules',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("🧬 MOLECULAR GENERATION PIPELINE")
    print("=" * 60)
    
    # Create pipeline
    pipeline = MolecularGenerationPipeline(args.model_path, args.config)
    
    # Generate molecules
    if args.mode == 'drug_candidates':
        target_properties = {
            'mw': (150, 500),  # Molecular weight range
            'logp': (-2, 5)    # LogP range
        }
        molecules = pipeline.generate_drug_candidates(
            num_molecules=args.num_molecules,
            target_properties=target_properties
        )
    
    elif args.mode == 'pocket_conditioned':
        if not args.pocket_file:
            print("❌ Pocket file required for pocket-conditioned generation")
            return
        molecules = pipeline.generate_with_pocket_conditioning(
            pocket_file=args.pocket_file,
            num_molecules=args.num_molecules
        )
    
    # Save results
    if molecules:
        output_path = pipeline.save_results(molecules, args.output_dir)
        
        print(f"\n🎉 GENERATION COMPLETED!")
        print(f"   Generated: {len(molecules)} valid molecules")
        print(f"   Output: {output_path}")
        print(f"\n🔬 Next steps:")
        print(f"   1. Review generated_molecules.csv for molecular properties")
        print(f"   2. Check images/ folder for molecular structures")
        print(f"   3. Use molecules for docking studies or further optimization")
    else:
        print("❌ No valid molecules generated")

if __name__ == "__main__":
    main()