import torch
import yaml
import argparse
from pathlib import Path
import numpy as np
from rdkit import Chem

from src.models.joint_2d_3d_model import Joint2D3DMolecularModel
from src.models.sde_diffusion import VESDE
from src.inference.conditional_generator import ConditionalMolecularGenerator
from src.data.molecular_dataset import CrossDockMolecularDataset

def load_model(checkpoint_path: str, config: dict, device: str):
    """Load trained model from checkpoint"""
    model = Joint2D3DMolecularModel(
        atom_types=config['model']['atom_types'],
        bond_types=config['model']['bond_types'],
        hidden_dim=config['model']['hidden_dim'],
        pocket_dim=config['model']['pocket_dim'],
        num_layers=config['model']['num_layers'],
        max_radius=config['model']['max_radius']
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Generate molecules using trained SDE model')
    parser.add_argument('--config', type=str, default='config/training_config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_molecules', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='generated_molecules/')
    parser.add_argument('--pocket_file', type=str, help='Optional pocket file for conditioning')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, config, device)
    
    # Initialize SDE
    sde = VESDE(
        sigma_min=config['sde']['sigma_min'],
        sigma_max=config['sde']['sigma_max'],
        N=config['sde']['num_steps']
    )
    
    # Initialize generator
    generator = ConditionalMolecularGenerator(
        model=model,
        sde=sde,
        device=device
    )
    
    # Prepare pocket data (if provided)
    pocket_data = {}
    if args.pocket_file:
        # TODO: Implement pocket loading from file
        print(f"Loading pocket from {args.pocket_file}")
        pass
    
    # Generate molecules
    print(f"Generating {args.num_molecules} molecules...")
    generated = generator.generate_molecules(
        pocket_data=pocket_data,
        num_molecules=args.num_molecules,
        max_atoms=config['generation']['max_atoms'],
        guidance_scale=config['generation']['guidance_scale']
    )
    
    # Convert to SMILES
    smiles_list = generator.molecules_to_smiles(generated['molecules'])
    
    # Save results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(f"{args.output_dir}/generated_smiles.txt", 'w') as f:
        for i, smiles in enumerate(smiles_list):
            if smiles:
                f.write(f"{i}\t{smiles}\n")
    
    # Compute basic statistics
    valid_count = sum(1 for s in smiles_list if s is not None)
    print(f"Generated {len(smiles_list)} molecules")
    print(f"Valid molecules: {valid_count} ({valid_count/len(smiles_list)*100:.1f}%)")
    
    if valid_count > 0:
        unique_smiles = set(s for s in smiles_list if s is not None)
        print(f"Unique molecules: {len(unique_smiles)} ({len(unique_smiles)/valid_count*100:.1f}%)")

if __name__ == "__main__":
    main()