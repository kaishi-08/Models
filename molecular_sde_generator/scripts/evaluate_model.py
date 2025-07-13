import torch
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.models.joint_2d_3d_model import Joint2D3DMolecularModel
from src.models.sde_diffusion import VESDE
from src.data.data_loaders import CrossDockDataLoader
from src.utils.molecular_utils import MolecularMetrics
from src.inference.conditional_generator import ConditionalMolecularGenerator

def evaluate_model(model, sde, test_loader, device, num_samples=100):
    """Evaluate model on test set"""
    model.eval()
    
    # Initialize generator
    generator = ConditionalMolecularGenerator(model, sde, device)
    
    results = {
        'reconstruction_loss': [],
        'generated_molecules': [],
        'validity': 0.0,
        'uniqueness': 0.0,
        'drug_likeness': {}
    }
    
    with torch.no_grad():
        # Compute reconstruction loss on test set
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(test_loader, desc="Computing test loss"):
            batch = batch.to(device)
            
            # Sample random times
            t = torch.rand(batch.batch.max().item() + 1, device=device)
            t = t[batch.batch]
            
            # Add noise according to SDE
            mean, std = sde.marginal_prob(batch.pos, t)
            noise = torch.randn_like(batch.pos)
            perturbed_pos = mean + std[:, None] * noise
            
            # Forward pass
            outputs = model(
                x=batch.x,
                pos=perturbed_pos,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                pocket_x=getattr(batch, 'pocket_x', None),
                pocket_pos=getattr(batch, 'pocket_pos', None),
                pocket_edge_index=getattr(batch, 'pocket_edge_index', None),
                pocket_batch=getattr(batch, 'pocket_batch', None)
            )
            
            # Compute score target and loss
            score_target = -noise / std[:, None]
            score_pred = outputs['pos_pred']
            loss = torch.nn.MSELoss()(score_pred, score_target)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_test_loss = total_loss / num_batches
        results['reconstruction_loss'] = avg_test_loss
        
        # Generate molecules for evaluation
        print(f"Generating {num_samples} molecules for evaluation...")
        generated = generator.generate_molecules(
            pocket_data={},  # Unconditional generation
            num_molecules=num_samples,
            max_atoms=50
        )
        
        smiles_list = generator.molecules_to_smiles(generated['molecules'])
        results['generated_molecules'] = smiles_list
        
        # Compute molecular metrics
        results['validity'] = MolecularMetrics.compute_validity(smiles_list)
        results['uniqueness'] = MolecularMetrics.compute_uniqueness(smiles_list)
        results['drug_likeness'] = MolecularMetrics.compute_drug_likeness(smiles_list)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained SDE model')
    parser.add_argument('--config', type=str, default='config/training_config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default='evaluation_results/')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = Joint2D3DMolecularModel(
        atom_types=config['model']['atom_types'],
        bond_types=config['model']['bond_types'],
        hidden_dim=config['model']['hidden_dim'],
        pocket_dim=config['model']['pocket_dim'],
        num_layers=config['model']['num_layers'],
        max_radius=config['model']['max_radius']
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize SDE
    sde = VESDE(
        sigma_min=config['sde']['sigma_min'],
        sigma_max=config['sde']['sigma_max'],
        N=config['sde']['num_steps']
    )
    
    # Create test loader
    test_loader = CrossDockDataLoader.create_test_loader(config)
    
    # Evaluate model
    results = evaluate_model(model, sde, test_loader, device, args.num_samples)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Test Reconstruction Loss: {results['reconstruction_loss']:.4f}")
    print(f"Validity: {results['validity']:.3f}")
    print(f"Uniqueness: {results['uniqueness']:.3f}")
    
    for metric, value in results['drug_likeness'].items():
        print(f"{metric}: {value:.3f}")
    
    # Save results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    import json
    with open(f"{args.output_dir}/evaluation_results.json", 'w') as f:
        # Convert non-serializable items
        serializable_results = {
            'reconstruction_loss': results['reconstruction_loss'],
            'validity': results['validity'],
            'uniqueness': results['uniqueness'],
            'drug_likeness': results['drug_likeness'],
            'num_generated': len(results['generated_molecules'])
        }
        json.dump(serializable_results, f, indent=2)
    
    # Save generated molecules
    with open(f"{args.output_dir}/generated_molecules.txt", 'w') as f:
        for i, smiles in enumerate(results['generated_molecules']):
            if smiles:
                f.write(f"{i}\t{smiles}\n")

if __name__ == "__main__":
    main()