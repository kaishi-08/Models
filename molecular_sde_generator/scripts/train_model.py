# scripts/train_model.py
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import wandb
import yaml
from pathlib import Path

from src.models.joint_2d_3d_model import Joint2D3DMolecularModel
from src.models.sde_diffusion import VESDE
from src.training.sde_trainer import SDEMolecularTrainer
from src.data.molecular_dataset import MolecularDataset

def main():
    # Load configuration
    with open('config/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize wandb
    wandb.init(project="molecular-sde-generation", config=config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_dataset = MolecularDataset(
        data_path=config['data']['train_path'],
        transform=None,
        include_pocket=True
    )
    
    val_dataset = MolecularDataset(
        data_path=config['data']['val_path'],
        transform=None,
        include_pocket=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Initialize model
    model = Joint2D3DMolecularModel(
        atom_types=config['model']['atom_types'],
        bond_types=config['model']['bond_types'],
        hidden_dim=config['model']['hidden_dim'],
        pocket_dim=config['model']['pocket_dim'],
        num_layers=config['model']['num_layers'],
        max_radius=config['model']['max_radius']
    ).to(device)
    
    # Initialize SDE
    sde = VESDE(
        sigma_min=config['sde']['sigma_min'],
        sigma_max=config['sde']['sigma_max'],
        N=config['sde']['num_steps']
    )
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['scheduler_step'],
        gamma=config['training']['scheduler_gamma']
    )
    
    # Initialize trainer
    trainer = SDEMolecularTrainer(
        model=model,
        sde=sde,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_wandb=True
    )
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        save_path=config['training']['save_path']
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()