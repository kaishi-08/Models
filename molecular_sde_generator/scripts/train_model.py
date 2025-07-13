import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import wandb
import yaml
from pathlib import Path

from src.models.joint_2d_3d_model import Joint2D3DMolecularModel
from src.models.sde_diffusion import VESDE
from src.training.sde_trainer import SDEMolecularTrainer
from src.data.molecular_dataset import CrossDockMolecularDataset
from src.data.data_loaders import CrossDockDataLoader

def main():
    # Load configuration
    with open('config/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize wandb
    wandb.init(project=config['logging']['project_name'], config=config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders using the factory
    train_loader = CrossDockDataLoader.create_train_loader(config)
    val_loader = CrossDockDataLoader.create_val_loader(config)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Initialize model
    model = Joint2D3DMolecularModel(
        atom_types=config['model']['atom_types'],
        bond_types=config['model']['bond_types'],
        hidden_dim=config['model']['hidden_dim'],
        pocket_dim=config['model']['pocket_dim'],
        num_layers=config['model']['num_layers'],
        max_radius=config['model']['max_radius']
    ).to(device)
    
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Initialize SDE
    sde = VESDE(
        sigma_min=config['sde']['sigma_min'],
        sigma_max=config['sde']['sigma_max'],
        N=config['sde']['num_steps']
    )
    
    # Initialize optimizer
    if config['optimizer']['type'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['optimizer']['lr'],
            betas=config['optimizer']['betas'],
            weight_decay=config['optimizer']['weight_decay']
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
    
    # Initialize scheduler
    if config['scheduler']['type'] == 'cosine_annealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['scheduler']['T_max'],
            eta_min=config['scheduler']['eta_min']
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.5
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
    print("Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        save_path=config['logging']['save_path'] + '/best_model.pth'
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()