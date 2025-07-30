# src/training/chemical_ddpm_trainer.py - DDPM Trainer with Chemical Constraints
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
import numpy as np
from pathlib import Path
import sys

# Add paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.ddpm_trainer import DDPMMolecularTrainer

class ValenceLoss(nn.Module):
    """Loss function to enforce valence constraints"""
    
    def __init__(self):
        super().__init__()
        # Standard valence rules
        self.valence_rules = {
            6: 4,   # Carbon
            7: 3,   # Nitrogen (can be 4 with charge)
            8: 2,   # Oxygen (can be 3 with charge)
            16: 6,  # Sulfur
            9: 1,   # Fluorine
            17: 1,  # Chlorine
            35: 1,  # Bromine
            53: 1   # Iodine
        }
        
    def forward(self, atom_logits: torch.Tensor, edge_index: torch.Tensor, 
                batch: torch.Tensor) -> torch.Tensor:
        """
        Compute valence constraint loss
        
        Args:
            atom_logits: [N, num_atom_types] predicted atom type logits
            edge_index: [2, E] edge connectivity
            batch: [N] batch indices
        """
        
        if atom_logits is None or edge_index.size(1) == 0:
            return torch.tensor(0.0, device=atom_logits.device if atom_logits is not None else 'cpu')
        
        # Get predicted atom types (assume offset by 6 for C=6, N=7, etc.)
        atom_probs = F.softmax(atom_logits, dim=-1)
        atom_types = torch.argmax(atom_probs, dim=-1) + 6
        
        # Count bonds per atom
        num_atoms = atom_logits.size(0)
        bond_counts = torch.zeros(num_atoms, device=atom_logits.device)
        
        if edge_index.size(1) > 0:
            # Count incoming edges (bonds) for each atom
            bond_counts.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=atom_logits.device))
        
        # Calculate valence violations
        total_violation = 0.0
        valid_atoms = 0
        
        for i in range(num_atoms):
            atom_type = atom_types[i].item()
            bond_count = bond_counts[i].item()
            max_valence = self.valence_rules.get(atom_type, 4)
            
            if bond_count > max_valence:
                # Quadratic penalty for valence violations
                violation = (bond_count - max_valence) ** 2
                total_violation += violation
            
            valid_atoms += 1
        
        if valid_atoms == 0:
            return torch.tensor(0.0, device=atom_logits.device)
        
        return torch.tensor(total_violation / valid_atoms, device=atom_logits.device)

class ChemicalPropertyLoss(nn.Module):
    """Loss to encourage drug-like molecular properties"""
    
    def __init__(self):
        super().__init__()
        # Drug-like property ranges
        self.target_mw_range = (150, 500)   # Molecular weight
        self.target_atom_range = (10, 30)   # Number of atoms
        self.max_bonds_per_atom = 4
        
        # Atomic weights for common elements
        self.atomic_weights = {
            6: 12.01,   # Carbon
            7: 14.01,   # Nitrogen
            8: 15.99,   # Oxygen
            16: 32.06,  # Sulfur
            9: 18.99,   # Fluorine
            17: 35.45,  # Chlorine
        }
        
    def forward(self, atom_logits: torch.Tensor, positions: torch.Tensor,
                edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute chemical property loss
        
        Args:
            atom_logits: [N, num_atom_types] predicted atom types
            positions: [N, 3] atomic positions
            edge_index: [2, E] edge connectivity
            batch: [N] batch indices
        """
        
        if atom_logits is None:
            return torch.tensor(0.0)
        
        device = atom_logits.device
        total_loss = 0.0
        num_molecules = 0
        
        # Process each molecule in batch
        unique_batches = torch.unique(batch)
        
        for batch_idx in unique_batches:
            mask = batch == batch_idx
            mol_atom_logits = atom_logits[mask]
            mol_positions = positions[mask] if positions is not None else None
            
            # Estimate molecular weight
            atom_probs = F.softmax(mol_atom_logits, dim=-1)
            atom_types = torch.argmax(atom_probs, dim=-1) + 6  # Offset for atomic numbers
            
            mol_weight = 0.0
            for atom_type in atom_types:
                atomic_num = atom_type.item()
                weight = self.atomic_weights.get(atomic_num, 12.0)  # Default to carbon
                mol_weight += weight
            
            # Molecular weight penalty
            mw_loss = 0.0
            if mol_weight < self.target_mw_range[0]:
                mw_loss = (self.target_mw_range[0] - mol_weight) ** 2 / 10000.0
            elif mol_weight > self.target_mw_range[1]:
                mw_loss = (mol_weight - self.target_mw_range[1]) ** 2 / 10000.0
            
            # Number of atoms penalty
            num_atoms = len(mol_atom_logits)
            atom_count_loss = 0.0
            if num_atoms < self.target_atom_range[0]:
                atom_count_loss = (self.target_atom_range[0] - num_atoms) ** 2 / 100.0
            elif num_atoms > self.target_atom_range[1]:
                atom_count_loss = (num_atoms - self.target_atom_range[1]) ** 2 / 100.0
            
            molecule_loss = mw_loss + atom_count_loss
            total_loss += molecule_loss
            num_molecules += 1
        
        if num_molecules == 0:
            return torch.tensor(0.0, device=device)
        
        return torch.tensor(total_loss / num_molecules, device=device)

class BondRealismLoss(nn.Module):
    """Loss to encourage realistic bond lengths and angles"""
    
    def __init__(self):
        super().__init__()
        # Typical bond lengths in Angstroms
        self.ideal_bond_lengths = {
            (6, 6): 1.54,   # C-C
            (6, 7): 1.47,   # C-N
            (6, 8): 1.43,   # C-O
            (6, 16): 1.82,  # C-S
            (7, 7): 1.45,   # N-N
            (7, 8): 1.40,   # N-O
            (8, 8): 1.48,   # O-O
        }
        self.default_bond_length = 1.5
        self.bond_length_tolerance = 0.3  # Angstroms
        
    def forward(self, positions: torch.Tensor, atom_logits: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute bond realism loss
        
        Args:
            positions: [N, 3] atomic positions
            atom_logits: [N, num_atom_types] predicted atom types
            edge_index: [2, E] edge connectivity
        """
        
        if positions is None or atom_logits is None or edge_index.size(1) == 0:
            return torch.tensor(0.0)
        
        device = positions.device
        atom_types = torch.argmax(atom_logits, dim=-1) + 6  # Offset for atomic numbers
        
        total_bond_loss = 0.0
        num_bonds = 0
        
        # Process bonds (assume undirected, so take every other edge)
        for i in range(0, edge_index.size(1), 2):
            if i >= edge_index.size(1):
                break
                
            atom1_idx = edge_index[0, i]
            atom2_idx = edge_index[1, i]
            
            # Get atom types
            atom1_type = atom_types[atom1_idx].item()
            atom2_type = atom_types[atom2_idx].item()
            
            # Get ideal bond length
            bond_key = tuple(sorted([atom1_type, atom2_type]))
            ideal_length = self.ideal_bond_lengths.get(bond_key, self.default_bond_length)
            
            # Calculate actual distance
            pos1 = positions[atom1_idx]
            pos2 = positions[atom2_idx]
            actual_distance = torch.norm(pos1 - pos2)
            
            # Compute loss (penalize deviations from ideal length)
            length_diff = torch.abs(actual_distance - ideal_length)
            
            # Use smooth L1 loss to avoid exploding gradients
            if length_diff > self.bond_length_tolerance:
                bond_loss = length_diff - self.bond_length_tolerance / 2
            else:
                bond_loss = (length_diff ** 2) / (2 * self.bond_length_tolerance)
            
            total_bond_loss += bond_loss
            num_bonds += 1
        
        if num_bonds == 0:
            return torch.tensor(0.0, device=device)
        
        return total_bond_loss / num_bonds

class ChemicalDDPMTrainer(DDPMTrainer):
    """DDPM Trainer with Chemical Constraints"""
    
    def __init__(self, base_model, ddpm, optimizer, scheduler=None, device='cuda', callbacks=None,
                 chemical_loss_weights=None):
        super().__init__(base_model, ddpm, optimizer, scheduler, device, callbacks)
        
        # Chemical loss functions
        self.valence_loss_fn = ValenceLoss()
        self.property_loss_fn = ChemicalPropertyLoss()
        self.bond_loss_fn = BondRealismLoss()
        
        # Loss weights (can be adjusted during training)
        self.chemical_weights = chemical_loss_weights or {
            'valence': 0.1,
            'property': 0.05,
            'bond_realism': 0.05,
            'warmup_epochs': 10  # Epochs to gradually increase chemical loss weight
        }
        
        self.epoch_counter = 0
        
        print(f"✅ Chemical DDPM Trainer initialized")
        print(f"   Chemical loss weights: {self.chemical_weights}")
    
    def compute_chemical_losses(self, model_output: Dict[str, torch.Tensor], 
                              batch) -> Dict[str, torch.Tensor]:
        """Compute all chemical constraint losses"""
        
        chemical_losses = {}
        total_chemical_loss = 0.0
        
        # Extract model outputs
        atom_logits = model_output.get('atom_logits')
        pos_pred = model_output.get('pos_pred', batch.pos)
        edge_index = batch.edge_index
        batch_idx = batch.batch
        
        try:
            # Valence constraint loss
            if atom_logits is not None:
                valence_loss = self.valence_loss_fn(atom_logits, edge_index, batch_idx)
                chemical_losses['valence_loss'] = valence_loss
                total_chemical_loss += self.chemical_weights['valence'] * valence_loss
            
            # Chemical property loss
            if atom_logits is not None:
                property_loss = self.property_loss_fn(atom_logits, pos_pred, edge_index, batch_idx)
                chemical_losses['property_loss'] = property_loss
                total_chemical_loss += self.chemical_weights['property'] * property_loss
            
            # Bond realism loss
            if pos_pred is not None and atom_logits is not None:
                bond_loss = self.bond_loss_fn(pos_pred, atom_logits, edge_index)
                chemical_losses['bond_loss'] = bond_loss
                total_chemical_loss += self.chemical_weights['bond_realism'] * bond_loss
            
        except Exception as e:
            print(f"Warning: Chemical loss computation failed: {e}")
            # Return zero losses to continue training
            chemical_losses = {
                'valence_loss': torch.tensor(0.0, device=self.device),
                'property_loss': torch.tensor(0.0, device=self.device),
                'bond_loss': torch.tensor(0.0, device=self.device)
            }
            total_chemical_loss = torch.tensor(0.0, device=self.device)
        
        chemical_losses['total_chemical_loss'] = total_chemical_loss
        return chemical_losses
    
    def get_chemical_weight_multiplier(self) -> float:
        """Get chemical loss weight multiplier based on training progress"""
        
        warmup_epochs = self.chemical_weights.get('warmup_epochs', 10)
        
        if self.epoch_counter < warmup_epochs:
            # Gradually increase chemical loss weight during warmup
            return self.epoch_counter / warmup_epochs
        else:
            return 1.0
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train one epoch with chemical constraints"""
        
        self.model.base_model.train()
        epoch_losses = {
            'total_loss': 0.0, 
            'ddpm_loss': 0.0,
            'chemical_loss': 0.0,
            'valence_loss': 0.0,
            'property_loss': 0.0,
            'bond_loss': 0.0
        }
        num_batches = 0
        
        # Get chemical loss multiplier
        chem_multiplier = self.get_chemical_weight_multiplier()
        
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue
            
            try:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                
                # Prepare DDPM inputs
                ddpm_kwargs = self._prepare_safe_kwargs(batch)
                target_pos = batch.pos.to(self.device)
                
                # DDPM loss
                ddpm_loss, ddpm_loss_dict = self.ddpm.compute_loss(
                    model=self.model,
                    x0=target_pos,
                    **ddmp_kwargs
                )
                
                # Get model outputs for chemical losses
                with torch.no_grad():
                    model_output = self.model.base_model(
                        x=batch.x,
                        pos=batch.pos,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        batch=batch.batch,
                        **self._get_pocket_kwargs(batch)
                    )
                
                # Chemical constraint losses
                chemical_losses = self.compute_chemical_losses(model_output, batch)
                
                # Combined loss with dynamic weighting
                chemical_component = chemical_losses['total_chemical_loss'] * chem_multiplier
                total_loss = ddpm_loss + chemical_component
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                # Update metrics
                epoch_losses['total_loss'] += total_loss.item()
                epoch_losses['ddpm_loss'] += ddpm_loss.item()
                epoch_losses['chemical_loss'] += chemical_component.item()
                epoch_losses['valence_loss'] += chemical_losses['valence_loss'].item()
                epoch_losses['property_loss'] += chemical_losses['property_loss'].item()
                epoch_losses['bond_loss'] += chemical_losses['bond_loss'].item()
                
                num_batches += 1
                
                # Progress logging
                if batch_idx % 50 == 0:
                    print(f"   Batch {batch_idx}: Total={total_loss.item():.4f}, "
                          f"DDPM={ddpm_loss.item():.4f}, "
                          f"Chem={chemical_component.item():.4f} "
                          f"(mult={chem_multiplier:.2f})")
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Average losses
        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def _get_pocket_kwargs(self, batch):
        """Extract pocket data from batch"""
        pocket_kwargs = {}
        
        pocket_attrs = ['pocket_x', 'pocket_pos', 'pocket_edge_index', 'pocket_batch']
        for attr in pocket_attrs:
            if hasattr(batch, attr):
                value = getattr(batch, attr)
                if value is not None:
                    pocket_kwargs[attr] = value.to(self.device)
        
        return pocket_kwargs
    
    def on_epoch_end(self):
        """Called at the end of each epoch"""
        self.epoch_counter += 1
    
    def train(self, train_loader, val_loader, num_epochs: int, save_path: str = None):
        """Full training loop with chemical constraints"""
        
        print(f"🚀 Starting Chemical DDPM training for {num_epochs} epochs...")
        print(f"   Device: {self.device}")
        print(f"   Chemical constraints: Valence + Properties + Bond Realism")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            print(f"\n📈 Chemical DDPM Epoch {epoch + 1}/{num_epochs}")
            
            # Train with chemical constraints
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Combine metrics
            epoch_metrics = {
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }
            
            # Print detailed metrics
            print(f"📊 Train - Total: {train_metrics['total_loss']:.4f}, "
                  f"DDPM: {train_metrics['ddpm_loss']:.4f}, "
                  f"Chemical: {train_metrics['chemical_loss']:.4f}")
            print(f"     ├─ Valence: {train_metrics['valence_loss']:.4f}")
            print(f"     ├─ Property: {train_metrics['property_loss']:.4f}")
            print(f"     └─ Bond: {train_metrics['bond_loss']:.4f}")
            print(f"📊 Val - Total: {val_metrics['total_loss']:.4f}")
            
            # Save best model
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                if save_path:
                    self.save_checkpoint(save_path, epoch, val_metrics['total_loss'])
                    print(f"💾 Best chemical model saved!")
            
            # Callbacks
            for callback in self.callbacks: