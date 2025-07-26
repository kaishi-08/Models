# src/utils/geometry_utils.py
import torch
import numpy as np
from typing import Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

class GeometryUtils:
    """Utility functions for 3D geometry operations"""
    
    @staticmethod
    def compute_distances(pos1: torch.Tensor, pos2: torch.Tensor = None) -> torch.Tensor:
        """Compute pairwise distances between positions"""
        if pos2 is None:
            pos2 = pos1
        
        # Expand dimensions for broadcasting
        pos1_expanded = pos1.unsqueeze(1)  # [N, 1, 3]
        pos2_expanded = pos2.unsqueeze(0)  # [1, M, 3]
        
        # Compute distances
        distances = torch.norm(pos1_expanded - pos2_expanded, dim=-1)
        return distances
    
    @staticmethod
    def get_neighbors(positions: torch.Tensor, radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get neighbors within radius"""
        distances = GeometryUtils.compute_distances(positions)
        
        # Create edge index for neighbors within radius
        edge_index = []
        edge_distances = []
        
        for i in range(positions.size(0)):
            for j in range(positions.size(0)):
                if i != j and distances[i, j] <= radius:
                    edge_index.append([i, j])
                    edge_distances.append(distances[i, j])
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_distances = torch.tensor(edge_distances, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_distances = torch.zeros((0,), dtype=torch.float)
        
        return edge_index, edge_distances
    
    @staticmethod
    def compute_angles(pos_center: torch.Tensor, pos_neighbor1: torch.Tensor,
                      pos_neighbor2: torch.Tensor) -> torch.Tensor:
        """Compute angles between three points"""
        vec1 = pos_neighbor1 - pos_center
        vec2 = pos_neighbor2 - pos_center
        
        # Normalize vectors
        vec1_norm = torch.norm(vec1, dim=-1, keepdim=True)
        vec2_norm = torch.norm(vec2, dim=-1, keepdim=True)
        
        vec1_normalized = vec1 / (vec1_norm + 1e-8)
        vec2_normalized = vec2 / (vec2_norm + 1e-8)
        
        # Compute angles
        cos_angles = torch.sum(vec1_normalized * vec2_normalized, dim=-1)
        cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
        angles = torch.acos(cos_angles)
        
        return angles
    
    @staticmethod
    def compute_dihedrals(pos1: torch.Tensor, pos2: torch.Tensor,
                         pos3: torch.Tensor, pos4: torch.Tensor) -> torch.Tensor:
        """Compute dihedral angles between four points"""
        # Vectors for dihedral calculation
        b1 = pos2 - pos1
        b2 = pos3 - pos2
        b3 = pos4 - pos3
        
        # Normal vectors to planes
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)
        
        # Normalize normal vectors
        n1_norm = torch.norm(n1, dim=-1, keepdim=True)
        n2_norm = torch.norm(n2, dim=-1, keepdim=True)
        
        n1_normalized = n1 / (n1_norm + 1e-8)
        n2_normalized = n2 / (n2_norm + 1e-8)
        
        # Compute dihedral angle
        cos_dihedral = torch.sum(n1_normalized * n2_normalized, dim=-1)
        cos_dihedral = torch.clamp(cos_dihedral, -1.0, 1.0)
        
        # Sign of dihedral
        sign = torch.sign(torch.sum(torch.cross(n1_normalized, n2_normalized, dim=-1) * 
                                   (b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)), dim=-1))
        
        dihedrals = sign * torch.acos(cos_dihedral)
        
        return dihedrals
    
    @staticmethod
    def apply_random_rotation(positions: torch.Tensor) -> torch.Tensor:
        """Apply random 3D rotation to positions"""
        # Generate random rotation matrix
        rotation = Rotation.random()
        rotation_matrix = torch.tensor(rotation.as_matrix(), dtype=torch.float)
        
        # Apply rotation
        rotated_positions = torch.matmul(positions, rotation_matrix.t())
        
        return rotated_positions
    
    @staticmethod
    def center_positions(positions: torch.Tensor) -> torch.Tensor:
        """Center positions around origin"""
        centroid = torch.mean(positions, dim=0, keepdim=True)
        centered_positions = positions - centroid
        return centered_positions
    
    @staticmethod
    def align_structures(pos1: torch.Tensor, pos2: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Align two structures using Kabsch algorithm"""
        # Center both structures
        pos1_centered = GeometryUtils.center_positions(pos1)
        pos2_centered = GeometryUtils.center_positions(pos2)
        
        # Compute covariance matrix
        H = torch.matmul(pos1_centered.t(), pos2_centered)
        
        # SVD decomposition
        U, S, V = torch.svd(H)
        
        # Compute rotation matrix
        R = torch.matmul(V, U.t())
        
        # Ensure proper rotation (not reflection)
        if torch.det(R) < 0:
            V[:, -1] *= -1
            R = torch.matmul(V, U.t())
        
        # Apply rotation to first structure
        pos1_aligned = torch.matmul(pos1_centered, R)
        
        # Compute RMSD
        rmsd = torch.sqrt(torch.mean(torch.sum((pos1_aligned - pos2_centered) ** 2, dim=-1)))
        
        return pos1_aligned, rmsd.item()

class ConformationGenerator:
    """Generate molecular conformations"""
    
    @staticmethod
    def generate_conformations(positions: torch.Tensor, edge_index: torch.Tensor,
                             num_conformations: int = 10, noise_scale: float = 0.1) -> torch.Tensor:
        """Generate multiple conformations by adding noise"""
        conformations = []
        
        for _ in range(num_conformations):
            # Add random noise
            noise = torch.randn_like(positions) * noise_scale
            noisy_positions = positions + noise
            
            # Optionally apply constraints based on bonds
            constrained_positions = ConformationGenerator._apply_bond_constraints(
                noisy_positions, edge_index
            )
            
            conformations.append(constrained_positions)
        
        return torch.stack(conformations)
    
    @staticmethod
    def _apply_bond_constraints(positions: torch.Tensor, edge_index: torch.Tensor,
                              target_bond_length: float = 1.5) -> torch.Tensor:
        """Apply simple bond length constraints"""
        if edge_index.size(1) == 0:
            return positions
        
        constrained_positions = positions.clone()
        
        # Simple constraint enforcement (can be improved)
        for _ in range(10):  # Few iterations of constraint enforcement
            row, col = edge_index
            
            # Current bond vectors and lengths
            bond_vectors = constrained_positions[row] - constrained_positions[col]
            bond_lengths = torch.norm(bond_vectors, dim=-1, keepdim=True)
            
            # Normalize to target length
            normalized_vectors = bond_vectors / (bond_lengths + 1e-8)
            target_vectors = normalized_vectors * target_bond_length
            
            # Update positions (simplified approach)
            displacement = (target_vectors - bond_vectors) * 0.1
            constrained_positions[row] += displacement * 0.5
            constrained_positions[col] -= displacement * 0.5
        
        return constrained_positions