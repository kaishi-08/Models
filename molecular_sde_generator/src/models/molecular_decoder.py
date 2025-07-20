# src/models/molecular_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from .base_model import MolecularModel

class MolecularDecoder(MolecularModel):
    """Decoder for generating molecular structures from latent representations"""
    
    def __init__(self, latent_dim: int, atom_types: int, bond_types: int,
                 hidden_dim: int = 128, num_layers: int = 3,
                 max_atoms: int = 50):
        super().__init__(atom_types, bond_types, hidden_dim)
        
        self.latent_dim = latent_dim
        self.max_atoms = max_atoms
        self.num_layers = num_layers
        
        # Latent to molecular graph decoder
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Graph generation layers
        self.graph_layers = nn.ModuleList([
            GraphDecoderLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Output heads
        self.atom_type_head = nn.Linear(hidden_dim, atom_types)
        self.atom_position_head = nn.Linear(hidden_dim, 3)
        self.atom_stop_head = nn.Linear(hidden_dim, 1)  # When to stop adding atoms
        
        # Bond prediction
        self.bond_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bond_types)
        )
        
        # Node embedding for sequential generation
        self.node_embedding = nn.Embedding(atom_types, hidden_dim)
        self.pos_embedding = nn.Linear(3, hidden_dim)
        
    def forward(self, latent: torch.Tensor, max_atoms: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Generate molecular graph from latent representation
        
        Args:
            latent: [batch_size, latent_dim] latent vectors
            max_atoms: Maximum number of atoms to generate
        """
        batch_size = latent.size(0)
        device = latent.device
        max_atoms = max_atoms or self.max_atoms
        
        # Project latent to hidden space
        h_latent = self.latent_projection(latent)  # [batch_size, hidden_dim]
        
        # Initialize molecular graphs
        molecules = []
        
        for b in range(batch_size):
            molecule = self._generate_single_molecule(h_latent[b], max_atoms, device)
            molecules.append(molecule)
        
        # Batch molecules
        return self._batch_molecules(molecules)
    
    def _generate_single_molecule(self, h_latent: torch.Tensor, max_atoms: int,
                                device: torch.device) -> Dict[str, torch.Tensor]:
        """Generate a single molecule autoregressively"""
        
        # Initialize with first atom
        atom_types = []
        positions = []
        edges = []
        node_features = [h_latent.unsqueeze(0)]  # Start with latent representation
        
        for step in range(max_atoms):
            # Current graph state
            if len(node_features) > 1:
                h_graph = torch.cat(node_features, dim=0)  # [num_atoms, hidden_dim]
                
                # Apply graph layers
                for layer in self.graph_layers:
                    h_graph = layer(h_graph, self._get_current_edges(len(node_features)))
            else:
                h_graph = node_features[0]
            
            # Get features for the current (last) atom
            h_current = h_graph[-1] if h_graph.dim() > 1 else h_graph
            
            # Predict atom properties
            atom_logits = self.atom_type_head(h_current)
            position = self.atom_position_head(h_current)
            stop_logits = self.atom_stop_head(h_current)
            
            # Check if we should stop
            if torch.sigmoid(stop_logits) > 0.5 and step > 0:
                break
            
            # Sample atom type
            atom_type = torch.multinomial(F.softmax(atom_logits, dim=-1), 1)
            
            # Store atom information
            atom_types.append(atom_type.item())
            positions.append(position.detach())
            
            # Create node embedding for next step
            atom_emb = self.node_embedding(atom_type)
            pos_emb = self.pos_embedding(position)
            next_node_feature = atom_emb + pos_emb + h_latent
            node_features.append(next_node_feature.unsqueeze(0))
            
            # Predict bonds to previous atoms
            if step > 0:
                for prev_idx in range(len(node_features) - 1):
                    edge_feature = torch.cat([h_current, node_features[prev_idx].squeeze()], dim=-1)
                    bond_logits = self.bond_predictor(edge_feature)
                    bond_type = torch.multinomial(F.softmax(bond_logits, dim=-1), 1)
                    
                    # Add edge if bond type > 0 (0 = no bond)
                    if bond_type.item() > 0:
                        edges.append([prev_idx, step, bond_type.item()])
                        edges.append([step, prev_idx, bond_type.item()])
        
        # Convert to tensors
        atom_types = torch.tensor(atom_types, dtype=torch.long, device=device)
        positions = torch.stack(positions) if positions else torch.zeros((0, 3), device=device)
        
        # Create edge tensors
        if edges:
            edge_list = [[e[0], e[1]] for e in edges]
            edge_attr = [e[2] for e in edges]
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t()
            edge_attr = torch.tensor(edge_attr, dtype=torch.long, device=device)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0,), dtype=torch.long, device=device)
        
        return {
            'atom_types': atom_types,
            'positions': positions,
            'edge_index': edge_index,
            'edge_attr': edge_attr
        }
    
    def _get_current_edges(self, num_atoms: int) -> torch.Tensor:
        """Get fully connected edges for current atoms"""
        edges = []
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    edges.append([i, j])
        
        if edges:
            return torch.tensor(edges, dtype=torch.long).t()
        else:
            return torch.zeros((2, 0), dtype=torch.long)
    
    def _batch_molecules(self, molecules: list) -> Dict[str, torch.Tensor]:
        """Batch multiple molecules together"""
        from torch_geometric.data import Data, Batch
        
        data_list = []
        for mol in molecules:
            data = Data(
                x=mol['atom_types'].unsqueeze(-1).float(),
                pos=mol['positions'],
                edge_index=mol['edge_index'],
                edge_attr=mol['edge_attr'].unsqueeze(-1).float()
            )
            data_list.append(data)
        
        batch = Batch.from_data_list(data_list)
        
        return {
            'x': batch.x,
            'pos': batch.pos,
            'edge_index': batch.edge_index,
            'edge_attr': batch.edge_attr,
            'batch': batch.batch
        }

class GraphDecoderLayer(nn.Module):
    """Graph decoder layer for molecular generation"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.message_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(input_dim + output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_atoms, input_dim] node features
            edge_index: [2, num_edges] edge connectivity
        """
        if edge_index.size(1) == 0:
            return x
        
        row, col = edge_index
        
        # Message passing
        messages = self.message_mlp(torch.cat([x[row], x[col]], dim=-1))
        
        # Aggregate messages
        out = torch.zeros_like(x)
        out = out.index_add(0, col, messages)
        
        # Update node features
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        out = self.norm(out)
        
        return out + x  # Residual connection