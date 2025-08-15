# src/utils/molecular_utils.py
import torch
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
from torch_geometric.utils import to_networkx
from torch import nn

class MolecularFeaturizer:
    """Utility class for molecular featurization"""
    
    @staticmethod
    def atom_to_features(atom: Chem.Atom) -> List[float]:
        """Convert RDKit atom to feature vector"""
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(),
            int(atom.IsInRing()),
            atom.GetMass()
        ]
        return features
    
    @staticmethod
    def bond_to_features(bond: Chem.Bond) -> List[float]:
        """Convert RDKit bond to feature vector"""
        features = [
            int(bond.GetBondType()),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            int(bond.GetStereo())
        ]
        return features
    
    @staticmethod
    def mol_to_graph(mol: Chem.Mol, add_hydrogens: bool = False) -> Dict[str, np.ndarray]:
        """Convert RDKit molecule to graph representation"""
        if add_hydrogens:
            mol = Chem.AddHs(mol)
        
        # Get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            features = MolecularFeaturizer.atom_to_features(atom)
            atom_features.append(features)
        
        # Get bond features and connectivity
        edge_index = []
        edge_features = []
        
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            features = MolecularFeaturizer.bond_to_features(bond)
            
            # Add both directions
            edge_index.extend([[start, end], [end, start]])
            edge_features.extend([features, features])
        
        # Get 3D coordinates if available
        conformer = mol.GetConformer() if mol.GetNumConformers() > 0 else None
        if conformer:
            positions = []
            for i in range(mol.GetNumAtoms()):
                pos = conformer.GetAtomPosition(i)
                positions.append([pos.x, pos.y, pos.z])
        else:
            # Generate 3D coordinates
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            conformer = mol.GetConformer()
            positions = []
            for i in range(mol.GetNumAtoms()):
                pos = conformer.GetAtomPosition(i)
                positions.append([pos.x, pos.y, pos.z])
        
        return {
            'atom_features': np.array(atom_features, dtype=np.float32),
            'edge_index': np.array(edge_index, dtype=np.int64).T,
            'edge_features': np.array(edge_features, dtype=np.float32),
            'positions': np.array(positions, dtype=np.float32)
        }

class MolecularMetrics:
    """Utility class for computing molecular metrics"""
    
    @staticmethod
    def compute_validity(smiles_list: List[str]) -> float:
        """Compute fraction of valid SMILES"""
        valid_count = 0
        for smiles in smiles_list:
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_count += 1
        return valid_count / len(smiles_list) if smiles_list else 0.0
    
    @staticmethod
    def compute_uniqueness(smiles_list: List[str]) -> float:
        """Compute fraction of unique molecules"""
        valid_smiles = []
        for smiles in smiles_list:
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    canonical_smiles = Chem.MolToSmiles(mol)
                    valid_smiles.append(canonical_smiles)
        
        if not valid_smiles:
            return 0.0
        
        unique_smiles = set(valid_smiles)
        return len(unique_smiles) / len(valid_smiles)
    
    @staticmethod
    def compute_drug_likeness(smiles_list: List[str]) -> Dict[str, float]:
        """Compute drug-likeness metrics"""
        metrics = {
            'lipinski_violations': [],
            'qed_scores': [],
            'sa_scores': []
        }
        
        for smiles in smiles_list:
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Lipinski rule of five
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    hbd = rdMolDescriptors.CalcNumHBD(mol)
                    hba = rdMolDescriptors.CalcNumHBA(mol)
                    
                    violations = sum([
                        mw > 500,
                        logp > 5,
                        hbd > 5,
                        hba > 10
                    ])
                    metrics['lipinski_violations'].append(violations)
                    
                    # QED score
                    try:
                        qed = Descriptors.qed(mol)
                        metrics['qed_scores'].append(qed)
                    except:
                        metrics['qed_scores'].append(0.0)
        
        # Compute averages
        result = {}
        for key, values in metrics.items():
            if values:
                result[f'avg_{key}'] = np.mean(values)
                result[f'std_{key}'] = np.std(values)
            else:
                result[f'avg_{key}'] = 0.0
                result[f'std_{key}'] = 0.0
        
        return result
    
    @staticmethod
    def compute_tanimoto_similarity(smiles1: str, smiles2: str) -> float:
        """Compute Tanimoto similarity between two molecules"""
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            
            if mol1 is None or mol2 is None:
                return 0.0
            
            fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
            
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except:
            return 0.0

class MolecularConverter:
    """Utility class for converting between molecular representations"""
    
    @staticmethod
    def graph_to_mol(atom_types: torch.Tensor, positions: torch.Tensor,
                    edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Optional[Chem.Mol]:
        """Convert graph representation to RDKit molecule"""
        try:
            mol = Chem.RWMol()
            
            # Add atoms
            atom_map = {}
            for i, atom_type in enumerate(atom_types):
                atomic_num = int(atom_type.item())
                if atomic_num > 0 and atomic_num <= 118:  # Valid atomic number
                    atom_idx = mol.AddAtom(Chem.Atom(atomic_num))
                    atom_map[i] = atom_idx
                else:
                    return None
            
            # Add bonds
            if edge_index.size(1) > 0:
                edge_index_np = edge_index.cpu().numpy()
                edge_attr_np = edge_attr.cpu().numpy()
                
                added_bonds = set()
                for i in range(edge_index.size(1)):
                    atom1, atom2 = edge_index_np[:, i]
                    bond_type = int(edge_attr_np[i])
                    
                    # Avoid duplicate bonds
                    bond_key = tuple(sorted([atom1, atom2]))
                    if bond_key in added_bonds:
                        continue
                    
                    if atom1 in atom_map and atom2 in atom_map and bond_type > 0:
                        bond_type_rdkit = [
                            Chem.BondType.SINGLE,
                            Chem.BondType.DOUBLE,
                            Chem.BondType.TRIPLE
                        ][min(bond_type - 1, 2)]
                        
                        mol.AddBond(atom_map[atom1], atom_map[atom2], bond_type_rdkit)
                        added_bonds.add(bond_key)
            
            # Add 3D coordinates
            if positions.size(0) == len(atom_map):
                conf = Chem.Conformer(len(atom_map))
                positions_np = positions.cpu().numpy()
                
                for i, pos in enumerate(positions_np):
                    if i in atom_map:
                        conf.SetAtomPosition(atom_map[i], tuple(pos))
                
                mol.AddConformer(conf)
            
            # Sanitize molecule
            Chem.SanitizeMol(mol)
            return mol
            
        except Exception as e:
            print(f"Error converting graph to molecule: {e}")
            return None
    
    @staticmethod
    def smiles_to_graph(smiles: str) -> Optional[Dict[str, torch.Tensor]]:
        """Convert SMILES to graph representation"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            graph_data = MolecularFeaturizer.mol_to_graph(mol)
            
            return {
                'x': torch.tensor(graph_data['atom_features'], dtype=torch.float),
                'pos': torch.tensor(graph_data['positions'], dtype=torch.float),
                'edge_index': torch.tensor(graph_data['edge_index'], dtype=torch.long),
                'edge_attr': torch.tensor(graph_data['edge_features'], dtype=torch.float)
            }
        except:
            return None

class ValenceConstraintLayer(nn.Module):
    """Layer that enforces valence constraints"""
    
    def __init__(self):
        super().__init__()
        # Chemical valence rules
        self.max_valences = {0: 4, 1: 3, 2: 2, 3: 6, 4: 1, 5: 1, 6: 1, 7: 1, 8: 4, 9: 4, 10: 4}
    
    def forward(self, h, edge_index, predicted_valences, atom_types):
        """Enforce valence constraints and compute violations"""
        if edge_index.size(1) == 0:
            return h, torch.tensor(0.0, device=h.device)
        
        # Count current bonds per atom
        row, col = edge_index
        bond_counts = torch.zeros(h.size(0), device=h.device)
        bond_counts.index_add_(0, row, torch.ones(row.size(0), device=h.device))
        
        # Compute valence violations
        violations = 0.0
        violation_mask = torch.zeros(h.size(0), device=h.device)
        
        for i, (atom_type, predicted_val, bond_count) in enumerate(zip(atom_types, predicted_valences, bond_counts)):
            max_allowed = self.max_valences.get(atom_type.item(), 4)
            
            if bond_count > max_allowed:
                violation = bond_count - max_allowed
                violations += violation ** 2
                violation_mask[i] = violation
        
        # Modify features for atoms with violations (chemical penalty)
        h = h * (1.0 - 0.1 * violation_mask.unsqueeze(1))  # Reduce feature magnitude for violating atoms
        
        return h, violations / h.size(0)


class BondConstraintLayer(nn.Module):
    """Layer that enforces bond type constraints"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, bond_logits, edge_index, atom_types):
        """Enforce bond type constraints"""
        if bond_logits.size(0) == 0:
            return bond_logits, torch.tensor(0.0, device=bond_logits.device)
        
        row, col = edge_index
        violations = 0.0
        
        # Chemical bond type constraints
        for i, (atom1, atom2) in enumerate(zip(row, col)):
            atom1_type = atom_types[atom1].item()
            atom2_type = atom_types[atom2].item()
            
            # Fluorine (4) can only form single bonds
            if atom1_type == 4 or atom2_type == 4:
                bond_logits[i, 1:] -= 100.0  # Heavy penalty for non-single bonds
                violations += torch.sum(torch.relu(bond_logits[i, 1:]))
            
            # Chlorine (5) typically single bonds
            if atom1_type == 5 or atom2_type == 5:
                bond_logits[i, 2:] -= 50.0  # Penalty for triple bonds
        
        return bond_logits, violations / bond_logits.size(0)

class ChemicalConstraintGNN(nn.Module):
    """GNN layer with chemical constraint awareness"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Message function with valence awareness
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),  # +2 for valences
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update function with chemical constraints
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Valence-aware attention
        self.valence_attention = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, h, edge_index, edge_attr, valences):
        """Forward pass with chemical constraints"""
        if edge_index.size(1) == 0:
            return h
        
        row, col = edge_index
        
        # Valence-aware attention weights
        valence_pairs = torch.stack([valences[row].float(), valences[col].float()], dim=-1)
        attention_weights = self.valence_attention(valence_pairs)
        
        # Messages with valence information
        messages = torch.cat([
            h[row], h[col], 
            valences[row].float().unsqueeze(1),
            valences[col].float().unsqueeze(1)
        ], dim=-1)
        
        messages = self.message_mlp(messages) * attention_weights
        
        # Aggregate messages
        h_agg = torch.zeros_like(h)
        h_agg.index_add_(0, row, messages)
        
        # Update with chemical awareness
        h_new = self.update_mlp(torch.cat([h, h_agg], dim=-1))
        
        return h + h_new 