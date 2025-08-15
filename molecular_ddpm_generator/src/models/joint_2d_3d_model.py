# src/models/joint_2d_3d_model.py - Enhanced with Chemical Constraints
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, radius_graph
from torch_geometric.utils import to_dense_batch
from .base_model import MolecularModel
from .egnn import CorrectedEGNNBackbone, create_corrected_egnn_backbone  
from utils.molecular_utils import BondConstraintLayer, ValenceConstraintLayer, ChemicalConstraintGNN

try:
    from .pocket_encoder import create_improved_pocket_encoder, SimplePocketEncoder
    IMPROVED_POCKET_AVAILABLE = True
except ImportError:
    IMPROVED_POCKET_AVAILABLE = False

def safe_global_pool(x, batch, pool_type='mean'):
    """Safe global pooling with fallback"""
    try:
        if pool_type == 'mean':
            return global_mean_pool(x, batch)
        else:
            return global_max_pool(x, batch)
    except Exception:
        unique_batch = torch.unique(batch)
        pooled = []
        for b in unique_batch:
            mask = batch == b
            if pool_type == 'mean':
                pooled.append(torch.mean(x[mask], dim=0))
            else:
                pooled.append(torch.max(x[mask], dim=0)[0])
        return torch.stack(pooled)

class ChemicalSpecialist2D(nn.Module):
    """Enhanced 2D processing with chemical constraints"""
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Chemical knowledge embeddings
        self.bond_type_embedding = nn.Embedding(5, 64)
        self.atom_type_embedding = nn.Embedding(11, 64)

        # Valence prediction network
        self.valence_predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)  # Max valence 8
        )
        
        # Chemical constraint enforcement
        self.valence_constraint_layer = ValenceConstraintLayer()
        
        # Chemical graph convolutions with constraints
        self.chemical_gnn = nn.ModuleList([
            ChemicalConstraintGNN(self.hidden_dim) for _ in range(3)
        ])
        
        # Bond type classifier with chemical rules
        self.bond_type_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, 128),  # +2 for valences
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Linear(64, 4)  # Single, Double, Triple, Aromatic
        )
        
        # Chemical property prediction with constraints
        self.chemical_properties = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
                
        self._initialize_chemical_knowledge()
        
    def _initialize_chemical_knowledge(self):
        # Chemical valence rules: C=4, N=3, O=2, S=6, F=1, Cl=1, etc.
        valence_init = torch.tensor([4, 3, 2, 6, 1, 1, 1, 1, 4, 4, 4], dtype=torch.float)
        
        # Initialize valence predictor with chemical knowledge
        with torch.no_grad():
            for atom_type, max_val in enumerate(valence_init):
                if max_val <= 8 and atom_type < 11:
                    # Set bias to prefer correct valence
                    self.valence_predictor[-1].bias[atom_type] = max_val - 1
        
    def forward(self, x, edge_index, edge_attr, batch):
        # Extract chemical features (no spatial coordinates)
        atom_types = x[:, 0].long().clamp(0, 10)

        # Chemical embeddings
        atom_emb = self.atom_type_embedding(atom_types)

        # Predict valence for each atom with chemical knowledge
        valence_logits = self.valence_predictor(atom_emb)
        predicted_valences = torch.argmax(valence_logits, dim=-1) + 1  # 1-8 valence
        
        # Bond embeddings with chemical constraints
        if edge_attr.size(1) > 0 and edge_index.size(1) > 0:
            bond_types = edge_attr[:, 0].long().clamp(0, 4)
            bond_emb = self.bond_type_embedding(bond_types)
        else:
            bond_emb = torch.zeros((0, 64), device=x.device)
        
        # Combine chemical features with valence information
        h_chemical = torch.cat([
            atom_emb, 
            F.one_hot(predicted_valences - 1, 8).float()  # Valence encoding
        ], dim=-1)
        
        # Pad to hidden_dim if needed
        if h_chemical.size(1) < self.hidden_dim:
            padding = torch.zeros(h_chemical.size(0), self.hidden_dim - h_chemical.size(1), 
                                device=h_chemical.device)
            h_chemical = torch.cat([h_chemical, padding], dim=-1)
        elif h_chemical.size(1) > self.hidden_dim:
            h_chemical = h_chemical[:, :self.hidden_dim]
        
        # Apply constraint-aware GNN layers
        for gnn_layer in self.chemical_gnn:
            h_chemical = gnn_layer(h_chemical, edge_index, bond_emb, predicted_valences)
        
        # Apply valence constraints and compute violations
        h_chemical, valence_violations = self.valence_constraint_layer(
            h_chemical, edge_index, predicted_valences, atom_types
        )
        
        # Predict bond types with chemical constraints
        bond_predictions = self._predict_constrained_bonds(h_chemical, edge_index, predicted_valences, atom_types)
        
        # Chemical analysis
        chemical_props = self.chemical_properties(h_chemical)
        
        return {
            'chemical_features': h_chemical,
            'chemical_properties': chemical_props,
            'atom_types': atom_types,
            'valence_predictions': valence_logits,
            'predicted_valences': predicted_valences,
            'bond_predictions': bond_predictions,
            'valence_violations': valence_violations
        }
    
    def _predict_constrained_bonds(self, h, edge_index, valences, atom_types):
        """Predict bond types with chemical constraints"""
        if edge_index.size(1) == 0:
            return torch.zeros((0, 4), device=h.device)
        
        row, col = edge_index
        
        # Bond features with valence and atom type information
        bond_features = torch.cat([
            h[row], h[col],
            valences[row].float().unsqueeze(1),
            valences[col].float().unsqueeze(1)
        ], dim=-1)
        
        # Predict bond types
        bond_logits = self.bond_type_classifier(bond_features)
        
        # Apply chemical constraints to bond predictions
        bond_logits = self._apply_chemical_bond_constraints(bond_logits, row, col, valences, atom_types)
        
        return bond_logits
    
    def _apply_chemical_bond_constraints(self, bond_logits, row, col, valences, atom_types):
        """Apply chemical constraints to bond predictions"""
        for i, (atom1_idx, atom2_idx) in enumerate(zip(row, col)):
            atom1_type = atom_types[atom1_idx].item()
            atom2_type = atom_types[atom2_idx].item()
            val1, val2 = valences[atom1_idx], valences[atom2_idx]
            
            # Fluorine (4) and halogens can only form single bonds
            if atom1_type in [4, 5] or atom2_type in [4, 5]:  # F, Cl
                bond_logits[i, 1:] -= 100.0  # Heavy penalty for non-single bonds
            
            # Low valence atoms (O=2) rarely form triple bonds
            if val1 <= 2 or val2 <= 2:
                bond_logits[i, 2] -= 50.0  # Penalty for triple bonds
            
            # Hydrogen-like atoms (valence 1) only single bonds
            if val1 <= 1 or val2 <= 1:
                bond_logits[i, 1:] -= 50.0  # Penalty for multiple bonds
        
        return bond_logits

class PhysicalSpecialist3D(nn.Module):    
    def __init__(self, hidden_dim=256, cutoff=10.0, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        
        # CORRECTED: Use proper EGNN implementation
        self.egnn_backbone = create_corrected_egnn_backbone(
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            cutoff=cutoff,
            sin_embedding=True,
            reflection_equiv=True
        )
        
        # Physical force modeling
        self.force_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 3)
        )
        
        # Non-bonded interaction detector
        self.interaction_classifier = nn.ModuleDict({
            'hydrogen_bond': nn.Linear(hidden_dim * 2 + 2, 1),
            'pi_pi_stacking': nn.Linear(hidden_dim * 2 + 2, 1),
            'van_der_waals': nn.Linear(hidden_dim * 2 + 2, 1)
        })
        
        # Conformation analysis
        self.conformer_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        print(f"✅ PhysicalSpecialist3D with CORRECTED EGNN")
        
    def forward(self, h, pos, batch, edge_index=None, edge_attr=None, atom_types=None):
        # CORRECTED EGNN processing with proper SE(3) equivariance
        egnn_outputs = self.egnn_backbone(
            h=h, pos=pos, batch=batch, 
            edge_index=edge_index, edge_attr=edge_attr,
            atom_types=atom_types
        )
        
        h_spatial = egnn_outputs['h']
        pos_final = egnn_outputs['pos']
        constraint_losses = egnn_outputs['constraint_losses']
        total_constraint_loss = egnn_outputs['total_constraint_loss']
        
        # Physical analysis
        forces = self.force_predictor(h_spatial)
        interactions = self._detect_interactions(h_spatial, pos_final, batch)
        conformer_features = self.conformer_analyzer(h_spatial)
        
        return {
            'spatial_features': h_spatial,
            'updated_positions': pos_final,
            'predicted_forces': forces,
            'interactions': interactions,
            'conformer_features': conformer_features,
            'constraint_losses': constraint_losses,
            'total_constraint_loss': total_constraint_loss
        }
    
    def _detect_interactions(self, h, pos, batch):
        interaction_edges = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=20)
        
        if interaction_edges.size(1) == 0:
            return {}
        
        row, col = interaction_edges
        distances = torch.norm(pos[row] - pos[col], dim=-1, keepdim=True)
        
        interactions = {}
        for interaction_type, classifier in self.interaction_classifier.items():
            normalized_distances = distances / 10.0
            edge_features = torch.cat([h[row], h[col], distances, normalized_distances], dim=-1)
            interaction_scores = torch.sigmoid(classifier(edge_features))
            interactions[interaction_type] = interaction_scores
        
        return interactions


class ComplementaryFusion(nn.Module):
    """Fuse complementary 2D chemical and 3D physical information with constraints"""
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        # Cross-attention with chemical bias
        self.chemical_to_physical = nn.MultiheadAttention(self.hidden_dim, num_heads=8, batch_first=True)
        self.physical_to_chemical = nn.MultiheadAttention(self.hidden_dim, num_heads=8, batch_first=True)

        
        # Feature fusion with constraint awareness
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + 32 + 64, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Consistency checker enhanced with chemical knowledge
        self.consistency_checker = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + 1, 64),  # +1 for valence info
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, chemical_output, physical_output):
        h_chem = chemical_output['chemical_features']
        h_phys = physical_output['spatial_features']
        
        # Cross-attention (requires 3D input for nn.MultiheadAttention)
        h_chem_3d = h_chem.unsqueeze(0)
        h_phys_3d = h_phys.unsqueeze(0)
        
        h_chem_attended, _ = self.chemical_to_physical(h_chem_3d, h_phys_3d, h_phys_3d)
        h_phys_attended, _ = self.physical_to_chemical(h_phys_3d, h_chem_3d, h_chem_3d)
        
        h_chem_attended = h_chem_attended.squeeze(0)
        h_phys_attended = h_phys_attended.squeeze(0)
        
        # Enforce chemical-physical consistency
        h_chem_consistent, h_phys_consistent = self.consistency_enforcer(
            h_chem_attended, h_phys_attended, chemical_output
        )
        
        # Integrate additional features
        chemical_props = chemical_output['chemical_properties']
        conformer_features = physical_output['conformer_features']
        
        # Fusion with chemical constraint awareness
        combined_features = torch.cat([
            h_chem_consistent,
            h_phys_consistent,
            chemical_props,
            conformer_features
        ], dim=-1)
        
        h_fused = self.fusion(combined_features)
        
        # Enhanced consistency check with valence information
        valence_info = chemical_output.get('valence_violations', torch.tensor(0.0))
        if isinstance(valence_info, torch.Tensor) and valence_info.numel() == 1:
            valence_info = valence_info.expand(h_chem.size(0), 1)
        elif not isinstance(valence_info, torch.Tensor):
            valence_info = torch.zeros(h_chem.size(0), 1, device=h_chem.device)
        
        consistency_input = torch.cat([h_chem, h_phys, valence_info], dim=-1)
        consistency_score = self.consistency_checker(consistency_input)
        
        return {
            'fused_features': h_fused,
            'consistency_score': consistency_score,
            'forces': physical_output['predicted_forces'],
            'interactions': physical_output['interactions'],
            'constraint_losses': physical_output['constraint_losses'],
            'total_constraint_loss': physical_output['total_constraint_loss'],
            'chemical_violations': chemical_output.get('valence_violations', 0.0)
        }

class Joint2D3DModel(MolecularModel):
    """Enhanced Joint 2D-3D Molecular Model with integrated chemical constraints"""
    
    def __init__(self, atom_types=11, bond_types=4, hidden_dim=256, 
                 pocket_dim=256, num_layers=6, max_radius=10.0,
                 max_pocket_atoms=1000, conditioning_type="add"):
        super().__init__(atom_types, bond_types, hidden_dim)
        
        self.hidden_dim = hidden_dim
        self.pocket_dim = pocket_dim
        self.conditioning_type = conditioning_type
        
        # Atom embedding
        self.atom_embedding = nn.Linear(6, self.hidden_dim)
        
        # Enhanced specialized processing branches with chemical constraints
        self.chemical_2d = ChemicalSpecialist2D(self.hidden_dim)
        self.physical_3d = PhysicalSpecialist3D(
            self.hidden_dim, 
            cutoff=max_radius,
            num_layers=num_layers
        )
        
        # Enhanced complementary fusion with chemical awareness
        self.fusion = ComplementaryFusion(self.hidden_dim)
        
        # Pocket encoder
        if IMPROVED_POCKET_AVAILABLE:
            self.pocket_encoder = create_improved_pocket_encoder(
                hidden_dim=self.hidden_dim,
                output_dim=pocket_dim
            )
        else:
            self.pocket_encoder = SimplePocketEncoder(
                input_dim=7,
                hidden_dim=self.hidden_dim,
                output_dim=pocket_dim,
                max_atoms=max_pocket_atoms
            )
        
        # Conditioning
        if conditioning_type == "add":
            assert pocket_dim == self.hidden_dim
            self.condition_transform = nn.Identity()
        else:
            self.condition_transform = nn.Linear(pocket_dim, self.hidden_dim)

        self.position_head = nn.Linear(self.hidden_dim, 3)

            
    def forward(self, x, pos, edge_index, edge_attr, batch,
                pocket_x=None, pocket_pos=None, pocket_edge_index=None, 
                pocket_batch=None, **kwargs):
        
        # Initial embedding
        h_init = self._embed_atoms_flexible(x)
        
        # Enhanced specialized processing with chemical constraints
        chemical_output = self.chemical_2d(x, edge_index, edge_attr, batch)
        physical_output = self.physical_3d(
            h_init, pos, batch, edge_index, edge_attr, 
            atom_types=chemical_output.get('atom_types')
        )
        
        # Enhanced complementary fusion with chemical awareness
        fusion_output = self.fusion(chemical_output, physical_output)
        
        # Pocket conditioning
        pocket_condition = self._encode_pocket(
            pocket_x, pocket_pos, pocket_edge_index, pocket_batch, batch
        )
        
        h_final = fusion_output['fused_features']
        if pocket_condition is not None:
            h_final = self._apply_conditioning(h_final, pocket_condition, batch)
        
        pos_pred = physical_output['updated_positions'] + self.position_head(h_final)
        
        # Compute chemical constraint losses
        constraint_losses = self.chemical_loss_computer(
            chemical_output, fusion_output, edge_index
        )
        
        return {
            'pos_pred': pos_pred,
            'node_features': h_final,
            'chemical_properties': chemical_output['chemical_properties'],
            'physical_forces': fusion_output['forces'],
            'interactions': fusion_output['interactions'],
            'consistency_score': fusion_output['consistency_score'],
            'valence_predictions': chemical_output['valence_predictions'],
            'predicted_valences': chemical_output['predicted_valences'],
            'chemical_violations': fusion_output.get('chemical_violations', 0.0),
            'constraint_losses': constraint_losses,
            'total_constraint_loss': sum(constraint_losses.values()) if constraint_losses else torch.tensor(0.0)
        }
    
    def _embed_atoms_flexible(self, x):
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got {x.dim()}D")
        
        input_dim = x.size(1)
        if input_dim < 6:
            padding = torch.zeros(x.size(0), 6 - input_dim, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
            return self.atom_embedding(x_padded.float())
        elif input_dim > 6:
            x_truncated = x[:, :6]
            return self.atom_embedding(x_truncated.float())
        else:
            return self.atom_embedding(x.float())
    
    def _encode_pocket(self, pocket_x, pocket_pos, pocket_edge_index, 
                      pocket_batch, ligand_batch):
        if pocket_x is None or pocket_pos is None:
            return None
        
        try:
            pocket_repr = self.pocket_encoder(
                pocket_x, pocket_pos, pocket_edge_index, pocket_batch
            )
            return pocket_repr
        except Exception:
            batch_size = ligand_batch.max().item() + 1
            return torch.zeros(batch_size, self.pocket_dim, device=ligand_batch.device)
    
    def _apply_conditioning(self, atom_features, pocket_condition, batch):
        if pocket_condition is None:
            return atom_features
        
        pocket_transformed = self.condition_transform(pocket_condition)
        
        # Handle batch size mismatch
        batch_size = pocket_transformed.size(0)
        max_batch_idx = batch.max().item()
        
        if max_batch_idx >= batch_size:
            last_condition = pocket_transformed[-1:].expand(max_batch_idx + 1 - batch_size, -1)
            pocket_transformed = torch.cat([pocket_transformed, last_condition], dim=0)
        
        batch_safe = torch.clamp(batch, 0, pocket_transformed.size(0) - 1)
        broadcasted_condition = pocket_transformed[batch_safe]
        
        return atom_features + broadcasted_condition
    
    def _compute_bond_consistency_loss(self, bond_logits, edge_index):
        """Compute bond type consistency loss"""
        if bond_logits.size(0) == 0:
            return torch.tensor(0.0)
        
        try:
            # Penalize excessive high-order bonds
            bond_probs = torch.softmax(bond_logits, dim=-1)
            
            # Triple bonds should be rare in drug-like molecules
            triple_bond_penalty = bond_probs[:, 2].mean() * 2.0 if bond_probs.size(1) > 2 else torch.tensor(0.0)
            
            # Moderate penalty for too many double bonds
            double_bond_excess = torch.clamp(bond_probs[:, 1].mean() - 0.2, min=0.0) if bond_probs.size(1) > 1 else torch.tensor(0.0)
            
            return triple_bond_penalty + double_bond_excess
            
        except Exception as e:
            return torch.tensor(0.0)
    
    def _compute_atom_valence_compatibility(self, atom_logits, predicted_valences):
        if atom_logits is None or predicted_valences is None:
            return torch.tensor(0.0)
        
        try:
            predicted_atoms = torch.argmax(atom_logits, dim=-1)
            
            compatibility_loss = 0.0
            for atom, valence in zip(predicted_atoms, predicted_valences):
                expected_valence = self.valence_rules.get(atom.item(), 4)
                if valence != expected_valence:
                    compatibility_loss += (valence - expected_valence) ** 2
            
            return compatibility_loss / len(predicted_atoms) if len(predicted_atoms) > 0 else torch.tensor(0.0)
            
        except Exception as e:
            return torch.tensor(0.0)


def create_joint2d3d_model(hidden_dim=256, num_layers=6, conditioning_type="add", **kwargs):
    return Joint2D3DModel(
        atom_types=11,
        bond_types=4,
        hidden_dim=hidden_dim,
        pocket_dim=hidden_dim,
        num_layers=num_layers,
        conditioning_type=conditioning_type,
        **kwargs
    )