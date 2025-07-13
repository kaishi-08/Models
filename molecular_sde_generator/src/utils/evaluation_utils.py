import numpy as np
import torch
from typing import List, Dict, Any
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from scipy.spatial.distance import pdist, squareform

class MolecularEvaluator:
    """Comprehensive molecular evaluation utilities"""
    
    @staticmethod
    def compute_diversity(smiles_list: List[str], metric: str = 'tanimoto') -> float:
        """Compute molecular diversity"""
        valid_mols = []
        for smiles in smiles_list:
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_mols.append(mol)
        
        if len(valid_mols) < 2:
            return 0.0
        
        if metric == 'tanimoto':
            fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in valid_mols]
            similarities = []
            
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    similarities.append(sim)
            
            diversity = 1 - np.mean(similarities)
            return diversity
        
        return 0.0
    
    @staticmethod
    def compute_fcd(generated_smiles: List[str], reference_smiles: List[str]) -> float:
        """Compute Fréchet ChemNet Distance (simplified version)"""
        # This is a placeholder - actual FCD requires ChemNet features
        # You would need to implement proper FCD calculation
        return 0.0
    
    @staticmethod
    def compute_scaffold_diversity(smiles_list: List[str]) -> float:
        """Compute Bemis-Murcko scaffold diversity"""
        from rdkit.Chem.Scaffolds import MurckoScaffold
        
        scaffolds = set()
        valid_count = 0
        
        for smiles in smiles_list:
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_count += 1
                    try:
                        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                        scaffold_smiles = Chem.MolToSmiles(scaffold)
                        scaffolds.add(scaffold_smiles)
                    except:
                        continue
        
        if valid_count == 0:
            return 0.0
        
        return len(scaffolds) / valid_count