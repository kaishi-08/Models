#!/usr/bin/env python3
# debug_preprocessing.py - Debug preprocessing issues

import pickle
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
import traceback

def debug_single_sample():
    print("🔍 Debugging single sample processing...")
    
    # Load index
    index_file = Path("data/crossdocked_pocket10/index.pkl")
    if not index_file.exists():
        print("❌ Index file not found!")
        return
    
    with open(index_file, 'rb') as f:
        index_data = pickle.load(f)
    
    print(f"✅ Index loaded: {len(index_data)} entries")
    
    # Test first 10 entries
    data_dir = Path("data/crossdocked_pocket10")
    
    for i, entry in enumerate(index_data[:10]):
        print(f"\n📋 Testing entry {i+1}: {entry}")
        
        if len(entry) < 4:
            print("   ❌ Invalid entry format")
            continue
        
        pocket_file, ligand_file, receptor_file, score = entry[:4]
        
        # Test ligand file
        ligand_path = data_dir / ligand_file
        print(f"   Ligand: {ligand_file}")
        print(f"   Full path: {ligand_path}")
        print(f"   Exists: {ligand_path.exists()}")
        
        if ligand_path.exists():
            try:
                # Try to load ligand
                if ligand_path.suffix.lower() == '.sdf':
                    supplier = Chem.SDMolSupplier(str(ligand_path))
                    mol = None
                    for m in supplier:
                        if m is not None:
                            mol = m
                            break
                    
                    if mol is not None:
                        print(f"   ✅ Ligand loaded: {mol.GetNumAtoms()} atoms")
                        
                        # Test conformer
                        if mol.GetNumConformers() == 0:
                            print("   ⚠️  No conformer, generating...")
                            try:
                                AllChem.EmbedMolecule(mol, randomSeed=42)
                                AllChem.MMFFOptimizeMolecule(mol)
                                print("   ✅ Conformer generated")
                            except Exception as e:
                                print(f"   ❌ Conformer generation failed: {e}")
                                continue
                        else:
                            print("   ✅ Conformer exists")
                        
                        # Test SMILES
                        smiles = Chem.MolToSmiles(mol)
                        print(f"   ✅ SMILES: {smiles}")
                        
                        # This would be a successful ligand
                        print("   ✅ Ligand processing would succeed")
                        
                    else:
                        print("   ❌ Failed to load molecule from SDF")
                        
                else:
                    print(f"   ❌ Unsupported file format: {ligand_path.suffix}")
                    
            except Exception as e:
                print(f"   ❌ Error processing ligand: {e}")
                traceback.print_exc()
        else:
            # Try to find alternative files
            parent_dir = ligand_path.parent
            print(f"   Parent dir: {parent_dir}")
            print(f"   Parent exists: {parent_dir.exists()}")
            
            if parent_dir.exists():
                sdf_files = list(parent_dir.glob("*.sdf"))
                mol_files = list(parent_dir.glob("*.mol"))
                mol2_files = list(parent_dir.glob("*.mol2"))
                
                print(f"   SDF files: {len(sdf_files)}")
                print(f"   MOL files: {len(mol_files)}")
                print(f"   MOL2 files: {len(mol2_files)}")
                
                if sdf_files:
                    print(f"   Found SDF files: {[f.name for f in sdf_files]}")
                    
                    # Try first SDF file
                    try:
                        supplier = Chem.SDMolSupplier(str(sdf_files[0]))
                        mol = None
                        for m in supplier:
                            if m is not None:
                                mol = m
                                break
                        
                        if mol is not None:
                            print(f"   ✅ Alternative SDF loaded: {mol.GetNumAtoms()} atoms")
                        else:
                            print("   ❌ Alternative SDF failed")
                    except Exception as e:
                        print(f"   ❌ Alternative SDF error: {e}")
        
        # Test pocket file
        pocket_path = data_dir / pocket_file
        print(f"   Pocket: {pocket_file}")
        print(f"   Full path: {pocket_path}")
        print(f"   Exists: {pocket_path.exists()}")
        
        if pocket_path.exists():
            try:
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure('pocket', pocket_path)
                
                atom_count = 0
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            for atom in residue:
                                atom_count += 1
                
                print(f"   ✅ Pocket loaded: {atom_count} atoms")
                
            except Exception as e:
                print(f"   ❌ Error processing pocket: {e}")
        else:
            # Try to find alternative pocket files
            parent_dir = pocket_path.parent
            if parent_dir.exists():
                pdb_files = list(parent_dir.glob("*.pdb"))
                print(f"   PDB files in dir: {len(pdb_files)}")
                if pdb_files:
                    print(f"   Found PDB files: {[f.name for f in pdb_files]}")
        
        print(f"   Score: {score}")

def test_file_formats():
    print("\n🧪 Testing different file formats...")
    
    data_dir = Path("data/crossdocked_pocket10")
    
    # Find some actual files
    sdf_files = list(data_dir.glob("**/*.sdf"))
    pdb_files = list(data_dir.glob("**/*.pdb"))
    mol_files = list(data_dir.glob("**/*.mol"))
    mol2_files = list(data_dir.glob("**/*.mol2"))
    
    print(f"Found file types:")
    print(f"   SDF files: {len(sdf_files)}")
    print(f"   PDB files: {len(pdb_files)}")
    print(f"   MOL files: {len(mol_files)}")
    print(f"   MOL2 files: {len(mol2_files)}")
    
    # Test a few SDF files
    if sdf_files:
        print(f"\n🧪 Testing SDF files:")
        for i, sdf_file in enumerate(sdf_files[:3]):
            print(f"   {i+1}. {sdf_file}")
            try:
                supplier = Chem.SDMolSupplier(str(sdf_file))
                mol_count = 0
                for mol in supplier:
                    if mol is not None:
                        mol_count += 1
                print(f"      ✅ {mol_count} molecules loaded")
            except Exception as e:
                print(f"      ❌ Error: {e}")
    
    # Test a few PDB files
    if pdb_files:
        print(f"\n🧪 Testing PDB files:")
        for i, pdb_file in enumerate(pdb_files[:3]):
            print(f"   {i+1}. {pdb_file}")
            try:
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure('test', pdb_file)
                atom_count = sum(1 for model in structure for chain in model 
                               for residue in chain for atom in residue)
                print(f"      ✅ {atom_count} atoms loaded")
            except Exception as e:
                print(f"      ❌ Error: {e}")

if __name__ == "__main__":
    debug_single_sample()
    test_file_formats()