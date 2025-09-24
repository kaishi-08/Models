#!/usr/bin/env python3
"""
Code để phát hiện conflict giữa OpenBabel và RDKit
và debug dữ liệu gây lỗi
"""

import numpy as np
import torch
import tempfile
import traceback
from pathlib import Path
import sys

def check_version_compatibility():
    """Kiểm tra phiên bản và compatibility của OpenBabel và RDKit"""
    print("=" * 60)
    print("CHECKING OPENBABEL & RDKIT COMPATIBILITY")
    print("=" * 60)
    
    # 1. Kiểm tra import
    try:
        import openbabel
        print("✓ OpenBabel import: OK")
        print(f"  OpenBabel version: {openbabel.OBReleaseVersion()}")
    except ImportError as e:
        print(f"✗ OpenBabel import: FAILED - {e}")
        return False
    
    try:
        from rdkit import Chem
        from rdkit import __version__ as rdkit_version
        print("✓ RDKit import: OK")
        print(f"  RDKit version: {rdkit_version}")
    except ImportError as e:
        print(f"✗ RDKit import: FAILED - {e}")
        return False
    
    # 2. Test cơ bản OpenBabel
    try:
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "sdf")
        ob_mol = openbabel.OBMol()
        print("✓ OpenBabel basic operations: OK")
    except Exception as e:
        print(f"✗ OpenBabel basic operations: FAILED - {e}")
        return False
    
    # 3. Test cơ bản RDKit
    try:
        mol = Chem.MolFromSmiles("CCO")
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
        print("✓ RDKit basic operations: OK")
    except Exception as e:
        print(f"✗ RDKit basic operations: FAILED - {e}")
        return False
    
    # 4. Test tích hợp OpenBabel -> RDKit
    try:
        test_xyz_content = """3
Test molecule
C    0.0000    0.0000    0.0000
C    1.5000    0.0000    0.0000  
O    0.0000    1.2000    0.0000
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as tmp:
            tmp.write(test_xyz_content)
            tmp_xyz = tmp.name
        
        with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp:
            tmp_sdf = tmp.name
        
        # OpenBabel conversion
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "sdf")
        ob_mol = openbabel.OBMol()
        obConversion.ReadFile(ob_mol, tmp_xyz)
        obConversion.WriteFile(ob_mol, tmp_sdf)
        
        # RDKit reading
        mol = Chem.SDMolSupplier(tmp_sdf, sanitize=False)[0]
        
        if mol is not None:
            print("✓ OpenBabel -> RDKit integration: OK")
        else:
            print("✗ OpenBabel -> RDKit integration: FAILED - mol is None")
            return False
            
        # Cleanup
        Path(tmp_xyz).unlink()
        Path(tmp_sdf).unlink()
        
    except Exception as e:
        print(f"✗ OpenBabel -> RDKit integration: FAILED - {e}")
        traceback.print_exc()
        return False
    
    print("✓ ALL COMPATIBILITY TESTS PASSED")
    return True


def debug_specific_molecules(positions, one_hot, mask, max_debug=10):
    """Debug từng molecule cụ thể để tìm molecule gây lỗi"""
    print("\n" + "=" * 60)
    print("DEBUGGING SPECIFIC MOLECULES")
    print("=" * 60)
    
    atom_types = np.argmax(one_hot, axis=-1)
    sections = np.where(np.diff(mask))[0] + 1
    positions_split = [torch.from_numpy(x) for x in np.split(positions, sections)]
    atom_types_split = [torch.from_numpy(x) for x in np.split(atom_types, sections)]
    
    problematic_molecules = []
    
    print(f"Total molecules: {len(positions_split)}")
    print(f"Debugging first {min(max_debug, len(positions_split))} molecules...\n")
    
    from src.utils.molecule_builder import build_molecule
    from src.utils.constants import dataset_params
    
    # Giả sử sử dụng crossdock dataset
    dataset_info = dataset_params['crossdock']
    
    for i, (pos, atom_type) in enumerate(zip(positions_split[:max_debug], atom_types_split[:max_debug])):
        print(f"Molecule {i}:")
        print(f"  Positions shape: {pos.shape}")
        print(f"  Atom types shape: {atom_type.shape}")
        print(f"  Atom types: {atom_type.tolist()}")
        print(f"  Position stats: min={pos.min():.3f}, max={pos.max():.3f}, mean={pos.mean():.3f}")
        
        # Kiểm tra dữ liệu không hợp lệ
        invalid_data = False
        
        if torch.isnan(pos).any():
            print(f"  ✗ WARNING: NaN detected in positions!")
            invalid_data = True
            
        if torch.isinf(pos).any():
            print(f"  ✗ WARNING: Inf detected in positions!")
            invalid_data = True
            
        if pos.shape[0] == 0:
            print(f"  ✗ WARNING: Empty molecule!")
            invalid_data = True
            
        # Kiểm tra atom types hợp lệ
        unique_atoms = torch.unique(atom_type).tolist()
        atom_decoder = dataset_info["atom_decoder"]
        for atom_idx in unique_atoms:
            if atom_idx >= len(atom_decoder):
                print(f"  ✗ WARNING: Invalid atom type index {atom_idx} (max: {len(atom_decoder)-1})")
                invalid_data = True
        
        if invalid_data:
            print(f"  ✗ INVALID DATA - Skipping molecule {i}")
            problematic_molecules.append((i, "Invalid input data"))
            continue
        
        # Test build_molecule với cả hai phương pháp
        print(f"  Testing OpenBabel method...")
        try:
            mol_ob = build_molecule(pos, atom_type, dataset_info, use_openbabel=True)
            if mol_ob is None:
                print(f"    ✗ OpenBabel returned None")
                problematic_molecules.append((i, "OpenBabel returned None"))
            else:
                print(f"    ✓ OpenBabel success: {mol_ob.GetNumAtoms()} atoms")
        except Exception as e:
            print(f"    ✗ OpenBabel failed: {type(e).__name__}: {e}")
            problematic_molecules.append((i, f"OpenBabel exception: {e}"))
        
        print(f"  Testing EDM method...")
        try:
            mol_edm = build_molecule(pos, atom_type, dataset_info, use_openbabel=False)
            if mol_edm is None:
                print(f"    ✗ EDM returned None")
            else:
                print(f"    ✓ EDM success: {mol_edm.GetNumAtoms()} atoms")
        except Exception as e:
            print(f"    ✗ EDM failed: {type(e).__name__}: {e}")
        
        # Test sanitization nếu có molecule
        if 'mol_ob' in locals() and mol_ob is not None:
            try:
                from rdkit import Chem
                Chem.SanitizeMol(mol_ob)
                print(f"    ✓ Sanitization success")
            except Exception as e:
                print(f"    ✗ Sanitization failed: {type(e).__name__}: {e}")
                problematic_molecules.append((i, f"Sanitization failed: {e}"))
        
        print()
    
    return problematic_molecules


def create_test_molecule_data():
    """Tạo dữ liệu test đơn giản để kiểm tra"""
    print("\n" + "=" * 60)
    print("TESTING WITH SIMPLE MOLECULES")
    print("=" * 60)
    
    # Molecule 1: Ethanol (CCO)
    pos1 = torch.tensor([
        [0.0000, 0.0000, 0.0000],  # C
        [1.5000, 0.0000, 0.0000],  # C
        [2.0000, 1.2000, 0.0000],  # O
    ], dtype=torch.float32)
    
    atom_types1 = torch.tensor([0, 0, 2])  # C, C, O (assuming C=0, O=2)
    
    # Molecule 2: Water (H2O) - nếu có H trong dataset
    pos2 = torch.tensor([
        [0.0000, 0.0000, 0.0000],  # O
        [1.0000, 0.0000, 0.0000],  # H (giả sử)
        [-0.3333, 0.9428, 0.0000],  # H (giả sử)
    ], dtype=torch.float32)
    
    atom_types2 = torch.tensor([2, 1, 1])  # O, N, N (thay H bằng N nếu không có H)
    
    # Combine data
    positions = torch.cat([pos1, pos2], dim=0)
    atom_types_combined = torch.cat([atom_types1, atom_types2], dim=0)
    
    # Create one-hot encoding
    num_atom_types = 10  # Giả sử có 10 loại atom
    one_hot = torch.zeros(len(atom_types_combined), num_atom_types)
    one_hot[torch.arange(len(atom_types_combined)), atom_types_combined] = 1
    
    # Create mask
    mask = torch.tensor([0, 0, 0, 1, 1, 1])  # molecule 0: 3 atoms, molecule 1: 3 atoms
    
    return positions.numpy(), one_hot.numpy(), mask.numpy()


def run_comprehensive_debug():
    """Chạy tất cả các test debug"""
    print("Starting comprehensive OpenBabel-RDKit debug...\n")
    
    # 1. Check compatibility
    if not check_version_compatibility():
        print("❌ COMPATIBILITY ISSUES DETECTED!")
        print("Recommendations:")
        print("  - Update RDKit: conda update rdkit")
        print("  - Update OpenBabel: conda update openbabel")
        print("  - Or try: pip install --upgrade rdkit openbabel")
        return False
    
    # 2. Test with simple molecules
    try:
        test_pos, test_one_hot, test_mask = create_test_molecule_data()
        print(f"Created test data: {test_pos.shape[0]} atoms, {len(np.unique(test_mask))} molecules")
        
        problematic = debug_specific_molecules(test_pos, test_one_hot, test_mask, max_debug=5)
        
        if problematic:
            print(f"❌ Found {len(problematic)} problematic test molecules!")
            for mol_id, error in problematic:
                print(f"  Molecule {mol_id}: {error}")
        else:
            print("✅ All test molecules processed successfully!")
            
    except Exception as e:
        print(f"❌ Test molecule debug failed: {e}")
        traceback.print_exc()
        return False
    
    return True


def debug_real_data(lig_coords, lig_one_hot, lig_mask, start_idx=0, max_debug=20):
    """Debug dữ liệu thực tế từ vị trí start_idx"""
    print(f"\n" + "=" * 60)
    print(f"DEBUGGING REAL DATA (starting from molecule {start_idx})")
    print("=" * 60)
    
    print(f"Dataset shape: coords={lig_coords.shape}, one_hot={lig_one_hot.shape}, mask={lig_mask.shape}")
    print(f"Unique molecules: {len(np.unique(lig_mask))}")
    
    # Lấy subset data để debug
    unique_masks = np.unique(lig_mask)
    if start_idx >= len(unique_masks):
        print(f"❌ start_idx {start_idx} >= number of molecules {len(unique_masks)}")
        return
    
    end_idx = min(start_idx + max_debug, len(unique_masks))
    debug_masks = unique_masks[start_idx:end_idx]
    
    # Filter data cho molecules cần debug
    mask_filter = np.isin(lig_mask, debug_masks)
    debug_coords = lig_coords[mask_filter]
    debug_one_hot = lig_one_hot[mask_filter]
    debug_mask = lig_mask[mask_filter]
    
    # Renumber masks to start from 0
    unique_debug_masks = np.unique(debug_mask)
    for i, old_mask in enumerate(unique_debug_masks):
        debug_mask[debug_mask == old_mask] = i
    
    print(f"Debugging molecules {start_idx} to {end_idx-1}")
    print(f"Debug data shape: coords={debug_coords.shape}, one_hot={debug_one_hot.shape}")
    
    return debug_specific_molecules(debug_coords, debug_one_hot, debug_mask, max_debug=len(unique_debug_masks))


if __name__ == "__main__":
    # Chạy comprehensive debug
    success = run_comprehensive_debug()
    
    if success:
        print("\n" + "🎉" * 20)
        print("COMPATIBILITY TESTS PASSED!")
        print("The issue is likely with specific data, not version conflicts.")
        print("🎉" * 20)
        
        print("\nNext steps:")
        print("1. Run debug_real_data() on your actual dataset")
        print("2. Use compute_smiles_safe() to handle problematic molecules")
        print("3. Check for data quality issues in problematic molecules")
    else:
        print("\n" + "❌" * 20) 
        print("COMPATIBILITY ISSUES DETECTED!")
        print("Fix environment issues before proceeding.")
        print("❌" * 20)