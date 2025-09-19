import numpy as np
import torch
from pathlib import Path
import pickle
import logging
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from src.utils.constants import dataset_params
from src.utils.molecule_builder import build_molecule

# Thiết lập logging với format tốt hơn
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('check_crossdock_validity.log'),
        logging.StreamHandler()  # Thêm log ra console
    ]
)

def check_ligand_and_pocket(pdbfile, sdffile, atom_dict, dist_cutoff=8.0):
    """
    Kiểm tra tính hợp lệ của ligand và pocket
    
    Returns:
        dict: Status với các thông tin chi tiết
    """
    status = {
        'valid': False, 
        'reason': '', 
        'lig_atoms': 0, 
        'pocket_atoms': 0, 
        'build_success': False,
        'ligand_heavy_atoms': 0,
        'pocket_residues': 0,
        'others_count': 0  # Đếm số atoms được map vào 'others'
    }

    # Kiểm tra file tồn tại
    if not pdbfile.exists():
        status['reason'] = f"PDB file not found: {pdbfile}"
        return status
    
    if not sdffile.exists():
        status['reason'] = f"SDF file not found: {sdffile}"
        return status

    # Parse PDB
    try:
        pdb_struct = PDBParser(QUIET=True).get_structure('', str(pdbfile))
        chains = list(pdb_struct.get_chains())
        if not chains:
            status['reason'] = f"No chains in PDB: {pdbfile}"
            return status
    except Exception as e:
        status['reason'] = f"Failed to parse PDB {pdbfile}: {str(e)}"
        return status

    # Parse SDF
    try:
        supplier = Chem.SDMolSupplier(str(sdffile), sanitize=False)
        ligand = supplier[0] if len(supplier) > 0 else None
        
        if ligand is None:
            status['reason'] = f"No valid molecule in SDF: {sdffile}"
            return status
            
        # Đếm heavy atoms (không phải H)
        status['ligand_heavy_atoms'] = sum(1 for atom in ligand.GetAtoms() 
                                          if atom.GetAtomicNum() > 1)
        
        # Kiểm tra số lượng heavy atoms tối thiểu
        if status['ligand_heavy_atoms'] < 5:
            status['reason'] = f"Too few heavy atoms in ligand: {status['ligand_heavy_atoms']}"
            return status
            
        try:
            Chem.SanitizeMol(ligand)
        except (ValueError, Chem.AtomValenceException) as e:
            status['reason'] = f"Sanitize failed for SDF {sdffile}: {str(e)}"
            return status
            
    except Exception as e:
        status['reason'] = f"Cannot read SDF {sdffile}: {str(e)}"
        return status

    # Lấy thông tin ligand atoms
    try:
        lig_atoms = []
        lig_coords = []
        others_count = 0
        
        for idx, atom in enumerate(ligand.GetAtoms()):
            symbol = atom.GetSymbol().capitalize()
            if atom.GetAtomicNum() > 1:  # Bỏ qua H
                if symbol in atom_dict:
                    lig_atoms.append(symbol)
                    pos = ligand.GetConformer(0).GetAtomPosition(idx)
                    lig_coords.append([pos.x, pos.y, pos.z])
                    if symbol == 'others':
                        others_count += 1
                elif 'others' in atom_dict:
                    # Map atom không xác định vào 'others' 
                    lig_atoms.append('others')
                    pos = ligand.GetConformer(0).GetAtomPosition(idx)
                    lig_coords.append([pos.x, pos.y, pos.z])
                    others_count += 1
                    logging.info(f"Mapped unknown atom {symbol} to 'others' in {sdffile}")
        
        lig_coords = np.array(lig_coords)
        status['lig_atoms'] = len(lig_atoms)
        status['others_count'] = others_count
        
        if status['lig_atoms'] == 0:
            status['reason'] = f"No valid heavy atoms in ligand {sdffile}"
            return status
            
        # Cảnh báo nếu có quá nhiều 'others' atoms
        if others_count > status['lig_atoms'] * 0.3:  # >30% là others
            status['reason'] = f"Too many unknown atoms ({others_count}/{status['lig_atoms']}) in ligand {sdffile}"
            return status
            
    except Exception as e:
        status['reason'] = f"Error processing ligand atoms: {str(e)}"
        return status

    # Tìm pocket residues
    pocket_residues = []
    try:
        for residue in pdb_struct[0].get_residues():
            if not is_aa(residue.get_resname(), standard=True):
                continue
                
            res_atoms = [a for a in residue.get_atoms() if a.element.capitalize() != 'H']
            if len(res_atoms) == 0:
                continue
                
            res_coords = np.array([a.get_coord() for a in res_atoms])
            
            # Tính khoảng cách tối thiểu
            distances = np.sqrt(np.sum(
                (res_coords[:, None, :] - lig_coords[None, :, :])**2, axis=-1
            ))
            min_dist = np.min(distances)
            
            if min_dist < dist_cutoff:
                pocket_residues.append(residue)
                
        status['pocket_residues'] = len(pocket_residues)
        
    except Exception as e:
        status['reason'] = f"Error computing pocket residues for {pdbfile}: {str(e)}"
        return status

    if not pocket_residues:
        status['reason'] = f"No pocket residues within {dist_cutoff}Å of ligand"
        return status

    # Xử lý pocket atoms
    try:
        pocket_atoms = []
        pocket_coords = []
        
        for residue in pocket_residues:
            for atom in residue.get_atoms():
                element = atom.element.capitalize()
                if element != 'H':  # Bỏ qua H
                    if element in atom_dict:
                        pocket_atoms.append(element)
                        pocket_coords.append(atom.coord)
                    elif 'others' in atom_dict:
                        # Map atom không xác định vào 'others'
                        pocket_atoms.append('others')
                        pocket_coords.append(atom.coord)
        
        status['pocket_atoms'] = len(pocket_atoms)
        
        if status['pocket_atoms'] == 0:
            status['reason'] = f"No valid pocket atoms found"
            return status
            
    except Exception as e:
        status['reason'] = f"Error processing pocket atoms: {str(e)}"
        return status

    # Test molecule building
    try:
        atom_types = np.array([atom_dict[atm] for atm in lig_atoms])
        lig_coords_torch = torch.tensor(lig_coords, dtype=torch.float32)
        atom_types_torch = torch.tensor(atom_types, dtype=torch.int64)
        
        mol = build_molecule(
            lig_coords_torch, 
            atom_types_torch, 
            dataset_params['crossdock_full'], 
            add_coords=True, 
            use_openbabel=False
        )
        
        if mol is None:
            status['reason'] = f"Build molecule returned None"
            return status
            
        Chem.SanitizeMol(mol)
        status['build_success'] = True
        
    except Exception as e:
        status['reason'] = f"Build/Sanitize molecule failed: {str(e)}"
        return status

    # Tất cả checks đã pass
    status['valid'] = True
    status['reason'] = "All checks passed"
    return status

def check_crossdock_database(index_path, data_dir, atom_dict, dist_cutoff=8.0, 
                            output_valid_index=True):
    """
    Kiểm tra database CrossDock và tạo index mới chỉ chứa valid samples
    
    Args:
        index_path: Path tới file index.pkl
        data_dir: Thư mục chứa data
        atom_dict: Dictionary mapping atom symbols
        dist_cutoff: Khoảng cách cutoff cho pocket
        output_valid_index: Có tạo index_2.pkl không
    """
    if not index_path.exists():
        print(f"File {index_path} không tồn tại!")
        logging.error(f"File {index_path} không tồn tại!")
        return

    try:
        with open(index_path, 'rb') as f:
            index_data = pickle.load(f)
        
        print(f"Đã load {len(index_data)} entries từ {index_path}")
        logging.info(f"Đã load {len(index_data)} entries từ {index_path}")

        # Khởi tạo counters và lists
        valid_samples = []
        invalid_samples = []
        
        stats = {
            'total': len(index_data),
            'valid': 0,
            'invalid': 0,
            'no_ligand_atoms': 0,
            'no_pocket_atoms': 0,
            'build_failed': 0,
            'file_not_found': 0,
            'parse_error': 0,
            'sanitize_error': 0,
            'few_heavy_atoms': 0,
            'too_many_others': 0  # Thêm counter cho trường hợp quá nhiều 'others'
        }

        # Progress bar
        pbar = tqdm(index_data, desc="Processing samples")

        for i, entry in enumerate(pbar):
            # Validation entry format
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                logging.warning(f"Invalid entry at index {i}: {entry}")
                invalid_samples.append((i, f"invalid_entry", "Invalid entry format"))
                stats['invalid'] += 1
                continue

            pocket_fn, ligand_fn = entry[0], entry[1]
            pdbfile = data_dir / pocket_fn
            sdffile = data_dir / ligand_fn
            sample_id = f"{pocket_fn}_{ligand_fn}"

            # Check sample
            status = check_ligand_and_pocket(pdbfile, sdffile, atom_dict, dist_cutoff)

            if status['valid']:
                valid_samples.append(entry)
                stats['valid'] += 1
                logging.info(f"Sample {i} ({sample_id}): VALID - "
                           f"lig_atoms={status['lig_atoms']}, "
                           f"pocket_atoms={status['pocket_atoms']}, "
                           f"heavy_atoms={status['ligand_heavy_atoms']}, "
                           f"pocket_residues={status['pocket_residues']}, "
                           f"others_atoms={status['others_count']}")
            else:
                invalid_samples.append((i, sample_id, status['reason']))
                stats['invalid'] += 1
                
                # Categorize errors
                reason = status['reason'].lower()
                if 'not found' in reason:
                    stats['file_not_found'] += 1
                elif 'no valid' in reason and 'ligand' in reason:
                    stats['no_ligand_atoms'] += 1
                elif 'no valid' in reason or 'no pocket' in reason:
                    stats['no_pocket_atoms'] += 1
                elif 'build' in reason:
                    stats['build_failed'] += 1
                elif 'parse' in reason:
                    stats['parse_error'] += 1
                elif 'sanitize' in reason:
                    stats['sanitize_error'] += 1
                elif 'few heavy atoms' in reason:
                    stats['few_heavy_atoms'] += 1
                elif 'too many unknown atoms' in reason:
                    stats['too_many_others'] += 1
                
                logging.warning(f"Sample {i} ({sample_id}): INVALID - {status['reason']}")

            # Update progress
            pbar.set_postfix(
                valid=stats['valid'], 
                invalid=stats['invalid'],
                valid_pct=f"{stats['valid']/stats['total']*100:.1f}%"
            )

        # Lưu valid index
        if output_valid_index and valid_samples:
            output_path = index_path.parent / 'index_2.pkl'
            with open(output_path, 'wb') as f:
                pickle.dump(valid_samples, f)
            print(f"\nĐã lưu {len(valid_samples)} valid samples vào {output_path}")
            logging.info(f"Đã lưu {len(valid_samples)} valid samples vào {output_path}")

        # Print detailed report
        print_detailed_report(stats, invalid_samples[:20])  # Chỉ show 20 đầu
        
        return valid_samples, invalid_samples, stats

    except Exception as e:
        print(f"Lỗi khi xử lý {index_path}: {str(e)}")
        logging.error(f"Lỗi khi xử lý {index_path}: {str(e)}")
        raise

def print_detailed_report(stats, sample_invalid_list):
    """In báo cáo chi tiết"""
    print("\n" + "="*60)
    print("DETAILED VALIDATION REPORT")
    print("="*60)
    
    print(f"Tổng samples: {stats['total']}")
    print(f"Valid: {stats['valid']} ({stats['valid']/stats['total']*100:.2f}%)")
    print(f"Invalid: {stats['invalid']} ({stats['invalid']/stats['total']*100:.2f}%)")
    
    print(f"\nCHI TIẾT LỖI:")
    print(f"  • File not found: {stats['file_not_found']}")
    print(f"  • Parse errors: {stats['parse_error']}")
    print(f"  • Few heavy atoms: {stats['few_heavy_atoms']}")
    print(f"  • Too many 'others': {stats['too_many_others']}")
    print(f"  • No ligand atoms: {stats['no_ligand_atoms']}")
    print(f"  • No pocket atoms: {stats['no_pocket_atoms']}")
    print(f"  • Sanitize errors: {stats['sanitize_error']}")
    print(f"  • Build failed: {stats['build_failed']}")
    
    if sample_invalid_list:
        print(f"\nMẪU INVALID SAMPLES (hiển thị {len(sample_invalid_list)} đầu tiên):")
        for i, (idx, sample_id, reason) in enumerate(sample_invalid_list):
            print(f"  {i+1:2d}. Index {idx} ({sample_id}): {reason}")
    
    print("="*60)

def main():
    # Configuration
    data_dir = Path('data/crossdocked_pocket10')
    index_path = data_dir / 'index.pkl'
    atom_dict = dataset_params['crossdock_full']['atom_encoder']
    dist_cutoff = 8.0
    
    print("CrossDock Database Validation & Filtering")
    print(f"Data directory: {data_dir}")
    print(f"Index file: {index_path}")
    print(f"Distance cutoff: {dist_cutoff}Å")
    print(f"Atom types: {len(atom_dict)} types")
    
    # Run validation
    valid_samples, invalid_samples, stats = check_crossdock_database(
        index_path, data_dir, atom_dict, dist_cutoff, output_valid_index=True
    )
    
    print(f"\nValidation completed. Check 'check_crossdock_validity.log' for details.")

if __name__ == "__main__":
    main()