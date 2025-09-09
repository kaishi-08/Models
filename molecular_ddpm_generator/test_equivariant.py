import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models.vis_dynamics import ViSNetDynamics
from torch_geometric.data import Data

# Hàm tạo ma trận quay ngẫu nhiên 3x3
def random_rotation_matrix():
    theta = torch.rand(3, dtype=torch.float64) * 2 * np.pi  # Use float64
    cos, sin = torch.cos(theta), torch.sin(theta)
    Rx = torch.tensor([[1, 0, 0], [0, cos[0], -sin[0]], [0, sin[0], cos[0]]], dtype=torch.float64)
    Ry = torch.tensor([[cos[1], 0, sin[1]], [0, 1, 0], [-sin[1], 0, cos[1]]], dtype=torch.float64)
    Rz = torch.tensor([[cos[2], -sin[2], 0], [sin[2], cos[2], 0], [0, 0, 1]], dtype=torch.float64)
    return Rz @ Ry @ Rx

# Hàm kiểm tra tính equivariant
def test_equivariance(model, xh_atoms, xh_residues, t, mask_atoms, mask_residues, tolerance=1e-4):
    model.eval()
    device = xh_atoms.device
    n_atoms = xh_atoms.size(0)
    n_residues = xh_residues.size(0)

    # Tách tọa độ và đặc trưng
    coords_atoms = xh_atoms[:, :3]
    feats_atoms = xh_atoms[:, 3:]
    coords_residues = xh_residues[:, :3]
    feats_residues = xh_residues[:, 3:]

    # Chạy mô hình với đầu vào gốc
    with torch.no_grad():
        atoms_noise_orig, residues_noise_orig, _ = model(
            xh_atoms, xh_residues, t, mask_atoms, mask_residues
        )
    coords_noise_atoms_orig = atoms_noise_orig[:, :3]
    feats_noise_atoms_orig = atoms_noise_orig[:, 3:]
    coords_noise_residues_orig = residues_noise_orig[:, :3]
    feats_noise_residues_orig = residues_noise_orig[:, 3:]

    # === Kiểm tra phép quay ===
    R = random_rotation_matrix().to(device).to(torch.float64)  # Ensure float64
    coords_atoms_rot = coords_atoms @ R
    coords_residues_rot = coords_residues @ R
    xh_atoms_rot = torch.cat([coords_atoms_rot, feats_atoms], dim=-1)
    xh_residues_rot = torch.cat([coords_residues_rot, feats_residues], dim=-1)

    with torch.no_grad():
        atoms_noise_rot, residues_noise_rot, _ = model(
            xh_atoms_rot, xh_residues_rot, t, mask_atoms, mask_residues
        )
    coords_noise_atoms_rot = atoms_noise_rot[:, :3]
    feats_noise_atoms_rot = atoms_noise_rot[:, 3:]
    coords_noise_residues_rot = residues_noise_rot[:, :3]
    feats_noise_residues_rot = residues_noise_rot[:, 3:]

    # Kiểm tra tính đồng biến của tọa độ
    coords_noise_atoms_expected = coords_noise_atoms_orig @ R
    coords_noise_residues_expected = coords_noise_residues_orig @ R
    coords_error_atoms = torch.norm(coords_noise_atoms_rot - coords_noise_atoms_expected).item()
    coords_error_residues = torch.norm(coords_noise_residues_rot - coords_noise_residues_expected).item()

    # Kiểm tra tính bất biến của đặc trưng
    feats_error_atoms = torch.norm(feats_noise_atoms_rot - feats_noise_atoms_orig).item()
    feats_error_residues = torch.norm(feats_noise_residues_rot - feats_noise_residues_orig).item()

    print(f"Rotation test - Coordinate error (atoms): {coords_error_atoms:.2e}")
    print(f"Rotation test - Coordinate error (residues): {coords_error_residues:.2e}")
    print(f"Rotation test - Feature error (atoms): {feats_error_atoms:.2e}")
    print(f"Rotation test - Feature error (residues): {feats_error_residues:.2e}")

    # === Kiểm tra phép tịnh tiến ===
    translation = torch.randn(1, 3, dtype=torch.float64).to(device)  # Use float64
    coords_atoms_trans = coords_atoms + translation
    coords_residues_trans = coords_residues + translation
    xh_atoms_trans = torch.cat([coords_atoms_trans, feats_atoms], dim=-1)
    xh_residues_trans = torch.cat([coords_residues_trans, feats_residues], dim=-1)

    with torch.no_grad():
        atoms_noise_trans, residues_noise_trans, _ = model(
            xh_atoms_trans, xh_residues_trans, t, mask_atoms, mask_residues
        )
    coords_noise_atoms_trans = atoms_noise_trans[:, :3]
    feats_noise_atoms_trans = atoms_noise_trans[:, 3:]
    coords_noise_residues_trans = residues_noise_trans[:, :3]
    feats_noise_residues_trans = residues_noise_trans[:, 3:]

    # Kiểm tra tính bất biến của tọa độ (sau khi loại bỏ trung tâm khối lượng)
    coords_error_atoms_trans = torch.norm(coords_noise_atoms_trans - coords_noise_atoms_orig).item()
    coords_error_residues_trans = torch.norm(coords_noise_residues_trans - coords_noise_residues_orig).item()
    feats_error_atoms_trans = torch.norm(feats_noise_atoms_trans - feats_noise_atoms_orig).item()
    feats_error_residues_trans = torch.norm(feats_noise_residues_trans - feats_noise_residues_orig).item()

    print(f"Translation test - Coordinate error (atoms): {coords_error_atoms_trans:.2e}")
    print(f"Translation test - Coordinate error (residues): {coords_error_residues_trans:.2e}")
    print(f"Translation test - Feature error (atoms): {feats_error_atoms_trans:.2e}")
    print(f"Translation test - Feature error (residues): {feats_error_residues_trans:.2e}")

    # === Kiểm tra pocket cố định ===
    if not model.update_pocket_coords:
        residues_coords_error = torch.norm(coords_noise_residues_orig).item()
        print(f"Fixed pocket test - Residue coordinate noise: {residues_coords_error:.2e}")
        assert residues_coords_error < tolerance, "Fixed pocket coordinates should have zero noise"

    # Kiểm tra tổng quát
    assert coords_error_atoms < tolerance, "Coordinate output (atoms) is not rotation equivariant"
    assert coords_error_residues < tolerance, "Coordinate output (residues) is not rotation equivariant"
    assert feats_error_atoms < tolerance, "Feature output (atoms) is not rotation invariant"
    assert feats_error_residues < tolerance, "Feature output (residues) is not rotation invariant"
    assert coords_error_atoms_trans < tolerance, "Coordinate output (atoms) is not translation invariant"
    assert coords_error_residues_trans < tolerance, "Coordinate output (residues) is not translation invariant"
    assert feats_error_atoms_trans < tolerance, "Feature output (atoms) is not translation invariant"
    assert feats_error_residues_trans < tolerance, "Feature output (residues) is not translation invariant"

    print("All equivariance tests passed!")

# Tạo dữ liệu giả lập
def create_dummy_data(atom_nf, residue_nf, n_dims, n_atoms=10, n_residues=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xh_atoms = torch.randn(n_atoms, n_dims + atom_nf, dtype=torch.float64).to(device)  # Use float64
    xh_residues = torch.randn(n_residues, n_dims + residue_nf, dtype=torch.float64).to(device)  # Use float64
    t = torch.tensor([0.5], dtype=torch.float64).to(device)  # Use float64
    mask_atoms = torch.zeros(n_atoms, dtype=torch.long).to(device)
    mask_residues = torch.zeros(n_residues, dtype=torch.long).to(device)
    return xh_atoms, xh_residues, t, mask_atoms, mask_residues

# Khởi tạo mô hình
atom_nf = 5
residue_nf = 3
n_dims = 3
model = ViSNetDynamics(
    atom_nf=atom_nf,
    residue_nf=residue_nf,
    n_dims=n_dims,
    hidden_nf=64,
    condition_time=True,
    update_pocket_coords=False,
    edge_cutoff_ligand=5.0,
    edge_cutoff_pocket=8.0,
    edge_cutoff_interaction=5.0,
    lmax=2,
    vecnorm_type='none',
    trainable_vecnorm=True,
    num_heads=4,
    num_layers=2,
    num_rbf=16,
    rbf_type="expnorm",
    trainable_rbf=False,
    activation="silu",
    attn_activation="silu",
    max_num_neighbors=32,
    vertex_type="Edge"
).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).to(torch.float64)  # Convert model to float64

# Tạo dữ liệu và chạy kiểm tra
xh_atoms, xh_residues, t, mask_atoms, mask_residues = create_dummy_data(atom_nf, residue_nf, n_dims)
test_equivariance(model, xh_atoms, xh_residues, t, mask_atoms, mask_residues, tolerance=1e-4)