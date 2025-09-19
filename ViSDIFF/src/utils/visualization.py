import torch
import numpy as np
import os
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils.molecule_builder import get_bond_order

# Các hàm hỗ trợ cần thiết
def draw_sphere(ax, x, y, z, size, color, alpha):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = size * np.outer(np.cos(u), np.sin(v))
    ys = size * np.outer(np.sin(u), np.sin(v)) * 0.8  # Correct for matplotlib
    zs = size * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x + xs, y + ys, z + zs, rstride=2, cstride=2, color=color,
                    linewidth=0, alpha=alpha)

def plot_molecule(ax, positions, atom_type, alpha, spheres_3d, hex_bg_color, dataset_info, is_pocket=False):
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    colors_dic = np.array(dataset_info['colors_dic'])
    radius_dic = np.array(dataset_info['radius_dic'])
    area_dic = 1500 * radius_dic ** 2
    areas = area_dic[atom_type]
    radii = radius_dic[atom_type]
    colors = colors_dic[atom_type]

    if spheres_3d:
        for i, j, k, s, c in zip(x, y, z, radii, colors):
            draw_sphere(ax, i.item(), j.item(), k.item(), 0.7 * s, c, alpha)
    else:
        ax.scatter(x, y, z, s=areas, alpha=0.9 * alpha, c=colors)

    if not is_pocket:
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                p1 = np.array([x[i], y[i], z[i]])
                p2 = np.array([x[j], y[j], z[j]])
                dist = np.sqrt(np.sum((p1 - p2) ** 2))
                atom1, atom2 = dataset_info['atom_decoder'][atom_type[i]], \
                               dataset_info['atom_decoder'][atom_type[j]]
                order = get_bond_order(atom1, atom2, dist)
                if order > 0:
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], linewidth=order, c='black')

def plot_data3d_combined(lig_positions, lig_atom_type, pocket_positions, pocket_atom_type, dataset_info,
                         save_path=None, spheres_3d=False, alpha=1.0, hex_bg_color='#FFFFFF'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_molecule(ax, lig_positions, lig_atom_type, alpha, spheres_3d, hex_bg_color, dataset_info, is_pocket=False)
    plot_molecule(ax, pocket_positions, pocket_atom_type, alpha * 0.5, spheres_3d, hex_bg_color, dataset_info, is_pocket=True)
    ax.set_axis_off()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close(fig)

def visualize_forward_process(lig_positions, lig_one_hot, pocket_positions, pocket_one_hot, model,
                              num_steps=10, dataset_info=None, save_dir='forward_process', spheres_3d=False):
    """
    Visualize forward diffusion process: Thêm noise vào ligand, giữ pocket fixed.
    - lig_positions: torch.Tensor (N_lig, 3) - tọa độ ligand gốc
    - lig_one_hot: torch.Tensor (N_lig, atom_nf) - one-hot encoding của ligand atoms
    - pocket_positions: torch.Tensor (N_pocket, 3) - tọa độ pocket
    - pocket_one_hot: torch.Tensor (N_pocket, residue_nf) - one-hot encoding của pocket (AA hoặc atoms)
    - model: ConditionalDDPMViSNet instance, có q_sample
    - num_steps: Số bước visualize (chia đều từ 0 đến T)
    - dataset_info: Dict chứa colors_dic, radius_dic, atom_decoder
    - save_dir: Thư mục lưu PNG và GIF
    """
    try:
        os.makedirs(save_dir)
    except OSError:
        pass

    save_paths = []
    T = model.T
    step_indices = np.linspace(0, T, num_steps, dtype=int)

    ligand = {'x': lig_positions, 'one_hot': lig_one_hot}
    pocket = {'x': pocket_positions, 'one_hot': pocket_one_hot}
    lig_mask = torch.zeros(lig_positions.size(0), dtype=torch.long, device=model.device)
    pocket_mask = torch.ones(pocket_positions.size(0), dtype=torch.long, device=model.device)

    for idx, t_val in enumerate(step_indices):
        t = torch.full((1,), t_val / T, dtype=torch.float32, device=model.device)
        xh_lig, _ = model.q_sample(ligand, pocket, t, lig_mask, pocket_mask)

        noisy_lig = xh_lig[:, :model.n_dims].cpu().numpy()
        lig_atom_type = lig_one_hot.argmax(dim=1).cpu().numpy()
        pocket_np = pocket_positions.cpu().numpy()
        pocket_atom_type = pocket_one_hot.argmax(dim=1).cpu().numpy()

        fn = os.path.join(save_dir, f'step_{t_val:03d}.png')
        plot_data3d_combined(noisy_lig, lig_atom_type, pocket_np, pocket_atom_type, dataset_info,
                             save_path=fn, spheres_3d=spheres_3d)
        save_paths.append(fn)

    imgs = [imageio.imread(fn) for fn in save_paths]
    gif_path = os.path.join(save_dir, 'forward_process.gif')
    imageio.mimsave(gif_path, imgs, subrectangles=True)
    print(f'Created GIF at {gif_path}')

def visualize_reverse_process(lig_positions, lig_one_hot, pocket_positions, pocket_one_hot, model,
                              num_steps=10, dataset_info=None, save_dir='reverse_process', spheres_3d=False):
    try:
        os.makedirs(save_dir)
    except OSError:
        pass

    save_paths = []
    T = model.T
    step_indices = np.linspace(T, 0, num_steps, dtype=int)

    ligand = {'x': lig_positions, 'one_hot': lig_one_hot}
    pocket = {'x': pocket_positions, 'one_hot': pocket_one_hot}
    lig_mask = torch.zeros(lig_positions.size(0), dtype=torch.long, device=model.device)
    pocket_mask = torch.ones(pocket_positions.size(0), dtype=torch.long, device=model.device)
    
    t = torch.full((1,), 1.0, dtype=torch.float32, device=model.device)
    xh_lig, _ = model.q_sample(ligand, pocket, t, lig_mask, pocket_mask)
    
    # Reverse process
    xh_lig_list = [xh_lig]
    for t_val in step_indices[1:]:  # Bỏ t=T vì đã có
        t = torch.full((1,), t_val / T, dtype=torch.float32, device=model.device)
        xh_lig, _ = model.p_sample(xh_lig, pocket, t, lig_mask, pocket_mask)
        xh_lig_list.append(xh_lig)

    # Visualize các bước
    for idx, (t_val, xh_lig) in enumerate(zip(step_indices, xh_lig_list)):
        lig_positions = xh_lig[:, :model.n_dims].cpu().numpy()
        lig_atom_type = lig_one_hot.argmax(dim=1).cpu().numpy()
        pocket_np = pocket_positions.cpu().numpy()
        pocket_atom_type = pocket_one_hot.argmax(dim=1).cpu().numpy()

        fn = os.path.join(save_dir, f'step_{t_val:03d}.png')
        plot_data3d_combined(lig_positions, lig_atom_type, pocket_np, pocket_atom_type, dataset_info,
                             save_path=fn, spheres_3d=spheres_3d)
        save_paths.append(fn)

    imgs = [imageio.imread(fn) for fn in save_paths]
    gif_path = os.path.join(save_dir, 'reverse_process.gif')
    imageio.mimsave(gif_path, imgs, subrectangles=True)
    print(f'Created GIF at {gif_path}')