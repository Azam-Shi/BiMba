import pymesh
import os
import sys
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
from IPython.core.debugger import set_trace
from sklearn.neighbors import KDTree
import random
import warnings 
from data_prepare.read_data_from_surface import read_data_from_surface
with warnings.catch_warnings(): 
    warnings.filterwarnings("ignore", category=FutureWarning)


def get_iface_verticies(mesh):
    iface = mesh.get_attribute('vertex_iface')
    vertices = mesh.vertices
    iface_indx = np.where(iface>0)
    if len(iface_indx[0])==0:
        print('WARNING:: No interface found!')
        iface_indx = np.where(iface==0)
    return vertices[iface_indx]


def compute_patch_center(mesh, radius):
    iface_vert = get_iface_verticies(mesh)
    center_point = np.mean(iface_vert, axis=0)
    kdt = KDTree(mesh.vertices)
    d, indx_cent = kdt.query(np.expand_dims(center_point, axis=0))
    return center_point, indx_cent[0][0]

def save_precompute(ppi, pid, patch_number, config, patch_coord):
    out_patch_dir = config['dirs']['patches']    
    my_precomp_dir = Path(out_patch_dir) / ppi / pid
    if not my_precomp_dir.exists():
        my_precomp_dir.mkdir(parents=True, exist_ok=True) 
    np.save(my_precomp_dir / f"{pid}_patch{patch_number}_coord.npy", patch_coord)
    np.save(my_precomp_dir / f"{pid}_patch{patch_number}_indx_c.npy", patch_number)


def compute_one_patch(ppi, config):
    """
    Perform precomputation for masif site, ppi search, or ligand.
    """
    print("{}".format(ppi))
    out_patch_dir = config['dirs']['patches']
    my_precomp_dir = Path(out_patch_dir) / ppi
    if not my_precomp_dir.exists():
        my_precomp_dir.mkdir(parents=True, exist_ok=True) 
    chains = ppi.split('_')
    ply_file = {}
    ply_file[f"{chains[1]}"] = f"{config['dirs']['surface_ply']}{chains[0]}_{chains[1]}.ply"
    if len(chains) > 2 and chains[2]:
        ply_file[f"{chains[2]}"] = f"{config['dirs']['surface_ply']}{chains[0]}_{chains[2]}.ply"
        pids = [f"{chains[1]}", f"{chains[2]}"]
    else:
        pids = [f"{chains[1]}"]
    for pid in pids:
        rho, neigh_indices, mask, input_feat, theta, iface_labels, verts = {}, {}, {}, {}, {}, {}, {}
        input_feat, rho, theta, mask, neigh_indices, iface_labels, verts = read_data_from_surface(ply_file[pid], config)
        np.save(my_precomp_dir / f"{pid}_mask", mask)
        np.save(my_precomp_dir / f"{pid}_list_indices", neigh_indices)
        np.save(my_precomp_dir / f"{pid}_iface_labels", iface_labels)
        np.save(my_precomp_dir / f"{pid}_input_feat", input_feat)
        np.save(my_precomp_dir / f"{pid}_theta_wrt_center", theta)
        np.save(my_precomp_dir / f"{pid}_rho_wrt_center", rho)
        np.save(my_precomp_dir / f"{pid}_X_all.npy", verts[:, 0])
        np.save(my_precomp_dir / f"{pid}_Y_all.npy", verts[:, 1])
        np.save(my_precomp_dir / f"{pid}_Z_all.npy", verts[:, 2])
        selected_indices = list(range(len(iface_labels)))
        random.shuffle(selected_indices)
        np.save(my_precomp_dir / f"{pid}_selected_patches.npy", np.array(selected_indices))
        for i in selected_indices: 
            L = verts[neigh_indices[i]]
            patch_coord = np.zeros((L.shape[0], 3))
            patch_coord = L
            save_precompute(ppi, pid, i, config, patch_coord)
        print(f"Precomputation for {ppi} completed.")
