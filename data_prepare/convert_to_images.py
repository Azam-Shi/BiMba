import pymesh #Importing pymesh here avoids library conflict (CXXABI_1.3.11)
from tqdm import tqdm
import numpy as np
import pdb
from sklearn.neighbors import KDTree
import os
from scipy import ndimage
from pathlib import Path
from Bio.PDB import PDBParser, DSSP

from data_prepare.map_patch_atom import map_patch_indices

def polar_to_cartesian(rho, theta, rotate_theta=0):
    cart_coord_x = np.zeros(rho.shape)
    cart_coord_y = np.zeros(rho.shape)
    for coord_i in range(0, rho.shape[0]):
        rho_coord = rho[coord_i]
        theta_coord = theta[coord_i]
        cart_coord_x[coord_i] = rho_coord*np.cos(theta_coord+rotate_theta)
        cart_coord_y[coord_i] = rho_coord*np.sin(theta_coord+rotate_theta)

    return cart_coord_x, cart_coord_y

def get_new_coord_patch(radius):
    new_patch_coord = []
    for i in range(0, radius*2):
        for j in range(0, radius*2):
            new_patch_coord.append((i-radius,j-radius))
    return np.array(new_patch_coord)

def compute_patch_grid(x, y, input_feat, radius, interpolate=True, stringarray=False):
    radius = int(radius)
    old_coord = np.stack((x,y), axis=-1)
    if not stringarray:
        patch_grid = np.zeros((radius*2, radius*2, input_feat.shape[1]))
    else:
        patch_grid = np.array(radius*2*[np.array(['x' for x in range(radius*2)], dtype=object)])
        patch_grid = np.expand_dims(patch_grid, axis=-1)
    for feature_i in range(0, patch_grid.shape[-1]):
        old_coord_patch = old_coord
        new_coord_patch = get_new_coord_patch(radius)
        kdt = KDTree(old_coord_patch)
        if interpolate:
            dist, indx_old = kdt.query(new_coord_patch, k=4)
        else:
            dist, indx_old = kdt.query(new_coord_patch, k=1)
        dist = np.square(dist)
        for grid_point_i in range(0, dist.shape[0]):
            x_new, y_new = new_coord_patch[grid_point_i]
            r_tmp = np.sqrt(x_new ** 2 + y_new ** 2)
            column_i = x_new + radius
            row_i = - y_new + radius -1
            if r_tmp>radius:
                patch_grid[row_i][column_i][feature_i] = 0
                continue
            if dist[grid_point_i][0]==0:
                neigh_index_i = indx_old[grid_point_i][0]
                if x_new == 0 and y_new == 0:
                    patch_grid[row_i][column_i][feature_i] = input_feat[0][feature_i]
                else:
                    patch_grid[row_i][column_i][feature_i] = input_feat[neigh_index_i][feature_i]
                continue
            dist_grid_point = dist[grid_point_i]
            result_grid_points = indx_old[grid_point_i]
            dist_to_include = []
            result_to_include = []
            for i, result_i in enumerate(result_grid_points):
                if result_i not in result_to_include:
                    result_to_include.append(result_i)
                    dist_to_include.append(dist_grid_point[i])

            if interpolate:
                total_dist = np.sum(1 / np.array(dist_to_include))
                interpolated_value = 0
                for i, result_old_i in enumerate(result_to_include):
                        interpolated_value += input_feat[result_old_i][feature_i] * (1/ dist_to_include[i])/total_dist
                patch_grid[row_i][column_i][feature_i] = interpolated_value
            else:
                try:
                    patch_grid[row_i][column_i][feature_i] = input_feat[result_grid_points[0]][feature_i]
                except IndexError:
                    patch_grid[row_i][column_i][feature_i] = 0
    return patch_grid


def read_patch(ppi, pid, ch, i, config):
    patch_dir = config['dirs']['patches'] + "/1JTD_B_A/"
    rho = np.load(patch_dir + "/" + '{}_rho_wrt_center.npy'.format(ch, str(i)), allow_pickle=True)
    patch_rho = rho[i]
    theta = np.load(patch_dir + "/" + '{}_theta_wrt_center.npy'.format(ch, str(i)), allow_pickle=True)
    patch_theta = theta[i]
    input_feat = np.load(patch_dir + "/" + '{}_input_feat.npy'.format(ch, str(i)), allow_pickle=True)
    patch_input_feat = input_feat[i]
    resnames = np.load(patch_dir + "/" + '{}/patch_{}_resnames.npy'.format(ch, i), allow_pickle=True)
    resnames = np.expand_dims(resnames, axis=1)

    # Read 3D coordinates
    coord_3d = np.load(patch_dir + "/" + '{}/{}_patch{}_coord.npy'.format(ch, ch, str(i)), allow_pickle=True)
    return patch_rho, patch_theta, patch_input_feat, resnames, coord_3d


def remove_comments(pdb_path, pdb_tmp_path):
    """
    Standartize PDB file by adding white spaces and making each line exactly 80 characters
    :param pdb_path: input PDB
    :param pdb_tmp_path: output PDB with fixed format
    :return: None
    """
    with open(pdb_path, 'r') as in_pdb:
        with open(pdb_tmp_path, 'w') as out:
            for line in in_pdb.readlines():
                if "USER" not in line:
                    newline = []
                    for i in range(80):
                        if i<len(line.strip('\n')):
                            newline.append(line[i])
                        else:
                            newline.append(' ')
                    if line[:4]=="ATOM" or line[:6]=="HETATM":
                        newline[77]=newline[13]
                    out.write(''.join(newline)+'\n')
    return None


def compute_dssp(ppi, config):
    # Compute DSSP values as described in https://biopython.org/docs/1.75/api/Bio.PDB.DSSP.html
    pid, ch1, ch2 = ppi.split('_')
    tmp_dir = config['dirs']['tmp']
    pdb_path = config['dirs']['protonated_pdb'] + '{}.pdb'.format(pid)
    pdb_tmp_path = f"{tmp_dir}/{pid}.pdb"
    remove_comments(pdb_path, pdb_tmp_path)
    parser = PDBParser(QUIET=1)
    struct = parser.get_structure(pid, pdb_tmp_path)
    model = struct[0]
    dssp = DSSP(model, pdb_tmp_path, dssp='mkdssp')
    os.remove(pdb_tmp_path)
    return dssp


def convert_dssp_to_feat(dssp, names_grid):
    """
    Convert DSSP object into grid of features
    Hydrogen bonds for each chain will be computed separate,
                    as residue from one side can form bonds with multiple residues from the other side.

    PHI PSI - IUPAC peptide backbone torsion angles
    :param dssp:
    :param names_grid:
    :return: numpy array dssp_features
    0 - Relative ASA;
    1 - NH–>O_1_relidx
    2 -
    """
    dssp_features = np.zeros((names_grid.shape[0], names_grid.shape[1], 1))

    for i in range(names_grid.shape[0]):
        for j in range(names_grid.shape[1]):
            curr_name = names_grid[i][j][0]
            if curr_name!=0:
                fields = curr_name.split(':')
                chain, resid = fields[0], fields[1]
                for key_i in dssp.keys():
                    if key_i[0]==chain and key_i[1][1] == int(resid):
                        dssp_key =key_i
                try:
                    dssp_features_i = dssp[dssp_key]
                except:
                    dssp_features[i][j][0] = 0
                    continue
                try:
                    dssp_features[i][j][0] = dssp_features_i[3]
                except:
                    dssp_features[i][j][0] = 0
    return dssp_features


def find_optimal_rotation(p1_rho, p1_theta, p2_rho, p2_theta, p1_coord_3d, p2_coord_3d, radius):
    optimal_angle = 0
    optimal_distance = np.inf
    angle_step = 6.28/100 
    curr_angle=0
    p1target_x, p1target_y = polar_to_cartesian(p1_rho, p1_theta)
    p1_coord_grid = compute_patch_grid(p1target_x, p1target_y, p1_coord_3d, radius)

    while curr_angle<6.28:
        p2_x, p2_y = polar_to_cartesian(p2_rho, p2_theta, curr_angle) # rotate only p2
        p2_coord_grid = compute_patch_grid(p2_x, p2_y, p2_coord_3d, radius)
        dist_grid = np.sqrt(np.sum(np.square(p1_coord_grid - p2_coord_grid), axis=-1))
        avg_dist = dist_grid.mean()
        if avg_dist < optimal_distance:
            optimal_angle=curr_angle
            optimal_distance = avg_dist

        curr_angle+=angle_step
    print(f"Optimal angle: {optimal_angle} radians.")
    p2_x, p2_y = polar_to_cartesian(p2_rho, p2_theta, optimal_angle)  # rotate only p2
    return (p1target_x, p1target_y), (p2_x, p2_y)

        
def convert_one_patch_to_image(ppi, ch, config):
    pid, ch = ppi.split('_') 
    my_grid_dir = config['dirs']['grid']  
    grid_dir = Path(my_grid_dir) / ppi / ch
    if not grid_dir.exists():
        grid_dir.mkdir(parents=True, exist_ok=True)

    radius = config['ppi_const']["radius"] 
    selected_indices = np.load(config['dirs']['patches'] + "/1JTD_B_A/" + ch + "_selected_patches.npy", allow_pickle=True)
    
    for i in range(len(selected_indices)):
        current_patch = selected_indices[i]   
        out_grid = grid_dir / ("patch_" + str(current_patch) + '.npy')
        out_resnames = grid_dir / ("patch_" + str(current_patch) + '_resnames')
        p1_rho, p1_theta, p1_input_feat, p1_resnames, p1_coord_3d = read_patch(ppi, pid, ch, current_patch, config)
        p1target_x, p1target_y = polar_to_cartesian(p1_rho, p1_theta)
        p1target_patch_grid = compute_patch_grid(p1target_x, p1target_y, p1_input_feat, radius)  # (r, r, n_feat)
        p1name_grid = compute_patch_grid(p1target_x, p1target_y, p1_resnames, radius, interpolate=False, stringarray=True)
        single_grid = p1target_patch_grid
        
        dssp = compute_dssp(ppi, config)
        dssp_grid_1 = convert_dssp_to_feat(dssp, p1name_grid)
        single_grid = np.concatenate([single_grid, dssp_grid_1], axis=-1)
        np.save(out_grid, single_grid)
        np.save(out_resnames, p1name_grid)
    return None


def convert_to_images(ppi_list, config):
    for ppi in tqdm(ppi_list): 
        try:
            pid, ch = ppi.split('_')
            convert_one_patch_to_image(ppi, ch, config)  
        except ValueError:
            try:
                pid, ch1, ch2 = ppi.split('_')
                convert_one_patch_to_image(pid + '_' + ch1, config)        
                convert_one_patch_to_image(pid + '_' + ch2, config)
            
            except ValueError:
                print(f"Couldn't convert to image for {ppi}.")
    return None