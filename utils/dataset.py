from torch.utils.data import Dataset
import numpy as np
import math, random
import os
import torch
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import plotly
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from collections import Counter
import meshio
from Bio.PDB import PDBParser, Selection
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import pyflann


def learn_background_mask(grid):
    mask = np.zeros((grid.shape[0], grid.shape[1]))
    radius = grid.shape[0] / 2
    for row_i in range(grid.shape[0]):
        for column_i in range(grid.shape[1]):
            x = column_i - radius
            y = radius - row_i
            if x ** 2 + y ** 2 <= radius ** 2:
                mask[row_i][column_i] = 1
    return mask

def compute_vertex_areas(mesh):
    points = mesh.points
    triangles = None

    # Find triangle faces from meshio cell block
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            triangles = cell_block.data
            break
    if triangles is None:
        raise ValueError("No triangle cells found in the mesh.")
    vertex_areas = np.zeros(len(points))

    for tri in triangles:
        p0, p1, p2 = points[tri[0]], points[tri[1]], points[tri[2]]
        # Compute area of triangle (half the norm of the cross product)
        tri_area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        # Distribute 1/3 of area to each vertex
        for idx in tri:
            vertex_areas[idx] += tri_area / 3.0
    return vertex_areas


def process_single_ppi(args):
    ppi, data_prepare_dir = args
    pid, ch = ppi.split("_")
    ply_path = os.path.join(data_prepare_dir, "05-surface_ply", f"{ppi}.ply")
    pdb_path = os.path.join(data_prepare_dir, "04-chains_pdbs", f"{ppi}.pdb")
    residues_list = []
    
    ply_mesh = meshio.read(ply_path)
    warnings.filterwarnings("ignore", category=PDBConstructionWarning)

    parser = PDBParser()
    struct = parser.get_structure(pdb_path, pdb_path)
    
    atoms = Selection.unfold_entities(struct, 'A')
    residues = Selection.unfold_entities(struct, 'R')
    print("number of residues for PPI {} is : {}".format(ppi, len(residues)))
    at_coord = np.array([at.get_coord() for at in atoms]).astype(float)

    flann = pyflann.FLANN()
    verts = ply_mesh.points.astype(float)
    vertex_areas = compute_vertex_areas(ply_mesh) 
    # extract iface column (index 6) from the point data
    iface_values = np.array(ply_mesh.point_data['iface']).astype(int)  # shape: (N,)
    
    if np.all(iface_values == 0) or np.all(iface_values == 1):
        print(f"⚠️ Error: iface values for {ppi} are all {iface_values[0]}. Skipping...")

    r,d = flann.nn(at_coord, verts)
    d = np.sqrt(d)
    n= len(r)
    
    res_areas = {}
    for ix, v in enumerate(verts): 
        res_id = atoms[r[ix]].get_parent().get_id()
        if res_id not in res_areas:
            res_areas[res_id] = []
        iface = iface_values[ix]
        res_areas[res_id].append((ix, iface))

    res_selected_patches = []
    all_res_indx = []
    for key in res_areas:
        pos_indx = []
        neg_indx = []
        for ix, iface in res_areas[key]:
            if ix is None:
                continue
            if iface == 1:
                pos_indx.append(ix)
            elif iface == 0:
                neg_indx.append(ix)

        if len(pos_indx) == len(res_areas[key]):
            print("All points are Pos: length of pos is {} and length of Key is{} ".format(len(pos_indx), len(res_areas[key])))
        elif len(pos_indx) > len(neg_indx):
            print("POS > NEG : we selected all pos and neg points. length of pos point is {} and length of neg point is {}".format(len(pos_indx), len(neg_indx)))

        if len(neg_indx) == len(res_areas[key]):
            k = 0 if not neg_indx else math.ceil(0.1 * len(neg_indx))
            neg_indx = random.sample(neg_indx, k)
        elif len(neg_indx) > len(pos_indx):
            neg_indx = random.sample(neg_indx, len(pos_indx))

        all_indx = pos_indx + neg_indx
        all_res_indx.extend(all_indx)  
    
    filtered_dir = os.path.join(data_prepare_dir, "Selected_points_list")
    if not os.path.exists(filtered_dir):
        os.makedirs(filtered_dir)
        
    np.save(os.path.join(filtered_dir, f"{ppi}_patches.npy"), all_res_indx)
    return ppi


def filter_patchs_parallel(ppi_list, data_prepare_dir):
    args = [(ppi, data_prepare_dir) for ppi in ppi_list]
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_single_ppi, args), total=len(args)))
    
class PrepDataset(Dataset):
    def __init__(self, ppi_list, training_mode, data_prepare_dir,
                 std=None, mean=None, feature_subset=None):

        self.ppi_list = ppi_list
        self.training_mode = training_mode
        self.data_prepare_dir = data_prepare_dir
        self.feature_subset = feature_subset
        self.mean = mean
        self.std = std
        self.grid_dir = os.path.join(data_prepare_dir, "07-grid")
        if training_mode:    
            self.pos_dict, self.neg_dict = self.select_pos_neg()
        else:
            self.pos_dict, self.neg_dict = self.all_pos_neg()

        count_pos = sum([len(self.pos_dict[x]) for x in self.pos_dict])
        count_neg = sum([len(self.neg_dict[x]) for x in self.neg_dict])
        print(f"Constructed dataset with {count_pos} positive and {count_neg} negative patches from {len(self.ppi_list)} proteins.")

        if self.feature_subset is not None:
            print(f"Using feature subset: {self.feature_subset}")

        sample_ppi = random.choice(list(self.pos_dict.keys()))
        sample_patch = random.choice(self.pos_dict[sample_ppi])
        sample_grid = np.load(os.path.join(self.grid_dir, sample_ppi, sample_patch + ".npy"), allow_pickle=True)
        self.background_mask = learn_background_mask(sample_grid)
        self.n_features = sample_grid.shape[-1]


    def select_pos_neg(self):
        filter_patchs_parallel(self.ppi_list, self.data_prepare_dir)
        pos_dict, neg_dict = {}, {}
        updated_ppi_list = []

        for ppi in self.ppi_list:
            pid, ch = ppi.split("_")
            patch_path = os.path.join(self.data_prepare_dir, "selected_points", f"{pid}_{ch}_patches.npy")
            extra_feats_path = os.path.join(self.data_prepare_dir, "09-external_feats", f"{pid}_{ch}_feats.npy")
            if os.path.exists(patch_path) and os.path.exists(extra_feats_path):
                selected_patches = np.load(patch_path, allow_pickle=True)
            else:
                print(f"Patch file not found for {ppi}. Skipping.")
                continue # skip this ppi if file doesn't exist
            
            labels = np.load(os.path.join(self.data_prepare_dir, "08-patch_info", f"{ppi}_iface_labels.npy"))
            
            if len(labels) > 9000:
                print(f"Skipping {ppi} due to too many patches: {len(labels)}")
                continue
            
            if (np.sum(labels) > 0.75 * len(labels) or np.sum(labels) < 30):
                print(f"Skipping {ppi} due to too many positive or too few patches: {np.sum(labels)}")
                continue # Skip if too many positive or too few patches

            pos_patches, neg_patches = [], []
            for patch in selected_patches:
                grid_path = os.path.join(self.grid_dir, ppi, f"g_{patch}.npy")
                if not os.path.exists(grid_path):
                    continue
                if labels[patch] == 1:
                    pos_patches.append(f"g_{patch}")
                else:
                    neg_patches.append(f"g_{patch}")

            min_class_size = min(len(pos_patches), len(neg_patches))
            if min_class_size < 2:
                continue  # skip this protein
            
            if len(pos_patches) > 0 and len(neg_patches) > 0:
                pos_dict[ppi] = pos_patches
                neg_dict[ppi] = neg_patches
                updated_ppi_list.append(ppi)         

        self.ppi_list = updated_ppi_list
        return pos_dict, neg_dict

    
    def all_pos_neg(self):
        filter_patchs_parallel(self.ppi_list, self.data_prepare_dir)
        pos_dict, neg_dict = {}, {}
        updated_ppi_list = []

        for ppi in self.ppi_list:
            pid, ch = ppi.split("_")
            patch_path = os.path.join(self.data_prepare_dir, "selected_points", f"{pid}_{ch}_patches.npy")
            selected_patches = np.load(patch_path, allow_pickle=True)
            extra_feats_path = os.path.join(self.data_prepare_dir, "09-external_feats", f"{pid}_{ch}_feats.npy")
            if os.path.exists(patch_path) and os.path.exists(extra_feats_path):
                selected_patches = np.load(patch_path, allow_pickle=True)
            else:
                print(f"Patch file not found for {ppi}. Skipping.")
                continue # skip this ppi if file doesn't exist
            
            labels = np.load(os.path.join(self.data_prepare_dir, "08-patch_info", f"{ppi}_iface_labels.npy"))
            if len(labels) > 9000:
                print(f"Skipping {ppi} due to too many patches: {len(labels)}")
                continue
            if (np.sum(labels) > 0.75 * len(labels) or np.sum(labels) < 30):
                print(f"Skipping {ppi} due to too many positive or too few patches: {np.sum(labels)}")
                continue # Skip if too many positive or too few patches
            
            pos_patches, neg_patches = [], []
            for patch in selected_patches:
                grid_path = os.path.join(self.grid_dir, ppi, f"g_{patch}.npy")
                if not os.path.exists(grid_path):
                    continue
                if labels[patch] == 1:
                    pos_patches.append(f"g_{patch}")
                else:
                    neg_patches.append(f"g_{patch}")         
                       
            min_class_size = min(len(pos_patches), len(neg_patches))
            if min_class_size < 2:
                continue  # skip this protein            
            if len(pos_patches) > 0 and len(neg_patches) > 0:
                pos_dict[ppi] = pos_patches
                neg_dict[ppi] = neg_patches
                updated_ppi_list.append(ppi)
        self.ppi_list = updated_ppi_list
        return pos_dict, neg_dict
    
    
    def __len__(self):
        return len(self.ppi_list)

    
    def rotate(self, grid):
        angle = np.random.randint(low=1, high=360)
        for feature_i in range(0, grid.shape[-1]):
            grid[:, :, feature_i] = ndimage.rotate(grid[:, :, feature_i], angle, reshape=False)
        return grid
    
    
    def __getitem__(self, idx):
        ppi = self.ppi_list[idx]
        pos = self.pos_dict[ppi]
        neg = self.neg_dict[ppi]
        pos_paths = [os.path.join(self.grid_dir, ppi, x + ".npy") for x in pos]
        neg_paths = [os.path.join(self.grid_dir, ppi, x + ".npy") for x in neg]
        
        if self.training_mode:
            pos_grids = [self.rotate(np.load(pos_path, allow_pickle=True)) for pos_path in pos_paths] 
            neg_grids = [self.rotate(np.load(neg_path, allow_pickle=True)) for neg_path in neg_paths]
        else:
            pos_grids = [np.load(pos_path, allow_pickle=True) for pos_path in pos_paths]
            neg_grids = [np.load(neg_path, allow_pickle=True) for neg_path in neg_paths]

        labels = np.array([1]*len(pos_grids) + [0]*len(neg_grids))
        grid = pos_grids + neg_grids
        if self.training_mode:
            combined = list(zip(grid, labels))
            random.shuffle(combined)
            grid, labels = zip(*combined)
        
        grid = np.swapaxes(np.array(grid), -1, 1).astype(np.float32)
        if self.feature_subset:
            grid = grid[:, self.feature_subset, :, :]
        if self.mean is not None and self.std is not None:
            for feature_i in range(grid.shape[1]):
                grid[:, feature_i, :, :] = (grid[:, feature_i, :, :] - self.mean[feature_i]) / self.std[feature_i]
            grid = np.logical_and(grid, self.background_mask) * grid

        pos_resname_paths = [os.path.join(self.grid_dir, ppi, x + "_rn.npy") for x in pos]
        neg_resname_paths = [os.path.join(self.grid_dir, ppi, x + "_rn.npy") for x in neg]
        all_resname_paths = pos_resname_paths + neg_resname_paths      
        
        extra_feats_path = os.path.join(self.data_prepare_dir, "09-external_feats", f"{ppi}_feats.npy")
        extra_feats = np.load(extra_feats_path, allow_pickle=True)   
        extra_feats_tensor = []
        for resname_path in all_resname_paths:
            if not os.path.exists(resname_path):
                print(f"Patch residue names file not found for {ppi}. Skipping.")
                exit(1)
            
            data = np.load(resname_path, allow_pickle=True)

            h, w, d = data.shape
            center_indices = [(h//2 - 1, w//2 - 1, 0),
                            (h//2 - 1, w//2, 0),
                            (h//2, w//2 - 1, 0),
                            (h//2, w//2, 0)]

            center_values = [data[idx] for idx in center_indices]
            processed_values = []
            for idx in center_indices:
                val = data[idx]
                if val is None or isinstance(val, int):
                    continue
                parts = val.split(":")
                processed_values.append(":".join(parts[:2]))
            counter = Counter(processed_values)
            most_common_value, count = counter.most_common(1)[0]
            chain_id, residue_num = most_common_value.split(":")
            flag = False
            
            for lines in extra_feats:                
                if lines[0] == chain_id and lines[1] == int(residue_num):
                    flag = True
                    extra_feats_tensor.append(lines[2:])
                    break
                
            if not flag:
                print(f"Warning: No matching residue found for PPI {ppi} {chain_id}:{residue_num} in extra features.")

        extra_feats_tensor = np.array(extra_feats_tensor, dtype=np.float32)   
        return grid, np.array(labels), ppi, extra_feats_tensor


class PDB_complex_testing(Dataset):
    def __init__(self, ppi_list, data_prepare_dir, device, std=None, mean=None, feature_subset=None):
        self.device = device
        self.ppi_list = ppi_list
        self.data_prepare_dir = data_prepare_dir
        self.grid_dir = os.path.join(data_prepare_dir, "07-grid")
        self.surface_ply_dir = os.path.join(data_prepare_dir, "05-surface_ply")
        self.pdb_dir = os.path.join(data_prepare_dir, "04-chains_pdbs")
        self.feature_subset = feature_subset
        self.mean = mean  
        self.std = std 
        self.grid_dict = {}
        self.all_patches_dict = {}

        for ppi in ppi_list:
            self.grid_dict[ppi], all_patches_dict = self.select_pos_neg(ppi)
            self.all_patches_dict = all_patches_dict

    def __len__(self):
        return sum(len(self.grid_dict[ppi]) for ppi in self.ppi_list)

    def select_pos_neg(self, ppi):
        pid, ch = ppi.split('_')
        grid_path = os.path.join(self.grid_dir, ppi) 
        ply_path = os.path.join(self.surface_ply_dir, f"{ppi}.ply")
        ply_mesh = meshio.read(ply_path) 
               
        parser = PDBParser()
        struct = parser.get_structure(os.path.join(self.pdb_dir, f"{ppi}.pdb"), os.path.join(self.pdb_dir, f"{ppi}.pdb"))

        atoms = Selection.unfold_entities(struct, 'A')
        residues = Selection.unfold_entities(struct, 'R')
        print("number of residues for PPI {} is : {}".format(ppi, len(residues)))
        at_coord = np.array([at.get_coord() for at in atoms]).astype(float)

        flann = pyflann.FLANN()
        verts = ply_mesh.points.astype(float)
        vertex_areas = compute_vertex_areas(ply_mesh)
        r,d = flann.nn(at_coord, verts)
        d = np.sqrt(d)
        n= len(r)
        res_areas = {res.get_id(): [] for res in residues}

        for ix, v in enumerate(verts):
            res_id = atoms[r[ix]].get_parent().get_id()
            atom_name = atoms[r[ix]].get_name()
            if atom_name in ["OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "NE", "ND1", "NE2", "NZ", "NH1", "NH2", "NE1", "SG", "SD"]:
                res_areas[res_id].append(ix)

        for ix, v in enumerate(verts):
            res_id = atoms[r[ix]].get_parent().get_id()
            atom_name = atoms[r[ix]].get_name()
            if atom_name in ["CD1", "CD2", "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2"]:
                res_areas[res_id].append(ix)
        
        for ix, v in enumerate(verts):
            res_id = atoms[r[ix]].get_parent().get_id()
            atom_name = atoms[r[ix]].get_name()
            if atom_name in ["O", "OXT", "N"]:
                res_areas[res_id].append(ix)
        
        sc_heteroatoms = {
            "OD1", "OD2", "OE1", "OE2", "OE", "OG", "OG1", "OG2", "OH",
            "NE", "NE1", "NE2", "ND1", "ND2", "NH1", "NH2", "NZ",
            "SG", "SD", "SE"
        }
        aromatic_rings = {
            "CG", "CD1", "CD2", "CE1", "CE2", "CE3",
            "CZ", "CZ2", "CZ3", "CH2"
        }
        carbonyl_amide_backbone = {"O", "OXT", "N"}

       
        for ix, v in enumerate(verts):
            res_id = atoms[r[ix]].get_parent().get_id()
            if not res_areas[res_id]: 
                atom_name = atoms[r[ix]].get_name()
                if atom_name not in sc_heteroatoms and atom_name not in aromatic_rings and atom_name not in carbonyl_amide_backbone:
                    print("Fallback to other atom:", atom_name)
                    res_areas[res_id].append(ix)

        all_patches = []
        all_patch_res_dict = {}
        count_points = 0
        for key in res_areas:
            count = 0
            for ix in res_areas[key]:
                if ix is None:
                    continue
                if key not in all_patch_res_dict:
                    all_patch_res_dict[key] = []
                all_patch_res_dict[key].append(ix)
                all_patches.append(ix)
                count += 1
            count_points += count
        print("selected {} number of points instead of {}".format(count_points, len(verts)))
        return all_patches, all_patch_res_dict



    def __getitem__(self, idx):
        ppi_idx = 0
        patch_idx = idx
        for ppi in self.ppi_list:
            extra_feats_path = os.path.join(self.data_prepare_dir, "09-external_feats", f"{ppi}_feats.npy")
            if not os.path.exists(extra_feats_path):
                print(f"External features file missing: {extra_feats_path}")
            continue  

        ppi = self.ppi_list[ppi_idx]
        pid, ch = ppi.split('_')
        patch_num = self.grid_dict[ppi][patch_idx]
        patch_file = f"g_{patch_num}.npy"
        patch_path = os.path.join(self.grid_dir, ppi, patch_file) 

        grid = np.load(patch_path, allow_pickle=True)
        grid_tensor = torch.tensor(grid, dtype=torch.float32).permute(2, 0, 1)  # [C, H, W]

        # Normalize (optional)
        if self.feature_subset:
            grid_tensor = grid_tensor[self.feature_subset, :, :]
        if self.mean is not None and self.std is not None:
            for i in range(grid_tensor.shape[0]):
                grid_tensor[i] = (grid_tensor[i] - self.mean[i]) / self.std[i]
        extra_feats = np.load(extra_feats_path, allow_pickle=True)
        base_name = patch_file.replace(".npy", "")
        rn_path = os.path.join(self.grid_dir, ppi, f"{base_name}_rn.npy") ## for masif-test

        if not os.path.exists(rn_path):
            raise FileNotFoundError(f"Residue name file missing: {rn_path}")

        data = np.load(rn_path, allow_pickle=True)
        h, w, d = data.shape
        center_indices = [(h//2 - 1, w//2 - 1, 0), (h//2 - 1, w//2, 0), (h//2, w//2 - 1, 0), (h//2, w//2, 0)]
        center_values = [data[idx] for idx in center_indices if data[idx] is not None and isinstance(data[idx], str)]
        processed_values = []
        for val in center_values:
            parts = val.split(":")
            if len(parts) >= 2:
                processed_values.append(":".join(parts[:2]))

        counter = Counter(processed_values)
        if not counter:
            raise ValueError(f"No valid residue ID found in {rn_path}")
        most_common_value, _ = counter.most_common(1)[0]
        chain_id, residue_num = most_common_value.split(":")
        matched_feat = None
        for row in extra_feats:
            if row[0] == chain_id and row[1] == int(residue_num):
                matched_feat = row[2:]
                break
        if matched_feat is None:
            raise ValueError(f"Residue {chain_id}:{residue_num} not found in {ppi} extra features.")
        extra_feats_tensor = np.array(matched_feat, dtype=np.float32)
        return grid_tensor, ppi, extra_feats_tensor


class PISToN_dataset(Dataset):
    def __init__(self, ppi_list, attn=None):
        """
        Args:
            ppi_list (list): A list of full paths to folders containing g_*.npy grid files.
        """
        self.ppi_list = ppi_list
        self.ppi_to_idx = {}
        self.grid = []
        self.grid_dir_list = []
        for i, ppi_path in enumerate(ppi_list):
            print(f"  [{i+1}/{len(ppi_list)}] {ppi_path}")

            if not os.path.isdir(ppi_path):
                print(f"Skipping: {ppi_path} is not a directory.")
                continue
            patch_files = sorted([
                f for f in os.listdir(ppi_path)
                if f.startswith("g_") and f.endswith(".npy") and not f.endswith("_rn.npy")
            ])

            if not patch_files:
                print(f"No valid g_*.npy files found in {ppi_path}")
                continue
            for pf in patch_files:
                patch_path = os.path.join(ppi_path, pf)
                patch_array = np.load(patch_path, allow_pickle=True)
                self.grid.append(patch_array)
                self.grid_dir_list.append(ppi_path)
            ppi_name = os.path.basename(os.path.dirname(ppi_path))
            self.ppi_to_idx[ppi_name] = i

        if not self.grid:
            raise ValueError("No patch data loaded. Check your input directories.")

        self.grid = np.stack(self.grid, axis=0)  # Shape: [N, H, W, C]
        self.grid = np.swapaxes(self.grid, -1, 1).astype(np.float32)  # Shape: [N, C, H, W]
        background_mask = learn_background_mask(self.grid[0].transpose(1, 2, 0))
        for i in range(self.grid.shape[1]):  # For each channel
            self.grid[:, i] = np.logical_and(self.grid[:, i], background_mask) * self.grid[:, i]

    def vis_patch(self, patch_name, ppi_path, html_path=None, attn=None):
        feature_pairs = {
            'shape_index': (0,),
            'ddc': (1,),
            'electrostatics': (2,),
            'charge': (3,),
            'hydrophobicity': (4,)
        }
        resnames_path = f"{ppi_path}/{patch_name}_rn.npy"
        patch_path = f"{ppi_path}/{patch_name}.npy"
        patch_np = np.load(patch_path, allow_pickle=True)
        patch_resnames = np.load(resnames_path, allow_pickle=True)
        n_feat = len(feature_pairs)
        key_names = list(feature_pairs.keys())
        fig = make_subplots(rows=1, cols=n_feat, subplot_titles=key_names, horizontal_spacing=0.02)

        for col_i, feature_name in enumerate(key_names):
            pair_i = feature_pairs[feature_name][0]
            patch_i = patch_np[:, :, pair_i]  # 2D feature map
            if attn is not None:
                mask = (attn > 0) * attn
                patch_i = patch_i * mask

            fig.add_trace(go.Heatmap(
                z=patch_i,
                customdata=patch_resnames,
                hovertemplate='<b>Value:%{z:.3f}</b><br>Amino Acid:%{customdata[0]}',
                name='',
                colorscale='RdBu',
                zmid=0,
                showscale=(col_i == n_feat - 1),
                colorbar=dict(title='.') if col_i == n_feat - 1 else None,
            ), row=1, col=col_i + 1)
            fig.update_yaxes(scaleanchor=f"x{col_i+1}", tickfont=dict(size=10), row=1, col=col_i + 1)
            fig.update_xaxes(tickfont=dict(size=10), row=1, col=col_i + 1)

        fig.update_layout(
            title_text=f'The interactive patch pair for {patch_name}. Hover to see the value and corresponding amino acid name.',
            height=350,
            margin=dict(t=120)
        )
        if html_path is not None:
            plotly.offline.plot(fig, filename=html_path)
        else:
            fig.show()
   
    def __len__(self):
        return self.grid.shape[0]

    def __getitem__(self, idx):
        return self.grid[idx]
    