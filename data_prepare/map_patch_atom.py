import numpy as np
import pandas as pd
from Bio.PDB import *
from scipy.spatial import cKDTree
import pdb
from Bio import Entrez,SeqIO, BiopythonWarning
import warnings
import os
from pathlib import Path

warnings.simplefilter('ignore', BiopythonWarning)

def get_start_res(resid, chain_id):
    chain_curr = ''
    start_res = []
    start_res_curr = resid[0]
    for i, res_i in enumerate(resid):
        if chain_id[i]!=chain_curr:
            start_res_curr=res_i
            chain_curr=chain_id[i]
        start_res.append(start_res_curr)
    return np.array(start_res)
  
    
def map_patch_indices(ppi, pid, ch, config):
    selected_indices = np.load(config['dirs']['patches'] + "/" + ppi + "/" + ch + "_selected_patches.npy", allow_pickle=True)
    all_list_indices = np.load(config['dirs']['patches'] + "/" + ppi + "/" + ch + '_list_indices.npy', allow_pickle=True)
    for i in range(len(selected_indices)):
        current_patch = selected_indices[i]
        mapping_table = config['dirs']['patches'] + "/" + ppi + "/" + ch  + "/patch_" + str(current_patch) + "_map.csv"
        mapping_df = pd.read_csv(mapping_table)
        indices_np = np.array(all_list_indices[current_patch])
        my_grid_dir = config['dirs']['patches']  
        grid_dir = Path(my_grid_dir) / ppi / ch
        if not grid_dir.exists():
            grid_dir.mkdir(parents=True, exist_ok=True)
        out_map = grid_dir / ("patch_" + str(current_patch) + '_resnames')
        res_names = np.array(['x' for i in range(len(indices_np))], dtype=object)
        
        for i, patch_i in enumerate(indices_np):
            tmp_df = mapping_df.iloc[[i]].reset_index(drop=True)
            if not tmp_df.empty:
                res_name_i = '{}:{}:{}-{}:{}'.format(tmp_df.loc[0]['chain_id'],tmp_df.loc[0]['res_ind'],
                                                tmp_df.loc[0]['residue_name'],tmp_df.loc[0]['atom_ind'],
                                                    tmp_df.loc[0]['atom_name']) #chain:resid:res_name-atom_id:atom_name
                res_names[i] = res_name_i
        np.save(out_map, res_names)


def map_patch_atom(ppi_list, config):
    print("Mapping each patch to a residue number...")
    for ppi in ppi_list:
        try:
            pid, ch1 = ppi.split('_')
            map_patch_atom_one(ppi, pid, ch1, config)
            map_patch_indices(ppi, pid, ch1, config)
        except ValueError:
            try:
                pid, ch1, ch2 = ppi.split('_')
                map_patch_atom_one(ppi, pid, ch1, config)
                map_patch_atom_one(ppi, pid, ch2, config)
                
                map_patch_indices(ppi, pid, ch1, config)
                map_patch_indices(ppi, pid, ch2, config)
            except ValueError:
                print(f"Skipping {ppi}: Unexpected format for mapping patches to atoms.")


def map_patch_atom_one(ppi, pid, ch, config):
    pdb_id = pid
    chain_name = ch
    patch_dir = config['dirs']['patches'] 
    pdb_chain_dir = config['dirs']['chains_pdb']
    out_mappings_dir = config['dirs']['patches'] + ppi + '/'
    iface_labels = config['dirs']['patches'] + '/{}/{}_iface_labels.npy'.format(ppi, ch)
    selected_indices = np.load(config['dirs']['patches'] + "/" + ppi + "/" + ch + "_selected_patches.npy", allow_pickle=True)
    all_list_indices = np.load(config['dirs']['patches'] + "/" + ppi + "/" + ch + '_list_indices.npy', allow_pickle=True)

    for i in range(len(selected_indices)):
        current_patch = selected_indices[i]
        out_table = patch_dir + "/" + ppi + "/" + chain_name  + "/patch_" + str(current_patch) + "_map.csv"
        if not os.path.exists(out_table):
            os.makedirs(os.path.dirname(out_table), exist_ok=True)
            with open(out_table, 'w') as f:
                pass

        # Read coordinates of
        patch_data = np.load(patch_dir+"/{}/{}/{}_patch{}_coord.npy".format(ppi, chain_name, chain_name, current_patch))
        x_coord = patch_data[:, 0]
        y_coord = patch_data[:, 1]
        z_coord = patch_data[:, 2]
        patch_coord = np.column_stack((x_coord,y_coord,z_coord))

        # Read interface
        iface_labels = np.load(patch_dir+"/{}/{}_iface_labels.npy".format(ppi, chain_name, i-1), allow_pickle=True)

        # Read PDB structure
        pdb_path = "{}/{}_{}.pdb".format(pdb_chain_dir, pdb_id, chain_name)
        parser = PDBParser()
        pdb_struct = parser.get_structure('{}_{}'.format(pdb_id, chain_name), pdb_path)

        ## Get heavy atoms
        heavy_atoms=[]
        heavy_orig_map = {}
        k=0
        for i, atom in enumerate(pdb_struct.get_atoms()):
            tags = atom.parent.get_full_id()
            if atom.element!='H' and tags[3][0]==' ': # if heavy atom and not heteroatom
                heavy_orig_map[k]=i #map heavy atom index to original pdb index
                heavy_atoms.append(atom)
                k+=1
        atom_coord = np.array([list(atom.get_coord()) for atom in heavy_atoms])
        atom_names = np.array([atom.get_id() for atom in heavy_atoms])
        residue_id = np.array([atom.parent.id[1] for atom in heavy_atoms])
        residue_name = np.array([atom.parent.resname for atom in heavy_atoms])
        chain_id = np.array([atom.get_parent().get_parent().get_id() for atom in heavy_atoms])

        # get start residue
        start_res = get_start_res(residue_id, chain_id)
        pdb_tree = cKDTree(atom_coord)
        dist, idx = pdb_tree.query(patch_coord) #idx is the index of pdb heavy atoms that close to every patch from [0 to N patches]
        result_pdb_idx=[]
        for i in idx:
            result_pdb_idx.append(heavy_orig_map[i])
        result_pdb_idx = np.array(result_pdb_idx) #index in original pdb
        df = pd.DataFrame({"patch_ind":range(0, len(result_pdb_idx)),
                        "atom_ind":result_pdb_idx,
                        "res_ind": residue_id[idx],
                        "atom_name":atom_names[idx],
                        "residue_name":residue_name[idx],
                        "chain_id":chain_id[idx]
                        })
        df.to_csv(out_table, index=False)