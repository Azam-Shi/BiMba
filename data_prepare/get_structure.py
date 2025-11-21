import pymesh #Importing pymesh here avoids library conflict (CXXABI_1.3.11)
from Bio.PDB import *
from subprocess import Popen, PIPE
from masif.source.input_output.protonate import protonate
import os
import pdb
from scipy.spatial import cKDTree
import numpy as np
import time
from utils.utils import get_date, extract_pdb_chain
from tqdm import tqdm


def protonate_pdb(ppi, config):
    """
    downlaod and add hydrogens to PPI
    """
    pid = ppi.split('_')[0]

    # Download pdb
    pdb_filename = config['dirs']['raw_pdb'] + pid + '.pdb'
    if not os.path.exists(pdb_filename):
        pdbl = PDBList()
        pdb_filename = pdbl.retrieve_pdb_file(pid, pdir=config['dirs']['raw_pdb'], file_format='pdb')
    # Protonate downloaded file
    protonated_file = config['dirs']['protonated_pdb']+"/"+pid+".pdb"
    protonate(pdb_filename, protonated_file)

def download(ppi_list, config, to_write=None):
    start = time.time()
    print("**** [ {} ] Start Downloading PDBs...".format(get_date()))
    print(ppi_list)
    processed_ppi = []
    for i in tqdm(range(len(ppi_list))):
        ppi = ppi_list[i]
        pid = ppi.split('_')[0]
        raw_pdb_filename = config['dirs']['protonated_pdb']+"/"+pid+".pdb"
        if not os.path.exists(raw_pdb_filename):
            protonate_pdb(ppi, config)
        else:
            print("PDB file {} already exists. Skipping...".format(pid))
        if os.path.exists(raw_pdb_filename):
            processed_ppi.append(ppi)
    if to_write is not None:
        with open(to_write, 'w') as out:
            for ppi in processed_ppi:
                out.write(ppi+'\n')
    print("**** [ {} ] Done with downloading PDBs.".format(get_date()))
    print("**** [ {} ] Took {:.2f}min.".format(get_date(), (time.time()-start)/60))
    return processed_ppi


def select_single_model(pdb_path, pdb_path_updated):
    with open(pdb_path_updated, 'w') as out:
        with open(pdb_path, 'r') as f:
            for line in f.readlines():
                if line[:5]=="MODEL" or line[:6]=="REMARK":
                    pass
                elif line[:6]=="ENDMDL":
                    break
                else:
                    out.write(line)

def get_coord_dict(pid, pdb_path, chain):
    parser = PDBParser(QUIET=True)
    try:
        pdb_struct = parser.get_structure(pid, pdb_path)
    except ValueError: # tbe PDB file contain multiple models
        pdb_path_updated = pdb_path.replace('.pdb','') + '_singleModel.pdb'
        select_single_model(pdb_path, pdb_path_updated)
        pdb_struct = parser.get_structure(pid, pdb_path_updated)
    RES_dict = {'atom_id': [], 'res_id': [], 'chain_id': [], 'atom_coord': []}

    all_atom_res_chain_pairs = []
    for i, atom in enumerate(pdb_struct.get_atoms()):
        res_id = atom.parent.id[1]
        chain_id = atom.get_parent().get_parent().get_id()
        atom_coord = list(atom.get_coord())
        atom_id = atom.serial_number
        if chain_id in chain:
            if (atom_id, res_id, chain_id) not in all_atom_res_chain_pairs:
                all_atom_res_chain_pairs.append((atom_id, res_id, chain_id))
                RES_dict['atom_id'].append(atom_id)
                RES_dict['res_id'].append(res_id)
                RES_dict['chain_id'].append(chain_id)
                RES_dict['atom_coord'].append(atom_coord)
    return RES_dict


def extract_pdb_one(ppi, config, use_refined=True):
    try:
        pid, ch = ppi.split('_')
        print(f"Loading original PDB...")
        pdb_file = f"{config['dirs']['protonated_pdb']}/{pid}.pdb"
        extract_pdb_chain(pdb_file, config['dirs']['chains_pdb'] + '/{}_{}.pdb'.format(pid, ch), ch)
    except ValueError:
        try:
            pid, ch1, ch2 = ppi.split('_')
            print(f"Loading original PDB...")
            pdb_file = f"{config['dirs']['protonated_pdb']}/{pid}.pdb"
            extract_pdb_chain(pdb_file, config['dirs']['chains_pdb'] + '/{}_{}.pdb'.format(pid, ch1), ch1)
            extract_pdb_chain(pdb_file, config['dirs']['chains_pdb'] + '/{}_{}.pdb'.format(pid, ch2), ch2)
        except ValueError:
            print("Cannot load PDB for {}".format(ppi))
        