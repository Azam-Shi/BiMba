from datetime import datetime
from importlib.machinery import SourceFileLoader
import os
from subprocess import Popen, PIPE
import shutil
import numpy as np

def get_date():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def read_config(args):
    if not args.config:
        from config_default import config
    else:
        config_module = SourceFileLoader("config", args.config).load_module()
        config = config_module.config

    print("[ {} ] Configuration parameters:".format(get_date()))
    print(config)
    return config

def get_processed(ppi_list, config):
    processed_ppis = []
    for ppi in ppi_list:
        try:
            pid, ch = ppi.split('_')
            if os.path.isfile(config['dirs']['patch_info'] + f"{pid}_{ch}_iface_labels.npy"):
                processed_ppis.append(pid + '_' + ch)
        except:
            pid, ch1, ch2 = ppi.split('_')
            if os.path.isfile(config['dirs']['patch_info'] + f"{pid}_{ch1}_iface_labels.npy"):
                processed_ppis.append(pid + '_' + ch1)
            if os.path.isfile(config['dirs']['patch_info'] + '/' + f"{pid}_{ch2}_iface_labels.npy"):
                processed_ppis.append(pid + '_' + ch2)
    return processed_ppis


def learn_background_mask(grid):
    """
    Returns the mask with zero elements outside the patch
    :param grid: example of a grid image
    :return: mask
    """
    mask = np.zeros((grid.shape[0], grid.shape[1]))
    radius = grid.shape[0]/2
    for row_i in range(grid.shape[0]):
        for column_i in range(grid.shape[1]):
            x = column_i - radius
            y = radius - row_i
            if x ** 2 + y ** 2 <= radius ** 2:
                mask[row_i][column_i] = 1
    return mask


def fix_residue_numbers(ppi, config):
    pid, ch1, ch2 = ppi.split('_')
    pdb_file = config['dirs']['protonated_pdb']+"/"+pid+".pdb"
    pdb_tmp_file = config['dirs']['protonated_pdb']+"/"+pid+"_tmp.pdb"
    shutil.copyfile(pdb_file, pdb_tmp_file)
    prev_resid=''
    prev_resname=''
    rename_flag=False
    all_latters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    letter_i = 0
    with open(pdb_file, 'w') as out:
        with open(pdb_tmp_file, 'r') as f:
            for line in f.readlines():
                if line[:4]=='ATOM':
                    curr_resname = line[17:20]
                    curr_resid = line[22:26]
                    if curr_resid==prev_resid and curr_resname!=prev_resname:
                        rename_flag=True
                        letter_i+=1
                    if curr_resid!=prev_resid:
                        rename_flag=False
                        letter_i=-1
                    if rename_flag:
                        line_list = [ch for ch in line]
                        line_list[26] = all_latters[letter_i]
                        # shift right the rest
                        for i in range(len(curr_resid)):
                            line_list[25-i] = curr_resid[-i-1]
                        line = ''.join(line_list)
                    prev_resid = curr_resid
                    prev_resname = curr_resname
                out.write(line)
    return

def merge_chains(pdb_in, ch1, ch2, pdb_out):
    with open(pdb_in, 'r') as f:
        with open(pdb_out, 'w') as out:
            for line in f.readlines():
                if line[:6] == 'HEADER':
                    continue
                if line[:4] == 'ATOM' or line[:6] == 'HETATM':
                    line = [char for char in line]
                    if line[21] in ch1:
                        line[21] = 'Z'
                    elif line[21] in ch2:
                        line[21] = 'A'
                    line = ''.join(line)
                out.write(line)
    return None

def rename_chains(pid, ch, chains_pdb_dir, reversed=True):
    chains_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    all_chains = chains_choices[:]
    if reversed:
        all_chains.reverse()
    PDB_TARGET = '{}_{}.pdb'.format(pid, ch)
    pdb_target_path = chains_pdb_dir + PDB_TARGET
    new_chains = []
    chains_seen = []
    with open('./' + PDB_TARGET, 'w') as out:
        with open(pdb_target_path, 'r') as f:
            for line in f.readlines():
                if line[:6] == 'HEADER':
                    continue
                if line[:4] == 'ATOM' or line[:6] == 'HETATM':
                    line = [char for char in line]
                    if line[21] not in chains_seen:
                        new_chains.append(all_chains.pop())
                        chains_seen.append(line[21])
                    line[21] = new_chains[-1]
                    line = ''.join(line)
                out.write(line)
    return './' + PDB_TARGET, ''.join(new_chains)


def extract_model(pdb_file, out_pdb, i):
    to_write=False
    with open(pdb_file, 'r') as f:
        with open(out_pdb, 'w') as out:
            for line in f.readlines():
                if line[:6]=='ENDMDL':
                    to_write=False
                if to_write:
                    out.write(line)
                if line[:5]=='MODEL':
                    if line.split(' ')[-1].strip('\n') == str(i):
                        to_write = True
                    else:
                        to_write = False

def reset_config(config, new_dir):
    config['dirs']['data_prepare'] = new_dir
    for dir_key in config['dirs'].keys():
        if dir_key not in ['data_prepare', 'savedModels']:
            old_dir = config['dirs'][dir_key]
            base_dir = old_dir.split('/')[-1] if old_dir[-1]!='/' else old_dir.split('/')[-2]
            config['dirs'][dir_key] = new_dir + '/' + base_dir + '/'

    for dir in config['dirs'].values():
        if not os.path.exists(dir):
            os.makedirs(dir)
    return config


def fill_opacity(ppi, config):
    pid, ch1, ch2 = ppi.split('_')
    with open(config['dirs']['protonated_pdb']+pid+'.pdb', 'w') as out:
        with open(config['dirs']['protonated_pdb']+pid+'_tmp.pdb', 'r') as f:
            for line in f.readlines():
                if line[:4] == 'ATOM' or line[:6] == 'HETATM':
                    line = line[:55] + ' 1.00' + line[60:] + '\n'
                out.write(line)
    os.remove(config['dirs']['protonated_pdb']+pid+'_tmp.pdb')


def extract_pdb_chain(pdb_full_file, pdb_chain_file, ch):
    with open(pdb_full_file, 'r') as f:
        with open(pdb_chain_file, 'w') as out:
            for line in f.readlines():
                if (line[0:4]=='ATOM' or line[0:6]=='HETATM') and line[21] in ch:
                    out.write(line)


def combine_pdb(pdb1, pdb2, out_pdb, pdb_dir):
    with open(pdb_dir+out_pdb, 'w') as out:
        for pdb_file in [pdb1, pdb2]:
            with open(pdb_dir+pdb_file, 'r') as f1:
                for line in f1.readlines():
                    line = line.strip('\n').strip('new').strip(' ')
                    out.write(line+'\n')