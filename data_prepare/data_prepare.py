from .get_structure import protonate_pdb
from .get_structure import download
from .triangulate import triangulate_single
from .compute_patches import compute_one_patch
from .convert_to_images import convert_to_images
from .map_patch_atom import map_patch_atom
from importlib.machinery import SourceFileLoader
from datetime import datetime
import time
from utils.utils import read_config, get_date, extract_model, reset_config, fill_opacity
from utils.utils import combine_pdb, merge_chains, refine_one, fix_residue_numbers
import pdb
import os
import shutil
from pdb2sql import StructureSimilarity


def triangulate(ppi, config):
    triangulate_single(ppi, config, overwrite=True)

def preprocess(processed_ppi, config):
    for ppi in processed_ppi:
        print("PPI == ", ppi)

        out_grid = config['dirs']['grid'] +  ppi + '.npy'
        print("out_grid ==", out_grid)
        if os.path.exists(out_grid):
            print(f"{out_grid} already exists...")
            continue

        try:
            triangulate_single(ppi, config)
        except:
            triangulate(ppi, config)
        try:
            compute_one_patch(ppi, config)
        except:
            triangulate(ppi, config)
            compute_one_patch(ppi, config)
        map_patch_atom([ppi], config)

        try:
            convert_to_images([ppi], config)
        except:
            print(f"WARNING::Couldn't convert to image for {ppi}.")
            print("Attempting to fix the file...")
            triangulate_single(ppi, config, overwrite=True)
            compute_one_patch(ppi, config)
            map_patch_atom([ppi], config)
            convert_to_images([ppi], config)

        try:
            pid, ch1, ch2 = ppi.split('_')
            shutil.copy(f"{config['dirs']['patches']}/{ppi}/{ch1}_iface_labels.npy", f"{config['dirs']['grid']}/{pid}_{ch1}_iface_labels.npy")
            shutil.copy(f"{config['dirs']['patches']}/{ppi}/{ch1}_selected_patches.npy", f"{config['dirs']['grid']}/{pid}_{ch1}_selected_patches.npy")
            shutil.copy(f"{config['dirs']['patches']}/{ppi}/{ch2}_iface_labels.npy", f"{config['dirs']['grid']}/{pid}_{ch2}_iface_labels.npy")
            shutil.copy(f"{config['dirs']['patches']}/{ppi}/{ch2}_selected_patches.npy", f"{config['dirs']['grid']}/{pid}_{ch2}_selected_patches.npy")
            shutil.rmtree(config['dirs']['refined'] + ppi)
            shutil.rmtree(config['dirs']['chains_pdb'] + ppi)
            shutil.rmtree(config['dirs']['surface_ply'] + ppi)  
        except Exception as e:
            print(f"WARNING::Couldn't delete useless files/folders. Error: {e}")


def check_processed(ppis, config):
    processed = []
    for ppi in ppis:
        if os.path.exists(config['dirs']['grid']+'{}.npy'.format(ppi)):
            processed.append(ppi)
    return processed


def fix_protonated(ppi_list, config):
    print("HellOOOOOOOO")
    for ppi in ppi_list:
        pid, ch1, ch2 = ppi.split('_')
        pdb_file = config['dirs']['protonated_pdb'] + pid + '.pdb'
        pdb_tmp_file = config['dirs']['protonated_pdb'] + "/" + pid + "_tmp.pdb"
        shutil.copyfile(pdb_file, pdb_tmp_file)
        with open(config['dirs']['protonated_pdb'] + pid + '.pdb', 'w') as out:
            with open(config['dirs']['protonated_pdb'] + pid + '_tmp.pdb', 'r') as f:
                for line in f.readlines():
                    if line[:4] == 'ATOM' or line[:6] == 'HETATM':
                        if line[72]!=' ':
                            line = line[:21] + line[72] + line[22:72] + ' \n'

                    out.write(line)
        os.remove(config['dirs']['protonated_pdb'] + pid + '_tmp.pdb')

def prepare(args):
    """
    Data prepare module
    """
    start = time.time()

    print("[ {} ] Start data prepare...".format(get_date()))

    ppi_list = []

    if (not args.list and not args.ppi) or (args.list is not None and args.ppi is not None):
        raise AssertionError('Specify either "--list" or "--ppi" input')

    if (args.list is not None):
        ppi_list = [x.strip('\n') for x in open(args.list)]
    elif (args.ppi is not None):
        ppi_list = [args.ppi]

    # Read config
    config = read_config(args)
    print("[ {} ] Configuration parameters:".format(get_date()))
    print(config)
    print("[ {} ] Preprocessing {} complexes".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), len(ppi_list)))

    if not args.no_download:
        processed_ppi = download(ppi_list, config)
    else:
        processed_ppi = ppi_list

    if args.fix_pdb:
        fix_protonated(ppi_list, config)

    if not args.download_only:
        preprocess(processed_ppi, config)

    print("[ {} ] The data preparation is complete.".format(get_date()))
    print("Total execution time for data preparation: {:.2f}m".format((time.time() - start)/60))
