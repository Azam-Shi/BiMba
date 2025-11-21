import pymesh
import numpy as np
import os
from shutil import copyfile, rmtree
from utils.utils import get_date, extract_pdb_chain
from masif.source.default_config.masif_opts import masif_opts
from masif.source.triangulation.computeMSMS import computeMSMS
from masif.source.triangulation.fixmesh import fix_mesh
from masif.source.input_output.extractPDB import extractPDB
from masif.source.input_output.save_ply import save_ply
from masif.source.triangulation.computeHydrophobicity import computeHydrophobicity
from masif.source.triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from masif.source.triangulation.computeAPBS import computeAPBS
from masif.source.triangulation.compute_normal import compute_normal
from sklearn.neighbors import KDTree


def triangulate_one(pid, ch, config, pdb_filename):
    """
    triangulate one chain
    """
    chains_pdb_dir = config['dirs']['chains_pdb']
    tmp_pdb_dir = chains_pdb_dir + pid + '_' + ch + '/'
    if not os.path.exists(tmp_pdb_dir):
        os.mkdir(tmp_pdb_dir)
    out_filename1 = tmp_pdb_dir
    extractPDB(pdb_filename, out_filename1+ '.pdb', ch)
    vertices1, faces1, normals1, names1, areas1 = computeMSMS(out_filename1+ '.pdb', protonate=True)
    vertex_hbond = computeCharges(out_filename1, vertices1, names1)
    vertex_hphobicity = computeHydrophobicity(names1)
    vertices2 = vertices1
    faces2 = faces1
    mesh = pymesh.form_mesh(vertices2, faces2)
    regular_mesh = fix_mesh(mesh, config['mesh']['mesh_res']) ## Azam: it # of vertices to be equal to # of residues (Not 100% sure actually)
    vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
    vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
                                          vertex_hbond, masif_opts)
    vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, \
                                               vertex_hphobicity, masif_opts)
    vertex_charges = computeAPBS(regular_mesh.vertices, out_filename1 + ".pdb", out_filename1)
    extract_pdb_chain(config['dirs']['protonated_pdb'] + pid + '.pdb',  chains_pdb_dir + '{}_{}.pdb'.format(pid, ch), ch)
    rmtree(tmp_pdb_dir)
    iface = np.zeros(len(regular_mesh.vertices))
    v3, f3, _, _, _ = computeMSMS(pdb_filename, protonate=True)
    mesh = pymesh.form_mesh(v3, f3)
    full_regular_mesh = mesh
    v3 = full_regular_mesh.vertices
    kdt = KDTree(v3)
    d, r = kdt.query(regular_mesh.vertices)
    d = np.square(d)
    assert (len(d) == len(regular_mesh.vertices))
    iface_v = np.where(d >= 2.0)[0]
    iface[iface_v] = 1.0
    outply = config['dirs']['surface_ply'] + pid + '_' + ch
    save_ply(outply + ".ply", regular_mesh.vertices, \
             regular_mesh.faces, normals=vertex_normal, charges=vertex_charges, \
             normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity, \
             iface=iface)
    return


def triangulate(ppi, config):
    print("\t[ {} ] Start triangulation... ".format(get_date()))
    print(ppi)
    processed_ppi = []
    try:
        pid, ch = ppi.split('_')
        outply1 = config['dirs']['surface_ply'] + pid + '_' + ch + '.ply'
        if os.path.exists(outply1) and os.path.exists(outply1) or os.path.exists(f"{config['dirs']['grid']}/{ppi}.npy"):
            print("Triangulated structures already exist for {}. Skipping...".format(ppi))
            processed_ppi.append(ppi)
        else:
            pdb_filename = config['dirs']['chains_pdb'] + pid + '_' + ch + '.pdb' #**
            triangulate_one(pid, ch, config, pdb_filename)
    except ValueError:
        try:
            pid, ch1, ch2 = ppi.split('_')
            outply1 = config['dirs']['surface_ply'] + pid + '_' + ch1 + '.ply'
            outply2 = config['dirs']['surface_ply'] + pid + '_' + ch2 + '.ply'
        except ValueError:
            print(f"Skipping {ppi}: Unexpected format for traiangulating.")
        if os.path.exists(outply1) and os.path.exists(outply1) or os.path.exists(f"{config['dirs']['grid']}/{ppi}.npy"):
            print("Triangulated structures already exist for {}. Skipping...".format(ppi))
            processed_ppi.append(ppi)
        else:
            pdb_filename = config['dirs']['chains_pdb'] + pid + '_' + ch1 + '.pdb' #**
            triangulate_one(pid, ch1, config, pdb_filename)
            
            pdb_filename = config['dirs']['chains_pdb'] + pid + '_' + ch2 + '.pdb'
            triangulate_one(pid, ch2, config, pdb_filename)
    if os.path.exists(outply1):
        processed_ppi.append(ppi)
    elif os.path.exists(outply1) and os.path.exists(outply2):
        processed_ppi.append(ppi)
    return processed_ppi

