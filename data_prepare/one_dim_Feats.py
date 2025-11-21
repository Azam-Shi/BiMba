import logging
import os
from pathlib import Path
import atom3.database as db
import numpy as np
import pandas as pd
import torch
from Bio.PDB import Selection, PDBIO
from Bio.PDB.DSSP import dssp_dict_from_pdb_file, DSSP
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.PDB.vectors import Vector
from Bio.Data.IUPACData import protein_letters_3to1
from scipy import spatial
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from Bio.PDB.PDBParser import PDBParser
from atom3.pair import Pair
from Bio.PDB import StructureBuilder
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree
from .constants import D3TO1


from .constants import AMINO_ACIDS, AMINO_ACID_IDX, MAX_NODES_PER_JOB, \
    PSAIA_COLUMNS, PDB_PARSER, DEFAULT_DATASET_STATISTICS, ATOM_COUNT_LIMIT, RCSB_BASE_URL, FEAT_COLS, \
    ALLOWABLE_FEATS, DEFAULT_MISSING_SS, DEFAULT_MISSING_PROTRUSION_INDEX, DEFAULT_MISSING_RSA, \
    DEFAULT_MISSING_RD, DEFAULT_MISSING_HSAAC, DEFAULT_MISSING_CN, DEFAULT_MISSING_SEQUENCE_FEATS, \
    DEFAULT_MISSING_NORM_VEC, HSAAC_DIM, NUM_ALLOWABLE_NANS
try:
    from types import SliceType
except ImportError:
    SliceType = slice


def get_dssp_dict_for_pdb_model(pdb_model, raw_pdb_filename):
    """Run DSSP to calculate secondary structure features for a given PDB file."""
    dssp_dict = {}
    try:
        dssp_dict = DSSP(pdb_model, raw_pdb_filename)
    except Exception:
        logging.info("No DSSP features found for {:}".format(pdb_model))
    return dssp_dict


def get_msms_rd_dict_for_pdb_model(pdb_model):
    """Run MSMS to calculate residue depth model for a given PDB model."""
    rd_dict = {}
    try:
        rd_dict = ResidueDepth(pdb_model)
    except Exception:
        logging.info("No MSMS residue depth model found for {:}".format(pdb_model))
    return rd_dict



# -------------------------------------------------------------------------------------------------------------------------------------
# Following code derived from PAIRpred (https://combi.cs.colostate.edu/supplements/pairpred/):
# -------------------------------------------------------------------------------------------------------------------------------------
def get_coords(residues):
    """
    Get atom coordinates given a list of biopython residues
    """
    Coords = []
    for (idx, r) in enumerate(residues):
        v = [ak.get_coord() for ak in r.get_list()]
        Coords.append(v)
    return Coords


def get_res_letter(residue):
    """
    Get the letter code for a biopython residue object
    """
    r2name = residue.get_resname()
    if r2name in protein_letters_3to1:
        scode = protein_letters_3to1[r2name]
    else:
        scode = '-'
    return scode

def get_side_chain_vector(residue):
    """
    Find the average of the unit vectors to different atoms in the side chain
    from the c-alpha atom. For glycine the average of the N-Ca and C-Ca is
    used.
    Returns (C-alpha coordinate vector, side chain unit vector) for residue r
    """
    u = None
    gly = 0
    if is_aa(residue) and residue.has_id('CA'):
        ca = residue['CA'].get_coord()
        dv = np.array([ak.get_coord() for ak in residue.get_unpacked_list()[4:]])
        if len(dv) < 1:
            if residue.has_id('N') and residue.has_id('C'):
                dv = [residue['C'].get_coord(), residue['N'].get_coord()]
                dv = np.array(dv)
                gly = 1
            else:
                return None
        dv = dv - ca
        if gly:
            dv = -dv
        n = np.sum(np.abs(dv) ** 2, axis=-1) ** (1. / 2)
        v = dv / n[:, np.newaxis]
        v = v.mean(axis=0)
        u = (Vector(ca), Vector(v))
    return u


def get_similarity_matrix(coords, sg=2.0, thr=1e-3):
    """
    Instantiates the distance based similarity matrix (S). S is a tuple of
    lists (I,V). |I|=|V|=|R|. Each I[r] refers to the indices
    of residues in R which are "close" to the residue indexed by r in R, and V[r]
    contains a list of the similarity scores for the corresponding residues.
    The distance between two residues is defined to be the minimum distance of
    any of their atoms. The similarity score is evaluated as
        s = exp(-d^2/(2*sg^2))
    This ensures that the range of similarity values is 0-1. sg (sigma)
    determines the extent of the neighborhood.
    Two residues are defined to be close to one another if their similarity
    score is greater than a threshold (thr).
    Residues (or ligands) for which DSSP features are not available are not
    included in the distance calculations.
    """
    sg = 2 * (sg ** 2)
    I = [[] for k in range(len(coords))]
    V = [[] for k in range(len(coords))]
    for i in range(len(coords)):
        for j in range(i, len(coords)):
            d = spatial.distance.cdist(coords[i], coords[j]).min()
            s = np.exp(-(d ** 2) / sg)
            if s > thr:
                I[i].append(j)
                V[i].append(s)
                if i != j:
                    I[j].append(i)
                    V[j].append(s)
    similarity_matrix = (I, V)
    coordinate_numbers = np.array([len(a) for a in similarity_matrix[0]])
    return similarity_matrix, coordinate_numbers


def get_hsacc(residues, similarity_matrix, raw_pdb_filename):
    """
    Compute the Half sphere exposure statistics
    The up direction is defined as the direction of the side chain and is
    calculated by taking average of the unit vectors to different side chain
    atoms from the C-alpha atom
    Anything within the up half sphere is counted as up and the rest as
    down
    """
    N = len(residues)
    Na = len(AMINO_ACIDS)
    UN = np.zeros(N)
    DN = np.zeros(N)
    UC = np.zeros((Na, N))
    DC = np.zeros((Na, N))
    for (i, r) in enumerate(residues):
        u = get_side_chain_vector(r)
        if u is None:
            UN[i] = np.nan
            DN[i] = np.nan
            UC[:, i] = np.nan
            DC[:, i] = np.nan
            logging.info(f'No side chain vector found for residue #{i} in PDB file {raw_pdb_filename}')
        else:
            idx = AMINO_ACID_IDX[get_res_letter(r)]
            UC[idx, i] = UC[idx, i] + 1
            DC[idx, i] = DC[idx, i] + 1
            n = similarity_matrix[0][i]
            for j in n:
                r2 = residues[j]
                if is_aa(r2) and r2.has_id('CA'):
                    v2 = r2['CA'].get_vector()
                    scode = get_res_letter(r2)
                    idx = AMINO_ACID_IDX[scode]
                    angle = u[1].angle((v2 - u[0]))
                    if angle < np.pi / 2.0:
                        UN[i] = UN[i] + 1
                        UC[idx, i] = UC[idx, i] + 1
                    else:
                        DN[i] = DN[i] + 1
                        DC[idx, i] = DC[idx, i] + 1
    UC = UC / (1.0 + UN)
    DC = DC / (1.0 + DN)
    return UC, DC


def min_max_normalize_feature_array(features):
    """Independently for each column, normalize feature array values to be in range [0, 1]."""
    scaler = MinMaxScaler()
    scaler.fit(features)
    features_scaled = scaler.transform(features)
    return features_scaled


def min_max_normalize_feature_tensor(features):
    """Normalize provided feature tensor to have its values be in range [0, 1]."""
    min_value = min(features)
    max_value = max(features)
    features_std = torch.tensor([(value - min_value) / (max_value - min_value) for value in features])
    features_scaled = features_std * (max_value - min_value) + min_value
    return features_scaled

def get_dssp_dict_for_pdb_file(pdb_filename):
    """Run DSSP to calculate secondary structure features for a given PDB file."""
    dssp_dict = {}
    try:
        dssp_tuple = dssp_dict_from_pdb_file(pdb_filename)
        dssp_dict = dssp_tuple[0]
    except Exception:
        logging.info("No DSSP features found for {:}".format(pdb_filename))
    return dssp_dict

def get_dssp_dict_for_pdb_model(pdb_model, raw_pdb_filename):
    """Run DSSP to calculate secondary structure features for a given PDB file."""
    dssp_dict = {}
    try:
        dssp_dict = DSSP(pdb_model, raw_pdb_filename)
    except Exception:
        logging.info("No DSSP features found for {:}".format(pdb_model))
    return dssp_dict

def get_msms_rd_dict_for_pdb_model(pdb_model):
    """Run MSMS to calculate residue depth model for a given PDB model."""
    rd_dict = {}
    try:
        rd_dict = ResidueDepth(pdb_model)
    except Exception:
        logging.info("No MSMS residue depth model found for {:}".format(pdb_model))
    return rd_dict

def get_hsaac_for_pdb_residues(residues, similarity_matrix, raw_pdb_filename):
    """Run BioPython to calculate half-sphere amino acid composition (HSAAC) for a given list of PDB residues."""
    hsaacs = np.array([DEFAULT_MISSING_HSAAC for _ in range(len(residues))])
    try:
        UC, DC = get_hsacc(residues, similarity_matrix, raw_pdb_filename)
        hsaacs = np.concatenate((UC, DC))
    except Exception:
        logging.info("No half-sphere amino acid compositions (HSAACs) found for PDB file {:}".format(raw_pdb_filename))
    return hsaacs

def get_dssp_value_for_residue(dssp_dict: dict, feature: str, chain: str, residue: int):
    """Return a secondary structure (SS) value or a relative solvent accessibility (RSA) value for a given chain-residue pair."""
    dssp_value = DEFAULT_MISSING_SS if feature == 'SS' else DEFAULT_MISSING_RSA
    try:
        if feature == 'SS':
            dssp_values = dssp_dict[chain, (' ', residue, ' ')]
            dssp_value = dssp_values[2]
        else:  # feature == 'RSA'
            dssp_values = dssp_dict[chain, (' ', residue, ' ')]
            dssp_value = dssp_values[3]
    except Exception:
        logging.info("No DSSP entry found for {:}".format((chain, (' ', residue, ' '))))
    return dssp_value

def get_msms_rd_value_for_residue(rd_dict: dict, chain: str, residue: int):
    """Return an alpha-carbon residue depth (RD) value for a given chain-residue pair."""
    ca_depth_value = DEFAULT_MISSING_RD
    try:
        rd_value, ca_depth_value = rd_dict[chain, (' ', residue, ' ')]
    except Exception:
        logging.info("No MSMS residue depth entry found for {:}".format((chain, (' ', residue, ' '))))
    return ca_depth_value[0] if type(ca_depth_value) == list else ca_depth_value

def get_hsaac_for_residue(hsaac_matrix: np.array, residue_counter: int, chain: str, residue_id: int):
    """Return a half-sphere amino acid composition (HSAAC) for a given chain-residue pair."""
    hsaac = np.array(DEFAULT_MISSING_HSAAC)
    try:
        hsaac = hsaac_matrix[:, residue_counter]
    except Exception:
        logging.info(
            "No half-sphere amino acid composition entry found for {:}".format((chain, (' ', residue_id, ' '))))
    return np.array(DEFAULT_MISSING_HSAAC) if len(hsaac) > HSAAC_DIM else hsaac

def get_cn_value_for_residue(cn_values: np.array, residue_counter: int, chain: str, residue_id: int):
    """Return a coordinate number value for a given chain-residue pair."""
    cn_value = DEFAULT_MISSING_CN
    try:
        cn_value = cn_values[residue_counter]
    except Exception:
        logging.info("No coordinate number entry found for {:}".format((chain, (' ', residue_id, ' '))))
    return cn_value

def get_norm_vec_for_residue(df: pd.DataFrame, ca_atom: pd.Series, chain: str, residue_id: int):
    """Return a normal vector for a given residue."""
    norm_vec = DEFAULT_MISSING_NORM_VEC
    try:
        cb_atom = df[(df.chain == ca_atom.chain) &
                     (df.residue == ca_atom.residue) &
                     (df.atom_name == 'CB')]
        n_atom = df[(df.chain == ca_atom.chain) &
                    (df.residue == ca_atom.residue) &
                    (df.atom_name == 'N')]
        vec1 = ca_atom[['x', 'y', 'z']].to_numpy() - cb_atom[['x', 'y', 'z']].to_numpy()
        vec2 = cb_atom[['x', 'y', 'z']].to_numpy() - n_atom[['x', 'y', 'z']].to_numpy()
        norm_vec = np.cross(vec1, vec2)
    except Exception:
        logging.info("No normal vector entry found for {:}".format(chain, (' ', residue_id, ' ')))
    if len(norm_vec) == 0:
        norm_vec = DEFAULT_MISSING_NORM_VEC
    return norm_vec

def parse_pdb_file_to_df(pdb_path, chain_id):
    """
    Parses a PDB file and extracts atom-level information for a specified chain.
    Args:
        pdb_path (str): Path to the PDB file.
        chain_id (str): Chain ID to extract.
    Returns:
        pd.DataFrame: DataFrame with columns: ['atom_name', 'residue', 'resname', 'chain', 'x', 'y', 'z']
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    model = structure[0]
    data = []
    for chain in model:
        if chain.id != chain_id:
            continue
        for residue in chain:
            if residue.id[0] != ' ':
                continue
            resname = residue.get_resname()
            resid = str(residue.id[1])
            for atom in residue:
                atom_name = atom.get_name()
                x, y, z = atom.coord
                data.append({
                    'atom_name': atom_name,
                    'residue': resid,
                    'resname': resname,
                    'chain': chain_id,
                    'x': x,
                    'y': y,
                    'z': z
                })
    return pd.DataFrame(data)

def get_res_letter_3to1(resname):
    """
    Convert 3-letter residue name to 1-letter code using the D3TO1 mapping from constants.
    Automatically strips whitespace and uppercases the input to ensure robustness.
    """
    return D3TO1.get(resname.strip().upper(), "-")
            
def get_polarity_vector(res_letter):
    """Return the polarity vector (one-hot encoded) for a given residue one-letter code."""
    nonpolar = {'A', 'V', 'L', 'I', 'M', 'F', 'W', 'P', 'G'}
    polar_uncharged = {'S', 'T', 'C', 'Y', 'N', 'Q'}
    positively_charged = {'K', 'R', 'H'}
    negatively_charged = {'D', 'E'}
    if res_letter in nonpolar:
        return [1, 0, 0, 0]
    elif res_letter in polar_uncharged:
        return [0, 1, 0, 0]
    elif res_letter in positively_charged:
        return [0, 0, 1, 0]
    elif res_letter in negatively_charged:
        return [0, 0, 0, 1]
    else:
        return [0, 0, 0, 0]

def postprocess_pruned_pair(ppi, config):
    try:
        pid, ch1_orig = ppi.split('_')
        pdb_filename = os.path.join(config['dirs']['chains_pdb'], f"{pid}_{ch1_orig}.pdb")
        external_feats_dir = config['dirs']['external_feats']
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pid, pdb_filename)
        for chain_id in list(ch1_orig):
            group_df = []
            group_residues = []
            dssp_dict = get_dssp_dict_for_pdb_model(structure[0], pdb_filename)
            rd_dict = get_msms_rd_dict_for_pdb_model(structure[0])
            df = parse_pdb_file_to_df(pdb_filename, chain_id=chain_id)
            group_df.append(df)
            residues = [
                res for res in Selection.unfold_entities(structure, 'R')
                if res.get_id()[0] == ' ' and res.get_parent().id == chain_id
            ]
            group_residues.extend(residues)
            if len(group_df) == 0 or len(group_residues) == 0:
                print(f"[WARN] No residues found for chain {chain_id} in {ppi}")
                continue
            df = pd.concat(group_df, ignore_index=True)
            print(f"Parsed {len(df)} atoms for chain {chain_id}")
            similarity_matrix, coordinate_numbers = get_similarity_matrix(get_coords(group_residues))
            hsaac_matrix = get_hsaac_for_pdb_residues(group_residues, similarity_matrix, pdb_filename)
            def approximate_protrusion_index(residues, radius=10.0):
                ca_coords = [res['CA'].coord for res in residues if 'CA' in res]
                if len(ca_coords) == 0:
                    logging.warning("No Ca atoms found for protrusion index computation.")
                    return np.zeros((len(residues), 1))
                ca_coords = np.array(ca_coords)
                tree = cKDTree(ca_coords)
                raw_scores = [len(tree.query_ball_point(coord, r=radius)) - 1 for coord in ca_coords]
                max_val = max(raw_scores) or 1
                norm_scores = [(1 - (s / max_val)) for s in raw_scores]
                return np.array(norm_scores).reshape(-1, 1)
            protrusion_scores = approximate_protrusion_index(group_residues)
            ss_values, rsa_values, rd_values = [], [], []
            hsaacs, cn_vals, norm_vecs = [], [], []
            one_letter_col, polarity_vectors = [], []
            residue_counter = 0
            residue_idx = 0
            print(f"{ppi}: rd={len(rd_values)}, cn={len(cn_vals)}, protrusion={len(protrusion_scores)}")
            for _, row in df.iterrows():
                is_ca = row.atom_name.strip() == 'CA'
                residue_id = row.residue.strip().lstrip("-+")
                residue_id = int(residue_id) if residue_id.isdigit() else -1

                ss = get_dssp_value_for_residue(dssp_dict, 'SS', chain_id, residue_id) if is_ca else '-'
                rsa = get_dssp_value_for_residue(dssp_dict, 'RSA', chain_id, residue_id) if is_ca else 0.0
                rd = get_msms_rd_value_for_residue(rd_dict, chain_id, residue_id) if is_ca else 0.0
                hs = get_hsaac_for_residue(hsaac_matrix, residue_counter, chain_id, residue_id) if is_ca else [0.0]*42
                cn = get_cn_value_for_residue(coordinate_numbers, residue_counter, chain_id, residue_id) if is_ca else 0.0
                nv = get_norm_vec_for_residue(df, row, chain_id, residue_id) if is_ca else [0.0]*3
                ss_values.append(ss)
                rsa_values.append(rsa)
                rd_values.append(rd)
                hsaacs.append(hs)
                cn_vals.append(cn)
                norm_vecs.append(nv)
                if is_ca:
                    res1 = get_res_letter_3to1(row.resname)
                    one_letter_col.append(res1)
                    polarity_vectors.append(get_polarity_vector(res1))
                    residue_counter += 1
                else:
                    one_letter_col.append('-')
                    polarity_vectors.append([0, 0, 0, 0])

            # === Normalize numeric features ===
            if len(rd_values) > 0:
                rd_values = min_max_normalize_feature_array(np.array(rd_values).reshape(-1, 1))
            if len(cn_vals) > 0:
                cn_vals = min_max_normalize_feature_array(np.array(cn_vals).reshape(-1, 1))
            if len(protrusion_scores) > 0:
                protrusion_scores = min_max_normalize_feature_array(protrusion_scores)
            df.insert(4, '1letter_resname', one_letter_col)
            df.insert(5, 'ss_value', ss_values)
            df.insert(6, 'rsa_value', rsa_values)
            df.insert(7, 'rd_value', rd_values)
            protrusion_col = []
            residue_idx = 0
            for _, row in df.iterrows():
                if row.atom_name.strip() == 'CA':
                    protrusion_col.append(protrusion_scores[residue_idx][0])
                    residue_idx += 1
                else:
                    protrusion_col.append(0.0)
            df.insert(8, 'protrusion_index', protrusion_col)
            df.insert(9, 'hsaac', hsaacs)
            df.insert(10, 'cn_value', cn_vals)
            df.insert(11, 'amide_norm_vec', norm_vecs)
            df.insert(12, 'polarity', polarity_vectors)

            # Save CSV
            output_csv_name = f"{pid}_{chain_id}_feats.csv"
            output_csv_path = os.path.join(external_feats_dir, output_csv_name)
            df.to_csv(output_csv_path, index=False)
    except Exception as e:
        print(f"[ERROR] Skipping PPI: {ppi} due to error:\n{e}\n")
    return
