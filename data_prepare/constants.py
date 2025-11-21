import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import CaPPBuilder

MAX_NODES_PER_JOB = 4
ATOM_COUNT_LIMIT = 17500
RCSB_BASE_URL = 'ftp://ftp.wwpdb.org/pub/pdb/data/biounit/coordinates/divided/'

# The PDB codes of structures added between DB4 and DB5 (to be used for testing dataset)
DB5_TEST_PDB_CODES = ['3R9A', '4GAM', '3AAA', '4H03', '1EXB',
                      '2GAF', '2GTP', '3RVW', '3SZK', '4IZ7',
                      '4GXU', '3BX7', '2YVJ', '3V6Z', '1M27',
                      '4FQI', '4G6J', '3BIW', '3PC8', '3HI6',
                      '2X9A', '3HMX', '2W9E', '4G6M', '3LVK',
                      '1JTD', '3H2V', '4DN4', 'BP57', '3L5W',
                      '3A4S', 'CP57', '3DAW', '3VLB', '3K75',
                      '2VXT', '3G6D', '3EO1', '4JCV', '4HX3',
                      '3F1P', '3AAD', '3EOA', '3MXW', '3L89',
                      '4M76', 'BAAD', '4FZA', '4LW4', '1RKE',
                      '3FN1', '3S9D', '3H11', '2A1A', '3P57']

# Postprocessing logger dictionary
DEFAULT_DATASET_STATISTICS = dict(num_of_processed_complexes=0, num_of_df0_residues=0, num_of_df1_residues=0,
                                  num_of_df0_interface_residues=0, num_of_df1_interface_residues=0,
                                  num_of_pos_res_pairs=0, num_of_neg_res_pairs=0, num_of_res_pairs=0,
                                  num_of_valid_df0_ss_values=0, num_of_valid_df1_ss_values=0,
                                  num_of_valid_df0_rsa_values=0, num_of_valid_df1_rsa_values=0,
                                  num_of_valid_df0_rd_values=0, num_of_valid_df1_rd_values=0,
                                  num_of_valid_df0_protrusion_indices=0, num_of_valid_df1_protrusion_indices=0,
                                  num_of_valid_df0_hsaacs=0, num_of_valid_df1_hsaacs=0,
                                  num_of_valid_df0_cn_values=0, num_of_valid_df1_cn_values=0,
                                  num_of_valid_df0_sequence_feats=0, num_of_valid_df1_sequence_feats=0,
                                  num_of_valid_df0_amide_normal_vecs=0, num_of_valid_df1_amide_normal_vecs=0)
PDB_PARSER = PDBParser()
CA_PP_BUILDER = CaPPBuilder()

# Dict for converting three letter codes to one letter codes
D3TO1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
         'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
RES_NAMES_LIST = list(D3TO1.keys())

# PSAIA features
PSAIA_COLUMNS = ['avg_cx', 's_avg_cx', 's_ch_avg_cx', 's_ch_s_avg_cx', 'max_cx', 'min_cx']

# Constants for calculating half sphere exposure statistics
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY-'
AMINO_ACID_IDX = dict(zip(AMINO_ACIDS, range(len(AMINO_ACIDS))))

# Default fill values for missing features
HSAAC_DIM = 42
DEFAULT_MISSING_FEAT_VALUE = np.nan
DEFAULT_MISSING_SS = '-'
DEFAULT_MISSING_RSA = DEFAULT_MISSING_FEAT_VALUE
DEFAULT_MISSING_RD = DEFAULT_MISSING_FEAT_VALUE
DEFAULT_MISSING_PROTRUSION_INDEX = [DEFAULT_MISSING_FEAT_VALUE for _ in range(6)]
DEFAULT_MISSING_HSAAC = [DEFAULT_MISSING_FEAT_VALUE for _ in range(HSAAC_DIM)]
DEFAULT_MISSING_CN = DEFAULT_MISSING_FEAT_VALUE
DEFAULT_MISSING_SEQUENCE_FEATS = np.array([DEFAULT_MISSING_FEAT_VALUE for _ in range(27)])
DEFAULT_MISSING_NORM_VEC = [DEFAULT_MISSING_FEAT_VALUE for _ in range(3)]

NUM_ALLOWABLE_NANS = 5

FEAT_COLS = [
    'resname',
    'ss_value',
    'rsa_value',
    'rd_value'
]
FEAT_COLS.extend(
    PSAIA_COLUMNS +
    ['hsaac',
     'cn_value',
     'sequence_feats',
     'amide_norm_vec',
     ])

ALLOWABLE_FEATS = [
    ["TRP", "PHE", "LYS", "PRO", "ASP", "ALA", "ARG", "CYS", "VAL", "THR",
     "GLY", "SER", "HIS", "LEU", "GLU", "TYR", "ILE", "ASN", "MET", "GLN"],
    ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-'], 
    [],  
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [[]],
    [],
    [[]],
    [[]],
]
