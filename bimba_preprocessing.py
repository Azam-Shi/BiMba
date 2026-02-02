#!/usr/local/bin/python

import argparse
from data_prepare.data_prepare import prepare
from data_prepare.data_prepare import prepare

parser = argparse.ArgumentParser(prog="BiMba")
parser.add_argument('--config', help='config file')
sp = parser.add_subparsers()

sp_prepare = sp.add_parser('prepare', help='Data preparation module')
sp_prepare.add_argument('--list', help='List with proteins in format PID_A or protein complexes in format PID_A_B')
sp_prepare.add_argument('--p', help='proteins in format PID_A or PID_A_B (mutually exclusive with the --list option)')
sp_prepare.add_argument('--no_download',  default=False, action="store_true", help='If set True, the pipeline will skip the download part.')
sp_prepare.add_argument('--download_only',  default=False, action="store_true", help='If set True, the program will only download PDB structures without processing them.')

sp_prepare.add_argument('--fix_pdb', default=False, action='store_true', help='If set True, fix PDB structures.')

sp_prepare.set_defaults(func=prepare)
args = parser.parse_args()

if vars(args).get('func') is None:
    print("Error: Please provide arguments (see the help message below).")
    parser.print_help()
    exit()

args.func(args)



