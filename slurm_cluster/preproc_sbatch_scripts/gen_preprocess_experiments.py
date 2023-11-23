#!/usr/bin/env python3

"""script for generating the preprocessing experiment for the wmh challenge data"""

# get the parser to find out how many individuals there are to process
from twaidata.mri_dataset_directory_parsers.parser_selector import select_parser
import os
import argparse

USER = os.getenv('USER')

SCRATCH_DISK = '/disk/scratch'  
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

SCRIPT_DIR = f"/home/{USER}/ipdis/Trustworthai-MRI-WMH/twaidata/MRI_preprep"

DATA_HOME = f"{SCRATCH_HOME}/ipdis/preprep/in_data"

def construct_parser():
    # preprocessing settings
    parser = argparse.ArgumentParser(description = "MRI nii.gz simple preprocessing pipeline")
    
    parser.add_argument('-n', '--name', required=True, help='name of dataset folder on filesystem, must correspond to a twaidata parser dataset name (e.g WMH_challenge_dataset)')
    parser.add_argument('-d', '--dfs_dir', required=True, help="path to the parent folder of the dataset on the distributed file system (i.e not on scratch space after rysnc but the original copy")
    parser.add_argument('-p', '--process_num', default=1, type=int, help='number of processes to split preprocessing across')
    parser.add_argument('-f', '--force', default='false', type=str, help='force redo reprocessing')
    return parser



def main(args):
    num_processes = args.process_num
    dataset_name = args.name
    dataset_original_dir = args.dfs_dir
    force = args.force
    
    RESULT_PATH = f"{SCRATCH_HOME}/ipdis/preprep/out_data/{dataset_name}"
    
    data_parser = select_parser(dataset_name, dataset_original_dir, RESULT_PATH)

    files_map = data_parser.get_dataset_inout_map()
    num_individuals = len(list(files_map.keys()))

    # generate experiment file
    output_file = open(f"preprocess_{dataset_name}_experiment.txt", "w")
    start = 0
    diff = num_individuals // num_processes
    for p in range(num_processes):
        end = start + diff

        if p == (num_processes - 1):
            end = -1 # on last process just do whatever files are remaining.

        expt_call = (
            f"python {SCRIPT_DIR}/simple_preprocess_st1.py "
            f"-i {DATA_HOME} "
            f"-o {RESULT_PATH} "
            f"-s {start} "
            f"-e {end} "
            f"-n {dataset_name} "
	    f"-f {force} "
        )
        print(expt_call, file=output_file)

        start = end

    output_file.close()
    
    
if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
            
    
