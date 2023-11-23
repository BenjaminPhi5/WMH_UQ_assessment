from fsl.wrappers import fast
import os
from natsort import natsorted
from joblib import Parallel, delayed
import argparse

main_root = "/home/s2208943/ipdis/data/preprocessed_data/EdData/"
output_folder = "/home/s2208943/ipdis/data/preprocessed_data/Ed_fast_WM_masks/"
domains = ['domainA', 'domainB', 'domainC', 'domainD']

def process(files_per_domain, domain_index, ending, classes):
    input_paths = [main_root + domains[domain_index] + "/imgs/" + f for f in files_per_domain[domain_index]]
    output_paths = [output_folder + f.split("_")[0] for f in files_per_domain[domain_index]]
    
    for i in range(len(input_paths)):
        fast(input_paths[i], out=output_paths[i]+ending, n_classes=classes)

def construct_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-t', '--type', required=True, help="what filename to use as input to fast, e.g _FLAIR_BET.nii.gz")
    parser.add_argument('-c', '--classes', default=3, type=int, help="number of classes used by FSL FAST. Use 3 for T1")
    
    return parser
        
def main(args):
    filetype_match = args.type
    classes = args.classes
    
    files_per_domain = []
    for d in domains:
        fdir = main_root + d + "/imgs/"
        files = os.listdir(fdir)
        files_per_domain.append([f for f in files if filetype_match in f])


    # fast is super slow so im splitting by domain and running in parallel,
    # but it all goes to the same folder
    Parallel(n_jobs=4)(
            delayed(process)(files_per_domain, domain_index, filetype_match, classes)
            for domain_index in range(len(domains))
    )
    
if __name__ == '__main__':
    print("hi")
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
