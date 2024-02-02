import sys
import os
import argparse
from natsort import natsorted
from abc import ABC, abstractmethod
import subprocess

# todo sort out these imports so that they work like a proper module should.
from twaidata.MRI_preprep.io import load_nii_img, save_nii_img, FORMAT
from twaidata.MRI_preprep.normalize_brain import normalize
from twaidata.MRI_preprep.resample import resample_and_return, resample_and_save
from twaidata.mri_dataset_directory_parsers.parser_selector import select_parser
from twaidata.MRI_preprep.skull_strip import *

def construct_parser():
    # preprocessing settings
    parser = argparse.ArgumentParser(description = "MRI nii.gz simple preprocessing pipeline")
    
    parser.add_argument('-i', '--in_dir', required=True, help='Path to parent of the dataset to be preprocessed')
    parser.add_argument('-o', '--out_dir', required=True, help='Path to the preprocessed data folder')
    parser.add_argument('-c', '--csv_file', default=None, help='CSV file containing preprocessing data for custom datasets')
    parser.add_argument('-n', '--name', required=True, help='Name of dataset to be processed')
    parser.add_argument('-s', '--start', default=0, type=int, help='individual in dataset to start from (if start = 0 will start from the first person in the dataset, if 10 will start from the 11th)')
    parser.add_argument('-e', '--end', default=-1, type=int, help="individual in dataset to stop at (if end = 10 will end at the 10th person in the dataset)")
    parser.add_argument('--out_spacing', default="1.,1.,3.", type=str, help="output spacing used in the resampling process. Provide as a string of comma separated floats, no spaces, i.e '1.,1.,3.")
    parser.add_argument('-f', '--force_replace', default="False", type=str, help="if true, files that already exist in their target preproessed form will be overwritten (set to true if a new preprocessing protocol is devised, otherwise leave false for efficiency)")
    parser.add_argument('-z', '--skip_if_any', default="False", type=str, help="if true, skips an individual if a single post processed file for that individual is found (useful when running across multiple machines to same output folder")
    parser.add_argument('-a', '--add_dsname_to_folder_name', default="True", type=str)

    return parser


def main(args):
    
    # the new simplified preprocessing pipeline will go as folows.
    # preprocess the t1
    # then the flair
    # then anything else.
    
    # ======================================================================================
    # SETUP PREPROCESSING PIPELINE
    # ======================================================================================
    
    # get the parser that maps inputs to outputs
    # csv file used for custom datasets
    parser = select_parser(args.name, args.in_dir, args.out_dir, args.csv_file, args.add_dsname_to_folder_name.lower() == "true")
    
    # get the files to be processed
    files_map = parser.get_dataset_inout_map()
    keys = natsorted(list(files_map.keys()))
    
    # select range of files to preprocess
    if args.end == -1:
        keys = keys[args.start:]
    else:
        keys = keys[args.start:args.end]
        
    print(f"starting at individual {args.start} and ending at individual {args.end}")
    
    # get the fsl directory used for brain extraction and bias field correction
    FSLDIR = os.getenv('FSLDIR')
    if 'FSLDIR' == "":
        raise ValueError("FSL is not installed. Install FSL to complete brain extraction")
        
    # parse the outspacing argument
    outspacing = [float(x) for x in args.out_spacing.split(",")]
    if len(outspacing) != 3: # 3D
        raise ValueError(f"malformed outspacing parameter: {args.out_spacing}")
    else:
        print(f"using out_spacing: {outspacing}")
            
    
    # ======================================================================================
    # RUN
    # ======================================================================================
    for ind in keys:
        ### initializing the preprocessing
        any_found = False
        print(f"processing individual: {ind}")
        ind_filemap = files_map[ind]
        
        # check whether all individuals have been done and can therefore be skipped
        can_skip = True
        if not args.force_replace.lower() == "true":
            for filetype in files_map[ind].keys():
                output_dir = ind_filemap[filetype]['outpath']
                output_filename = ind_filemap[filetype]['outfilename']

                if output_filename != None and not os.path.exists(os.path.join(output_dir, output_filename + FORMAT)):
                    can_skip = False
                    break
                else:
                    any_found = True
        else:
            can_skip = False

        # a temp file created to tell other threads that this individual is being processed
        # used when the z flag is true.
        tmp_file = os.path.join(output_dir, f"{ind}_temp.txt")
        if os.path.exists(tmp_file):
            any_found = True
            
        can_skip = can_skip or (any_found and args.skip_if_any.lower() == "true")
        if can_skip:
            print(f"skipping, because preprocessed individual {ind} file exists and force_replace set to false")
            continue
        
        # create the output directory if it does not exist
        if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
        # setting the temp file
        with open(tmp_file, "w") as f:
            f.write(f"processing {ind}")
            
            
        ### run the preprocessing
        def get_filetype_data(files_data):
            infile = files_data['infile']
            output_dir = files_data['outpath']
            output_filename = files_data['outfilename']
            islabel = files_data['islabel']
            
            print(f"processing file: {infile}")
        
            # check that the file exists
            if not os.path.exists(infile):
                raise ValueError(f"target file doesn't exist: {infile}")
                
            return infile, output_dir, output_filename, islabel
            
        
        ### process the t1: resample, bias correct, skull extract, normalize
        # load info
        infile, output_dir, output_filename, islabel = get_filetype_data(ind_filemap["T1"])
        outfile = os.path.join(output_dir, output_filename)
        # resample
        resample_and_save(infile, out_file+ "_resamped_"+FORMAT, is_label=islabel, out_spacing=outspacing, overwrite=True)
        # bias correct
        bias_field_corr_command = [os.path.join(*[FSLDIR,'bin', 'fast']), '-b', '-B', out_file+ "_resamped_"+FORMAT]
        _ = subprocess.call(bias_field_corr_command)
        # skull strip (takes t1 path, out path, mask out path)
        mask_outfile = out_file + "BET_mask"+FORMAT
        skull_strip_and_save(out_file+"_resamped_"+"_restore.nii.gz"+FORMAT, out_file + "_skull_extracted_"+FORMAT, mask_outfile)
        # normalize
        normalize(out_file + "_skull_extracted_"+FORMAT, out_file+FORMAT)
        print("outfile post normalize: ", out_file)
        
        ### process the flair: resample, skull extract, normalize
        # load info
        infile, output_dir, output_filename, islabel = get_filetype_data(ind_filemap["FLAIR"])
        outfile = os.path.join(output_dir, output_filename)
        # resample
        resample_and_save(infile, out_file+ "_resamped_"+FORMAT, is_label=islabel, out_spacing=outspacing, overwrite=True)
        # skull extract
        apply_mask_and_save(out_file+ "_resamped_"+FORMAT, mask_outfile, out_file+ "_skull_extracted_"+FORMAT)
        # normalize
        normalize(out_file + "_skull_extracted_"+FORMAT, out_file+FORMAT)
        print("outfile post normalize: ", out_file)
        
        ### process any labels: resample
        for key in ind_filemap.keys():
            if key in ["T1", "FLAIR"]:
                continue
            
            # load info
            infile, output_dir, output_filename, islabel = get_filetype_data(ind_filemap[key])
            outfile = os.path.join(output_dir, output_filename)
            
            if not islabel:
                print(f"skipping {key} because it is not a label")
                continue
            
            # resample
            resample_and_save(infile, out_file+ "_resamped_"+FORMAT, is_label=islabel, out_spacing=outspacing, overwrite=True)
        
if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
            
    
