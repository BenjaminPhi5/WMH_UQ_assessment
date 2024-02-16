import argparse
import zipfile
import os

"""
Script for unzipping the WMH Challenge Dataset. There are spaces and dots in the
filenames of some of the test data, which the linux unzip command didn't like.
I have written this script that renames those files, (which the rest of my preprocessing code will then presume, so I should always use this script to unzip the
data in future.
I also create a separate domain for each different setting for Amsterdam, flattening the file hierarchy by one step for the Amsterdam files.
"""


def get_parser():
    parser = argparse.ArgumentParser(description='unzip WMH challenge data')
    parser.add_argument('--zip_path', type=str, help="path to zip file")
    parser.add_argument('--output_dir', type=str, help="output folder for extraction")

    return parser

def main(args):
    folder_dir = args.output_dir
    zipname = args.zip_path

    z = zipfile.ZipFile(zipname)

    for i, f in enumerate(z.filelist):
        print(f.filename)
        
        if " " in f.filename:
            # rectify the folder_path of the filename prior to extraction
            path_components = f.filename.split(os.path.sep)
            folder_path = os.path.sep.join(path_components[:-1])
            folder_path = folder_path.replace(' ', '_').replace('.','')
            name = path_components[-1]
            
            f.filename = f"{folder_path}{os.path.sep}{name}"

        if "Amsterdam" in f.filename:
            f.filename = f.filename.replace("Amsterdam/", "Amsterdam_")
        
            
        z.extract(f, path=folder_dir)

if __name__ =='__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)