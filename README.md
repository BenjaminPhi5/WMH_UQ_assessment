# Trustworthai-MRI-WMH
MRes dissertation project in trustworthy AI for MRI segmentation using uncertainty quantification techniques.

Please note this repository is currently being developed for a different purpose and is rough and ready at the moment.
Please contact me by email if you are having any issues :)

If you wish to make changes please either create a fork, create your own branch, create a pull request. 
I have only tested this on Linux.

### Install the repository
git clone this repository.
To install the twaidata, trustworthai packages and their dependencies, 
`cd` into the repository

`pip install -e .` - don't forget the `.` . I strongly recommend using a conda/virtualenv environment with python >=3.9.


### Preprocessing data

**For preprocessing the WMH challenge dataset.**
Please first download the dataset and unzip all files in the dataset.
You must have the following structure (and ensure the directory name is `WMH_challenge_dataset`:
```
structure of WMH dataset:
WMH_challenge_dataset/    
    public/
        GE3T/
            <id>/
                orig/
                    T1.nii.gz
                    FLAIR.nii.gz
                    ...         (other files, some regression data and '3D' versions)
                pre/
                wmh.nii.gz
        Singapore/
            ...
        Utretch/
            ...
```

**installing FSL**
This preprocessing pipeline uses FSL to perform brain extraction and bias correction.
Please install from the following: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation

**stage 1**
The first preprocessing stage applies brain extraction, resampling and normalizing for FLAIR, T1, label (wmh) files (and any other modalities e.g T2,but these are not present in the WMH dataset).
Bias Correction is also applied for T1.

To run this stage:

```
in_dir= <your path to parent directory of the dataset, (parent of the WMH_challenge_dataset folder)>
out_dir= <your path to folder to store preprocessed datasets>
name=WMH_challenge_dataset # name of the dataset on your filesystem
start=0  # individual to start at
end=-1   
out_spacing=1.,1.,3.      # spacing used during resampling

python twaidata/MRI_preprep/simple_preprocess_st1.py -i ${in_dir} -o ${out_dir} -n ${name} -s ${start} -e ${end} --out_spacing ${out_spacing}
```
Running FSL/bet and FSL/fast takes a long time, I recommend parallising this, which you can do by running the script with different start and end values. s 0 e -1 runs over all individuals, s 0 e 10 does the first 10 etc. There are 60 individuals in total across the WMH challenge dataset.

This will create a folder with the same name as given in the `name` flag inside the `out_dir` folder.


**stage 2**
See the command below. This combines the files of each domain of a dataset into one file. During model training the whole dataset can be easily cached in memory and removes file io during training. By default all images are cropped to 224 by 160 during this process, but this can be changed by setting the `-H` and `-W` parameters. The `-l` flag allows you to extract a specific label (e.g the WMH challenge dataset has 1 for WMH and 2 for other pathology (0=neither)). if `l` is not given then both labels will be extracted, if l is 1 then only the WMH label will be extracted.
Ideally `in_dir` is not the same as `out_dir`. In dir should be the parent folder of your preprocessed WMH dataset.


```
in_dir_wmhchal= <your preprocessed data folder>
out_dir_wmhchal=<your preprocessed data folder>/collated
name=WMH_challenge_dataset

python twaidata/FileCollation_preprep/preprocess_file_collation.py -i ${in_dir_wmhchal} -o ${out_dir_wmhchal} -n ${name} -d Singapore -H 224 -W 160 -l 1
```

This will yield the results of stage two preprocessing in `{out_dir}/${name}`.

**For preprocessing other datasets**
pass a csv file to the first preprocessing script that maps each file in the dataset into an output of the following
format:

```
dataset
    - domain 1
        - imgs
            -<id1>_FLAIR.nii.gz
            -<id1>_T1.nii.gz
            ...
        - labels
            -<id1>_wmh.nii.gz
            -<id1>_stroke.nii.gz (optionally)
            ...
     - domain 2
         ...
      
```

The `FromFileParser` class in `twaidata/mri_dataset_directory_parsers/from_text_file` explains the structure of the csv.



**Running a simple example model**

The train.py script or the Train UNets notebook provide an example of how to train either a 2D or 3D UNet,
using the results of the stage two preprocessing.

An example of a 2D and a 3D model can now be found in the assets folder, and the Evaluate Model notebook
allows you to check the preformance. 
Both should achieve around generalized Dice loss of 0.22 and 0.25 (i.e Dice of ~ 0.78 and 0.75) on the validation and test 
sets respectively.

(2D unet: 0.229 on validation, 0.247 on test)
(3D unet: 0.214 on validatoin, 0.247 on test)

Note that the 3D UNet may be slightly stochastic in its scores as it takes a random selection of 32 contiguous
slices from each image each time it encounters it.

The models are trained with only random flips as augmentation, but there are some other ones you can try 
(and there will be some elastic deformations pushed hopefully at some later time).
