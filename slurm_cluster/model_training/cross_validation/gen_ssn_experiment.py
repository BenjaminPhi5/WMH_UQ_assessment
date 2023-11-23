"""Script for generating experiments.txt"""
import os
import numpy as np

# The home dir on the node's scratch disk
USER = 's2208943'
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = '/disk/scratch_big'  
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

base_call = (f"python /home/s2208943/ipdis/UQ_WMH_methods/trustworthai/run/train/example_script/ssn_train.py ")
base_name = "ssn_cross_validate_32"

args = {
    "ckpt_dir": f"{SCRATCH_HOME}/results/cross_validated_models/",
    "dataset": "ed",
    "seed": 3407,
    "test_split": 0.15,
    "val_split": 0.15,
    "empty_slice_retention": 0.1,
    "ssn_rank": 15,
    "ssn_epsilon": 1e-05,
    "ssn_mc_samples": 10,
    "ssn_sample_dice_coeff": 0.05,
    "ssn_pre_head_layers": 32,
    "dice_factor": 5,
    "xent_factor": 0.01,
    "dice_empty_slice_weight": 0.5,
    "lr": 0.0002,
    "dropout_p": 0.0,
    "max_epochs": 200,
    "early_stop_patience": 15,
    "weight_decay": 0.0001,
    "batch_size": 12,
    "cross_validate": True,
    "cv_test_fold_smooth":1,
    "overwrite":True,
}
args_string = " ".join([f"--{key}={value}" for key, value in args.items()])
ensemble_size = 10


output_file = open("experiment.txt", "w")

splits = int(np.floor(1/args['val_split']))

for ens in range(ensemble_size):
    for split in range(splits):
        expt_call = (
            f"{base_call} "
            f"{args_string} "
            f"--cv_split {split} "
            f"--model_name {base_name}_split{split}_ens{ens} "
        )
        print(expt_call, file=output_file)

output_file.close()