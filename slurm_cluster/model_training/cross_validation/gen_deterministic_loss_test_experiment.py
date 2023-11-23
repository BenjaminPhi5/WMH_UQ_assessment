"""Script for generating experiments.txt"""
import os
import numpy as np

# The home dir on the node's scratch disk
USER = 's2208943'
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = '/disk/scratch_big'  
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

base_call = (f"python /home/s2208943/ipdis/UQ_WMH_methods/trustworthai/run/train/example_script/deterministic_train.py ")
base_name = "ssn_cross_validate_32"

args = {
    "ckpt_dir": f"{SCRATCH_HOME}/results/cross_validated_models/",
    "dataset": "ed",
    "seed": 3407,
    "test_split": 0.15,
    "val_split": 0.15,
    "empty_slice_retention": 0.1,
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


# ############################################ #
######### SET EXPERIMENT NAME!!!!!!! ###########
# ############################################ #
output_file = open("deterministic_losses_experiment.txt", "w")

split = 0

for loss_name in [
    "dice", "brier", "xent", "tversky", "dice+xent"
]:
    if loss_name == 'tversky':
        model_name_aug = '07'
    else:
        model_name_aug = ''
    expt_call = (
        f"{base_call} "
        f"{args_string} "
        f"--tversky_beta 0.7 "
        f"--loss_name {loss_name} "
        f"--cv_split {split} "
        f"--model_name deterministic_{loss_name}{model_name_aug}_split{split} "
    )
    print(expt_call, file=output_file)
    
loss_name = "tversky"
expt_call = (
    f"{base_call} "
    f"{args_string} "
    f"--tversky_beta 0.9 "
    f"--loss_name {loss_name} "
    f"--cv_split {split} "
    f"--model_name deterministic_tversky09_split{split} "
)
print(expt_call, file=output_file)

output_file.close()