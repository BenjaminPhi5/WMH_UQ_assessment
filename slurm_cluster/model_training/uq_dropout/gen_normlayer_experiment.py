
"""
script for training a 2D Unet, one with batch norm, one with instance norm, and one with group norm
"""

import os

USER = os.getenv('USER')

SCRIPT_PATH = f"/home/{USER}/ipdis/Trustworthai-MRI-WMH/trustworthai/models/uq_models/train_UQ_unet.py"
SLURM_SCRIPT_PATH = f"/home/{USER}/ipdis/Trustworthai-MRI-WMH/slurm_cluster/model_training/uq_dropout/"

MAX_EPOCHS = 400

def exp_call(dropout_type, dropconnect_type, dropout_p, dropconnect_p, norm_type, 
            use_multidim_dropout, use_multidim_dropconnect, gn_groups):
    return "".join((
        f"python {SCRIPT_PATH} ",
        f"--dropout_type {dropout_type} " if dropout_type else "",
        f"--dropconnect_type {dropconnect_type} " if dropconnect_type else "",
        f"--dropout_p {dropout_p} " if dropout_p else "",
        f"--dropconnect_p {dropconnect_p} " if dropconnect_p else "",
        f"--norm_type {norm_type} " if norm_type else "",
        f"--use_multidim_dropout {use_multidim_dropout} " if use_multidim_dropconnect != None else "",
        f"--use_multidim_dropconnect {use_multidim_dropconnect} " if use_multidim_dropconnect != None else "",
	f"--gn_groups {gn_groups} " if gn_groups != None else "",
        f"--max_epochs {MAX_EPOCHS} "
    ))

def main():
    # gen experiment file
    output_file = open(os.path.join(SLURM_SCRIPT_PATH, "uq_normlayer_experiment.txt"), "w")
    
    # bernoulli dropconnect
    call = exp_call(None, None, None, None, 'bn', None, None, None)
    print(call, file=output_file)

    call = exp_call(None, None, None, None, 'in', None, None, None)
    print(call, file=output_file)

    call = exp_call(None, None, None, None, 'gn', None, None, 4)
    print(call, file=output_file)

    output_file.close()
    
if __name__ == '__main__':
    main()
        
