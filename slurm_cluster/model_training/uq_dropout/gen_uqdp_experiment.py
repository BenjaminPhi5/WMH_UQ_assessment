
"""
script for generating the uq using dropout and dropconnect model training experiment
"""

import os

USER = os.getenv('USER')

SCRIPT_PATH = f"/home/{USER}/ipdis/Trustworthai-MRI-WMH/trustworthai/models/uq_models/train_UQ_unet.py"
SLURM_SCRIPT_PATH = f"/home/{USER}/ipdis/Trustworthai-MRI-WMH/slurm_cluster/model_training/uq_dropout/"

MAX_EPOCHS = 400

def exp_call(dropout_type, dropconnect_type, dropout_p, dropconnect_p, norm_type, 
            use_multidim_dropout, use_multidim_dropconnect):
    return "".join((
        f"python {SCRIPT_PATH} ",
        f"--dropout_type {dropout_type} " if dropout_type else "",
        f"--dropconnect_type {dropconnect_type} " if dropconnect_type else "",
        f"--dropout_p {dropout_p} " if dropout_p else "",
        f"--dropconnect_p {dropconnect_p} " if dropconnect_p else "",
        f"--norm_type {norm_type} " if norm_type else "",
        f"--use_multidim_dropout {use_multidim_dropout} " if use_multidim_dropconnect != None else "",
        f"--use_multidim_dropconnect {use_multidim_dropconnect} " if use_multidim_dropconnect != None else "",
        f"--max_epochs {MAX_EPOCHS} "
    ))

def main():
    # gen experiment file
    output_file = open(os.path.join(SLURM_SCRIPT_PATH, "uq_dropout_experiment.txt"), "w")
    
    """
    want to do: bernoulli dropconnect
                gaussian drop connect
                bernoulli dropout
                gaussian dropout
                spike and slab dropout
                'slab and spike' dropout?
    
    do each for p = 0.05, 0.1, 0.2, 0.3, 0.4
    and for standard and 2D (multidim flag set) versions
    """
    
    ps = [0.05, 0.1, 0.2, 0.3, 0.4]
    multidim = [True, False]
    
    
    for p in ps:
        for mdim in multidim:
            # bernoulli dropconnect
            call = exp_call(None, "bernoulli", None, p, 'bn', mdim, mdim)
            print(call)
            print(call, file=output_file)

            # gaussian dropconnect
            call = exp_call(None, "gaussian", None, p, 'bn', mdim, mdim)
            print(call, file=output_file)

            # bernoulli dropout
            call = exp_call("bernoulli", None, p, None, 'bn', mdim, mdim)
            print(call, file=output_file)

            # gaussian dropout
            call = exp_call("gaussian", None, p, None, 'bn', mdim, mdim)
            print(call, file=output_file)

            # spike and slab dropout
            call = exp_call("bernoulli", "gaussian", p, p, 'bn', mdim, mdim)
            print(call, file=output_file)

            # slab and spike dropout
            call = exp_call("gaussian", "bernoulli", p, p, 'bn', mdim, mdim)
            print(call, file=output_file)
    
    output_file.close()
    
if __name__ == '__main__':
    main()
        
