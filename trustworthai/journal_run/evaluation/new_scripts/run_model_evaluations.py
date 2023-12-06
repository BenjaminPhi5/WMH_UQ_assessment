import argparse
import os
import subprocess

def construct_parser():
    parser = argparse.ArgumentParser(description='Process hyperparameters file and execute main script')
    parser.add_argument('--repo_dir', default='/home/s2208943/ipdis/WMH_UQ_assessment', type=str)
    parser.add_argument('--result_dir', default=None, type=str)
    parser.add_argument('--ckpt_dir', default=None, type=str)
    parser.add_argument('--eval_split', default='val', type=str)
    parser.add_argument('--script_loc', default='trustworthai/journal_run/evaluation/new_scripts/stochastic_model_basic_eval.py', type=str)
    parser.add_argument('--dataset', default="ed", type=str)
    return parser

def execute_model_evaluation(hyperparameters_file, args):
    # Read hyperparameters from file
    hyperparameters = {}
    with open(hyperparameters_file, 'r') as fp:
        firstline = True
        for line in fp:
            if firstline:
                firstline = False # the first line contains the path to the best checkpoint. The rest are parameters.
                continue
            if line.strip():
                    key, value = line.strip().split(': ')
                    hyperparameters[key] = value

    # Create command to execute main script
    script = os.path.join(args.repo_dir, args.script_loc)
    print(script)
    command = ['python', script, f'--repo_dir={args.repo_dir}', f'--result_dir={args.result_dir}', f'--eval_split={args.eval_split}', f'--ckpt_dir={args.ckpt_dir}', f'--dataset={args.dataset}']

    keys_to_ignore = [
        'ckpt_dir', 'dataset'
    ]

    for key, value in hyperparameters.items():
        if key not in keys_to_ignore:
            command.append(f'--{key}={value}')

    # Execute the command
    print(" ".join(command))
    subprocess.call(command)

def main(args):
    models = os.listdir(args.ckpt_dir)
    for m in models:
        file = os.path.join(os.path.join(args.ckpt_dir, m), "best_ckpt.txt")
        execute_model_evaluation(file, args)
        

if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)


