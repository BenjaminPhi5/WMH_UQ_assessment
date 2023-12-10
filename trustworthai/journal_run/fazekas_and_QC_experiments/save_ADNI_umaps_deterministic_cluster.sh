#!/bin/bash

python ../evaluation/new_scripts/run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --result_dir=/home/s2208943/ipdis/data/preprocessed_data/ADNI_300_output_maps --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --script_loc=trustworthai/journal_run/fazekas_and_QC_experiments/saving_ADNI_umaps_script.py --uncertainty_type=deterministic --model_name=ens0_cv0

# parser.add_argument('--repo_dir', default='/home/s2208943/ipdis/WMH_UQ_assessment', type=str)
#     parser.add_argument('--result_dir', default=None, type=str)
#     parser.add_argument('--ckpt_dir', default=None, type=str)
#     parser.add_argument('--eval_split', default='val', type=str)
#     parser.add_argument('--script_loc', default='trustworthai/journal_run/evaluation/new_scripts/stochastic_model_basic_eval.py', type=str)
#     parser.add_argument('--dataset', default="ed", type=str)
#     parser.add_argument('--overwrite', default="false", type=str)
#     parser.add_argument('--uncertainty_type', default='deterministic', type=str)
#     parser.add_argument('--eval_sample_num', default=10, type=int)
#     parser.add_argument('--model_name', default="", type=str)