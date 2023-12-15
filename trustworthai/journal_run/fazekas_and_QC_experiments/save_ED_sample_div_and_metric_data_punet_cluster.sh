#!/bin/bash

python ../evaluation/new_scripts/run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --result_dir=/home/s2208943/ipdis/data/preprocessed_data/EdData_output_maps --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --script_loc=trustworthai/journal_run/fazekas_and_QC_experiments/saving_Ed_sample_div_and_metric_data.py --eval_split=test --uncertainty_type=punet --model_name=punet_cv0

python ../evaluation/new_scripts/run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --result_dir=/home/s2208943/ipdis/data/preprocessed_data/EdData_output_maps --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --script_loc=trustworthai/journal_run/fazekas_and_QC_experiments/saving_Ed_sample_div_and_metric_data.py --eval_split=test --uncertainty_type=punet --model_name=punet_cv1

python ../evaluation/new_scripts/run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --result_dir=/home/s2208943/ipdis/data/preprocessed_data/EdData_output_maps --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --script_loc=trustworthai/journal_run/fazekas_and_QC_experiments/saving_Ed_sample_div_and_metric_data.py --eval_split=test --uncertainty_type=punet --model_name=punet_cv2

python ../evaluation/new_scripts/run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --result_dir=/home/s2208943/ipdis/data/preprocessed_data/EdData_output_maps --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --script_loc=trustworthai/journal_run/fazekas_and_QC_experiments/saving_Ed_sample_div_and_metric_data.py --eval_split=test --uncertainty_type=punet --model_name=punet_cv3

python ../evaluation/new_scripts/run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --result_dir=/home/s2208943/ipdis/data/preprocessed_data/EdData_output_maps --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --script_loc=trustworthai/journal_run/fazekas_and_QC_experiments/saving_Ed_sample_div_and_metric_data.py --eval_split=test --uncertainty_type=punet --model_name=punet_cv4

python ../evaluation/new_scripts/run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --result_dir=/home/s2208943/ipdis/data/preprocessed_data/EdData_output_maps --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --script_loc=trustworthai/journal_run/fazekas_and_QC_experiments/saving_Ed_sample_div_and_metric_data.py --eval_split=test --uncertainty_type=punet --model_name=punet_cv5

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