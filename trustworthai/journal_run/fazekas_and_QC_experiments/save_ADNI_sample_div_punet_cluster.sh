#!/bin/bash

python ../evaluation/new_scripts/run_model_evaluations.py --repo_dir=/home/s2208943/ipdis/WMH_UQ_assessment --result_dir=/home/s2208943/ipdis/data/preprocessed_data/ADNI_300_output_maps --ckpt_dir=/home/s2208943/ipdis/results/journal_models/cross_validated_models --script_loc=trustworthai/journal_run/fazekas_and_QC_experiments/saving_ADNI_sample_div_data.py --uncertainty_type=punet --model_name=punet_cv0