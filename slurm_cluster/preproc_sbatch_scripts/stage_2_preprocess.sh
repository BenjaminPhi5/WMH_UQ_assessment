#!/bin/bash
# ====================
# Options for sbatch
# ====================

# Location for stdout log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --output=/home/%u/slurm_logs/slurm-%A_%a.out

# Location for stderr log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --error=/home/%u/slurm_logs/slurm-%A_%a.out

# Maximum number of nodes to use for the job
# #SBATCH --nodes=1

# Megabytes of RAM required. Check `cluster-status` for node configurations
#SBATCH --mem=4000

# Number of CPUs to use. Check `cluster-status` for node configurations
#SBATCH --cpus-per-task=2

# Maximum time for the job to run, format: days-hours:minutes:seconds
#SBATCH --time=04:00:00

# =====================
# Logging information
# =====================

# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"


# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Make script bail out after first error
set -e

# Activate your conda environment
CONDA_ENV_NAME=ip
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}


# rsync the data
dfs_wmh_in=/home/${USER}/ipdis/data/preprocessed_data/WMH_challenge_dataset
dfs_ed_in=/home/${USER}/ipdis/data/preprocessed_data/EdData
dfs_out=/home/${USER}/ipdis/data/preprocessed_data/collated

mkdir -p ${dfs_out}

in_dir=/disk/scratch/${USER}/ipdis/preprep/out_data
out_dir=/disk/scratch/${USER}/ipdis/preprep/out_data/collated

in_dir_wmhchal=${in_dir}/WMH_challenge_dataset
out_dir_wmhchal=${out_dir}/WMH_challenge_dataset

in_dir_ed=${in_dir}/EdData
out_dir_ed=${out_dir}/EdData


# width and height of centre crop in second stage of preprocessing
H=224
W=160

program=/home/${USER}/ipdis/Trustworthai-MRI-WMH/twaidata/FileCollation_preprep/preprocess_file_collation.py

# rsync over the data
rsync --archive --update --compress --progress ${dfs_wmh_in}/ ${in_dir_wmhchal}

# stage two preprocessing - collate each domain into a single file for all imgs and one for all labels
echo preprocessing Singapore
python ${program} -i ${in_dir} -o ${out_dir_wmhchal} -n WMH_challenge_dataset -d Singapore -H ${H} -W ${W}

echo preprocessing Utrecht
python ${program} -i ${in_dir} -o ${out_dir_wmhchal} -n WMH_challenge_dataset -d Utrecht -H ${H} -W ${W}

echo preprocessing GE3T
python ${program} -i ${in_dir} -o ${out_dir_wmhchal} -n WMH_challenge_dataset -d GE3T -H ${H} -W ${W}

# rsync back the results
rsync --archive --update --compress --progress ${out_dir_wmhchal}/ ${dfs_out}/

echo FINISHED WMH CHALLENGE

# rsync over the data
rsync --archive --update --compress --progress ${dfs_ed_in} ${in_dir_ed}

echo preprocessing domain A
python ${program} -i ${in_dir} -o ${out_dir_ed} -n EdData -d domainA -H ${H} -W ${W}

echo preprocessing domain B
python ${program} -i ${in_dir} -o ${out_dir_ed} -n EdData -d domainB -H ${H} -W ${W}

echo preprocessing domain C
python ${program} -i ${in_dir} -o ${out_dir_ed} -n EdData -d domainC -H ${H} -W ${W}

echo preprocessing domain D
python ${program} -i ${in_dir} -o ${out_dir_ed} -n EdData -d domainD -H ${H} -W ${W}

# rsync back the results
rsync --archive --update --compress --progress ${out_dir_ed}/ ${dfs_out}/

echo FINISHED ED DATA