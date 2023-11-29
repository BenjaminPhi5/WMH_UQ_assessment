#!/bin/bash

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Make script bail out after first error
set -e

# Activate your conda environment
CONDA_ENV_NAME=wmh
echo "Activating conda environment: ${CONDA_ENV_NAME}"
source activate ${CONDA_ENV_NAME}

if [ -d "/disk/scratch_big" ]; then
  SCRATCH_DISK=/disk/scratch_big
else
  SCRATCH_DISK=/disk/scratch
fi

SCRATCH_HOME=${SCRATCH_DISK}/${USER}

mkdir -p ${SCRATCH_HOME}

echo ${SCRATCH_HOME}
