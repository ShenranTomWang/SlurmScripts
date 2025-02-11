#!/bin/bash

#SBATCH --job-name=my_jupyter_notebook
#SBATCH --account=st-jzhu71-1-gpu
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --constraint=gpu_mem_32

################################################################################

export NOTEBOOK_HOME_DIR="/scratch/st-jzhu71-1/shenranw"

# Change directory into the job dir
cd $SLURM_SUBMIT_DIR

# Load software environment
module load cuda/12.4.0 intel-oneapi-compilers/2023.1.0 python/3.11.6 gcc
export ENVDIR=/scratch/st-jzhu71-1/shenranw/envs/CoT     # change accordingly
source $ENVDIR/bin/activate
export TRITON_CACHE_DIR="/scratch/st-jzhu71-1/shenranw/triton_cache"
export HF_HOME="/scratch/st-jzhu71-1/shenranw/transformers_cache"

# Set RANDFILE location to writeable dir
export RANDFILE=$TMPDIR/.rnd

# Generate a unique token (password) for Jupyter Notebooks
export APPTAINERENV_JUPYTER_TOKEN=$(openssl rand -base64 15)

# Find a unique port for Jupyter Notebooks to listen on
readonly PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# Print connection details to file
cat > ./connection_${SLURM_JOB_ID}.txt <<END

1. Create an SSH tunnel to Jupyter Notebooks from your local workstation using the following command:

ssh -N -L 8888:${HOSTNAME}:${PORT} ${USER}@sockeye.arc.ubc.ca

2. Point your web browser to http://localhost:8888

3. Login to Jupyter Notebooks using the following token (password):

${APPTAINERENV_JUPYTER_TOKEN}

When done using Jupyter Notebooks, terminate the job by:

1. Quit or Logout of Jupyter Notebooks
2. Issue the following command on the login node (if you did Logout instead of Quit):

scancel ${SLURM_JOB_ID}

END

# Execute jupyter within the Apptainer container
jupyter notebook --no-browser --port=${PORT} --ip=0.0.0.0 --notebook-dir=$NOTEBOOK_HOME_DIR
