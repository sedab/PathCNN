#!/bin/bash
#SBATCH --partition=gpu8_medium
#SBATCH --job-name=train_PCNN
#SBATCH --gres=gpu:1
##SBATCH --output=outputs/rq_train1_%A_%a.out
##SBATCH --error=outputs/rq_train1_%A_%a.err
#SBATCH --mem=200GB


# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
cluster='bigpurple'

# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel:
ssh -N -L ${port}:${node}:${port} ${user}@${cluster}.nyumc.org

For more info and how to connect from windows,
  see research.computing.yale.edu/jupyter-nb
Here is the MobaXterm info:

Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: ${cluster}.nyumc.org
SSH login: $user
SSH port: 22
https://research.computing.yale.edu/support/hpc/guides/running-jupyter-notebooks-clusters
Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

#install

# load modules or conda environments here
# e.g. farnam:
module purge
module load anaconda3/gpu/5.2.0
module load cuda91/toolkit/9.1.85
module load miniconda3/4.5.1
python3 -m pip install pybind11 --user
#python3 -m pip install '/gpfs/share/apps/miniconda3/4.5.1/lib/python3.6/site-packages/fasttext_pybind.cpython-36m-x86_64-linux-gnu.so' --user
module load python/gpu/3.6.5


source activate myenv
# DON'T USE ADDRESS BELOW.
# DO USE TOKEN BELOW
jupyter-notebook --no-browser --port=${port} --ip=${node} --NotebookApp.token=''

