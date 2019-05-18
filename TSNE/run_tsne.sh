#!/bin/bash
#
#SBATCH --job-name=tsne
#SBATCH --time=46:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --output=outputs/tsne_%A.out
#SBATCH --error=outputs/tsne_%A.err

module purge
module load python3/intel/3.5.3
module load pytorch/python3.5/0.2.0_3
module load torchvision/python3.5/0.1.9

cd /scratch/sb3923/deep-cancer
#cd /scratch/jmw784/capstone/deep-cancer

python3 -u tsne.py  > logs/$1_tsne.log

#SBATCH --gres=gpu:1
