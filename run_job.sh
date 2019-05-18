#!/bin/bash
#
#SBATCH --job-name=charrrr
#SBATCH --gres=gpu:1
#SBATCH --time=47:00:00
#SBATCH --mem=15GB
#SBATCH --output=outputs/%A.out
#SBATCH --error=outputs/%A.err

module purge
module load python3/intel/3.5.3
module load pytorch/python3.5/0.2.0_3
module load torchvision/python3.5/0.1.9
python3 -m pip install comet_ml --user

cd /scratch/sb3923/deep-cancer

python3 -u train.py $1 --experiment $2 > logs/$2.log
