#!/bin/bash
#
#SBATCH --partition=gpu4_medium
#SBATCH --job-name=PathCNN train
#SBATCH --gres=gpu:4
#SBATCH --mem=100GB
#SBATCH --output=outputs/rq_train1_%A_%a.out
#SBATCH --error=outputs/rq_train1_%A_%a.err


module purge
module load python/gpu/3.6.5

echo "Starting at `date`"
echo "Job name: $SLURM_JOB_NAME JobID: $SLURM_JOB_ID"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."

##### above is for on nyulmc hpc: bigpurple #####
##### below is for on nyu hpc: prince #####

##!/bin/bash
##
##SBATCH --job-name=charrrr
##SBATCH --gres=gpu:1
##SBATCH --time=47:00:00
##SBATCH --mem=15GB
##SBATCH --output=outputs/%A.out
##SBATCH --error=outputs/%A.err

#module purge
#module load python3/intel/3.5.3
#module load pytorch/python3.5/0.2.0_3
#module load torchvision/python3.5/0.1.9
#python3 -m pip install comet_ml --user

cd /gpfs/scratch/bilals01/test-repo/PathCNN/

python3 -u train.py $1 --experiment $2 > logs/$2.log
