#!/bin/bash
#SBATCH --partition=gpu8_medium
#SBATCH --job-name=PathCNN_train
#SBATCH --gres=gpu:4
#SBATCH --output=outputs/rq_TSNE_%A_%a.out
#SBATCH --error=outputs/rq_TSNE_%A_%a.err
#SBATCH --mem=100GB

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
#python3 -m pip install comet_ml —user

echo "Starting at `date`"
echo "Job name: $SLURM_JOB_NAME JobID: $SLURM_JOB_ID"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."


module purge
module load python/gpu/3.6.5

cd /gpfs/scratch/bilals01/test-repo/PathCNN/

python3 -u tsne.py $1  > logs/$2_tsne.log

