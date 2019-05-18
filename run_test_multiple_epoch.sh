#!/bin/bash
#
#BATCH --mail-type=ALL
#SBATCH --mail-user bz957r@nyumc.org
#
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --partition=gpu4_long
#SBATCH --gres=gpu:1
#
# project id
#SBATCH --job-name=full
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --output=outputs/rq_train_%A.out
#SBATCH --error=outputs/rq_train_%A.err


echo "Starting at `date`"
echo "Job name: $SLURM_JOB_NAME JobID: $SLURM_JOB_ID"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."

echo "experiment:"
echo $1
echo $2
echo $3


module purge
#module load python3/intel/3.5.3 pytorch/python3.5/0.2.0_3 torchvision/python3.5/0.1.9
module load python/gpu/3.6.5


cd /gpfs/data/abl/deepomics/tsirigoslab/histopathology/Repos/deep-cancer

# check if next checkpoint available
declare -i count=0
declare -i step=1

while true; do
    echo count
    #python3 -u test_ds1.py --data 'lung' --experiment 'tr_lung_full'  --model "epoch_$count.pth" 
    #python3 -u test_ds1.py --data 'lung' --experiment 'train_lung_full_step_'  --model "step_$count.pth" 
    #python3 -u test_ds1.py --data 'lung' --experiment 'train_lung_no_aug'  --model "epoch_$count.pth" 
    #python3 -u test_ds1.py --data 'lung' --experiment 'train_augmentation_analysis'  --model "step_$count.pth" 
    python3 -u test_ds1.py --data 'lung' --experiment 'full'  --model "epoch_$count.pth" 
    #python3 -u test_ds1.py --data 'lung' --experiment 'full_no_aug'  --model "step_$count.pth" 
    count=`expr "$count" + "$step"`
done

