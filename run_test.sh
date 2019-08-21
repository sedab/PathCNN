#!/bin/bash
#SBATCH --partition=gpu8_medium
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --job-name=test_PCNN
#SBATCH --gres=gpu:4
#SBATCH --output=outputs/rq_train1_%A_%a.out
#SBATCH --error=outputs/rq_train1_%A_%a.err
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

nparam="--model=epoch_2.pth --root_dir=/gpfs/data/abl/deepomics/tsirigoslab/histopathology/Tiles/LngTilesSorted/ --num_class=3 --tile_dict_path=/gpfs/data/abl/deepomics/tsirigoslab/histopathology/Tiles/Lng_FileMappingDict.p --val=test"

nexp="/gpfs/scratch/bilals01/test-repo/experiments/exp2"

output="/gpfs/scratch/bilals01/test-repo/logs/log_test_single_class.log" 

python3 -u /gpfs/scratch/bilals01/test-repo/PathCNN/test.py $nparam --experiment $nexp > $output
