##!/bin/bash
##
##SBATCH --job-name=charrrr
##SBATCH --gres=gpu:1
##SBATCH --time=47:00:00
##SBATCH --mem=15GB
##SBATCH --output=outputs/%A.out
##SBATCH --error=outputs/%A.err

module purge
module load python3/intel/3.5.3
module load pytorch/python3.5/0.2.0_3
module load torchvision/python3.5/0.1.9
python3 -m pip install comet_ml —user

echo "Starting at `date`"
echo "Job name: $SLURM_JOB_NAME JobID: $SLURM_JOB_ID"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."


#sort
python -u /scratch/sb3923/PathCNN/Tiling/0d_SortTiles.py --SourceFolder="/beegfs/sb3923/DeepCancer/alldata/LungTiles" --JsonFile="/beegfs/sb3923/DeepCancer/alldata/LungTiles/LUNG_metadata.cart.2017-10-06T17-57-09.290314.json" --Magnification=20 --MagDiffAllowed=0 --SortingOption=3 --PercentTest=15 --PercentValid=15 --PatientID=12 --nSplit 0

#create dictinory
python3 -u  /scratch/sb3923/PathCNN/Tiling/BuildTileDictionary.py --data  Lung --file_path /beegfs/sb3923/DeepCancer/alldata/
