#!/bin/bash
#
#SBATCH --job-name=train
##SBATCH --gres=gpu:p100:1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=15GB
#SBATCH --output=outputs/%A.out
#SBATCH --error=outputs/%A.err

module purge
module load python3/intel/3.5.3
module load pytorch/python3.5/0.2.0_3
module load torchvision/python3.5/0.1.9

echo "Starting at `date`"
echo "Job name: $SLURM_JOB_NAME JobID: $SLURM_JOB_ID"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."

nparam="--root_dir=/beegfs/sb3923/DeepCancer/alldata/LungTilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/LungTilesSorted/Lung_FileMappingDict.p --val=test --imgSize=227 --model_type=alexnet"
#nparam="--root_dir=/beegfs/sb3923/DeepCancer/alldata/LungTilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/LungTilesSorted/Lung_FileMappingDict.p --val=valid --imgSize=224 --model_type=vgg16"

#nexp="/scratch/sb3923/PathCNN_data/experiments/train_ds1_alexnet"
#nexp="/scratch/sb3923/PathCNN_data/experiments/train_ds1_vgg16"
#nexp="/scratch/sb3923/PathCNN_data/experiments/train_ds2_alexnet"
#nexp="/scratch/sb3923/PathCNN_data/experiments/train_ds2_vgg16"
nexp="/scratch/sb3923/PathCNN_data/experiments/train2_ds3_alexnet"
#nexp="/scratch/sb3923/PathCNN_data/experiments/train2_ds3_vgg16"
#nexp="/scratch/sb3923/PathCNN_data/experiments/train_full_alexnet"
#nexp="/scratch/sb3923/PathCNN_data/experiments/train_full_vgg16"

out="/scratch/sb3923/PathCNN_data/logs/test"

# check if next checkpoint available
declare -i count=1
declare -i step=1 

while true; do
    echo count
    PathToEpoch="${nexp}/checkpoints/"
    Cmodel="epoch_$count.pth"
    output="${out}/log2_alexnet_ds3_${Cmodel}.log"
    echo $PathToEpoch
    echo $Cmodel
    echo $output
    if [ -f $PathToEpoch/$Cmodel ]; then
    python3 -u test.py  --experiment $nexp  --model $Cmodel $nparam> $output
    else
        break
    fi
    count=`expr "$count" + "$step"`
done
