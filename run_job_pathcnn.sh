#!/bin/bash
#
#SBATCH --job-name=train
##SBATCH --gres=gpu:p100:1
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --mem=15GB
#SBATCH --output=outputs/%A.out
#SBATCH --error=outputs/%A.err


##### above is for on nyulmc hpc: bigpurple #####
##### below is for on nyu hpc: prince #####
##!/bin/bash
##SBATCH --partition=gpu8_long
##SBATCH --job-name=train_PCNN
##SBATCH --gres=gpu:1
##SBATCH --output=outputs/rq_train1_%A_%a.out
##SBATCH --error=outputs/rq_train1_%A_%a.err
##SBATCH --mem=200GB
 

module purge
module load python3/intel/3.5.3
module load pytorch/python3.5/0.2.0_3
module load torchvision/python3.5/0.1.9


echo "Starting at `date`"
echo "Job name: $SLURM_JOB_NAME JobID: $SLURM_JOB_ID"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."


#nparam="--cuda --calc_val_auc  --augment --dropout=0.1  --imgSize=227 --model_type=alexnet --root_dir=/beegfs/sb3923/DeepCancer/alldata/LungTilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/LungTilesSorted/Lung_FileMappingDict.p" 
#nparam="--cuda --calc_val_auc  --augment --dropout=0.1  --imgSize=224 --model_type=vgg16 --root_dir=/beegfs/sb3923/DeepCancer/alldata/LungTilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/LungTilesSorted/Lung_FileMappingDict.p"
#nparam="--cuda --calc_val_auc  --augment --dropout=0.1 --imgSize=227 --model_type=alexnet --root_dir=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds1TilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds1_FileMappingDict.p"
#nparam="--cuda --calc_val_auc  --augment --dropout=0.1  --imgSize=224 --model_type=vgg16 --root_dir=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds1TilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds1_FileMappingDict.p"
#nparam="--cuda --calc_val_auc  --augment --dropout=0.1  --imgSize=227 --model_type=alexnet --root_dir=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds2TilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds2_FileMappingDict.p"
#nparam="--cuda --calc_val_auc  --augment --dropout=0.1  --imgSize=224 --model_type=vgg16 --root_dir=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds2TilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds2_FileMappingDict.p"
#nparam="--cuda --calc_val_auc  --augment --dropout=0.1  --imgSize=227 --model_type=alexnet --root_dir=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds3TilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds3_FileMappingDict.p"
#nparam="--cuda --calc_val_auc  --augment --dropout=0.1  --imgSize=224 --model_type=vgg16 --root_dir=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds3TilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds3_FileMappingDict.p"
nparam="--cuda --calc_val_auc  --augment --dropout=0.1 --root_dir=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds1TilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds1_FileMappingDict.p"

#nexp="/scratch/sb3923/PathCNN_data/experiments/train_full_alexnet"
#nexp="/scratch/sb3923/PathCNN_data/experiments/train_full_vgg16"
#nexp="/scratch/sb3923/PathCNN_data/experiments/train_ds1_alexnet"
#nexp="/scratch/sb3923/PathCNN_data/experiments/train_ds1_vgg16"
#nexp="/scratch/sb3923/PathCNN_data/experiments/train_ds2_alexnet"
#nexp="/scratch/sb3923/PathCNN_data/experiments/train_ds2_vgg16"
#nexp="/scratch/sb3923/PathCNN_data/experiments/train_ds3_alexnet"
#nexp="/scratch/sb3923/PathCNN_data/experiments/train_ds3_vgg16"
nexp="/scratch/sb3923/PathCNN_data/experiments/train_ds1_pathcnn"

#output="/scratch/sb3923/PathCNN_data/logs/train_full_alexnet.log" 
#output="/scratch/sb3923/PathCNN_data/logs/train_full_vgg16.log"
#output="/scratch/sb3923/PathCNN_data/logs/train_ds1_alexnet.log"
#output="/scratch/sb3923/PathCNN_data/logs/train_ds1_vgg16.log"
#output="/scratch/sb3923/PathCNN_data/logs/train_ds2_alexnet.log"
#output="/scratch/sb3923/PathCNN_data/logs/train_ds2_vgg16.log"
#output="/scratch/sb3923/PathCNN_data/logs/train_ds3_alexnet.log"
#output="/scratch/sb3923/PathCNN_data/logs/train_ds3_vgg16.log"
output="/scratch/sb3923/PathCNN_data/logs/train_ds1_pathcnn.log"

python3 -u /scratch/sb3923/PathCNN/train.py $nparam --experiment $nexp > $output
