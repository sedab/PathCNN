#!/bin/bash
#
#SBATCH --job-name=train
#SBATCH --gres=gpu:p40:1
##SBATCH --gres=gpu:v100:1
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=5
#SBATCH --time=60:00:00
#SBATCH --mem=100GB
#SBATCH --output=outputs/train_%A.out
#SBATCH --error=outputs/train_%A.err

##### below is for on nyulmc hpc: bigpurple #####
##### above is for on nyu hpc: prince #####

##!/bin/bash
##SBATCH --partition=gpu8_medium
##SBATCH --ntasks=8
##SBATCH --cpus-per-task=1
##SBATCH --job-name=train_PCNN
##SBATCH --gres=gpu:1
##SBATCH --output=outputs/rq_train1_%A_%a.out
##SBATCH --error=outputs/rq_train1_%A_%a.err
##SBATCH --mem=250GB

#module purge
#module load python/gpu/3.6.5

module purge
module load python3/intel/3.5.3
module load pytorch/python3.5/0.2.0_3
module load torchvision/python3.5/0.1.9

echo "Starting at `date`"
echo "Job name: $SLURM_JOB_NAME JobID: $SLURM_JOB_ID"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."


exp_name="lung_x10_1024_tr" 
model="7layers" #3layers, 4layers, 5layers, 7layers_v1, 7layers_v2, 6layers, 8 layers, resnet18, alexnet, vgg16
im_size="1024" #224, 299
lls=14400 #1024-14400, 512-3136, 256-576

echo "$exp_name"
#--optimizer=SGD
#--niter=1
#lung
#nparam="--cuda --calc_val_auc --augment  --init=xavier --dropout=0.1 --imgSize=${im_size} --nonlinearity=leaky --model_type=${model} --root_dir=/beegfs/sb3923/DeepCancer/alldata/LungTilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/LungTilesSorted/Lung_FileMappingDict.p" 
#lung 5x
#nparam="--cuda --calc_val_auc  --augment --init=xavier --dropout=0.1  --imgSize=${im_size} --nonlinearity=leaky --model_type=${model} --root_dir=/beegfs/sb3923/DeepCancer/alldata/Lung5xTilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/Lung5xTilesSorted/Lung5x_FileMappingDict.p"
#kidney
#nparam="--cuda --calc_val_auc  --augment --init=xavier --dropout=0.1 --imgSize=${im_size} --nonlinearity=leaky --model_type=${model}  --root_dir=/beegfs/sb3923/DeepCancer/alldata/KidneyTilesSorted/ --num_class=4 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/KidneyTilesSorted/Kidney_FileMappingDict.p"
#breast
#nparam="--cuda --calc_val_auc  --augment --init=xavier --dropout=0.1 --imgSize=${im_size} --nonlinearity=leaky --model_type=${model}  --root_dir=/beegfs/sb3923/DeepCancer/alldata/BreastTilesSorted/ --num_class=2 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/BreastTilesSorted/Breast_FileMappingDict.p"

#breast subtypes
#nparam="--cuda --calc_val_auc  --augment --init=xavier --dropout=0.1 --imgSize=${im_size} --nonlinearity=leaky --model_type=${model} --root_dir=/beegfs/sb3923/DeepCancer/alldata/BreastSubtypes_basal_vs_rest_no_normal/BreastTilesSorted/ --num_class=2 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/BreastSubtypes_basal_vs_rest_no_normal/Breast_FileMappingDict.p"
#nparam="--cuda --calc_val_auc --augment --init=xavier --dropout=0.1 --imgSize=${im_size} --nonlinearity=leaky --model_type=${model} --root_dir=/beegfs/sb3923/DeepCancer/alldata/BreastSubtypes_no_normal/BreastTilesSorted/ --num_class=4 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/BreastSubtypes_no_normal/Breast_FileMappingDict.p"

#downsampled lung
#ds1
#nparam="--cuda --calc_val_auc --augment  --init=xavier --dropout=0.1 --imgSize=${im_size} --nonlinearity=leaky --model_type=${model} --root_dir=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds1TilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds1_FileMappingDict.p" 
#ds2
#nparam="--cuda --calc_val_auc --augment  --init=xavier --dropout=0.1 --imgSize=${im_size} --nonlinearity=leaky --model_type=${model} --root_dir=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds2TilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds2_FileMappingDict.p" 
#ds3
#nparam="--cuda --calc_val_auc --augment  --init=xavier --dropout=0.1 --imgSize=${im_size} --nonlinearity=leaky --model_type=${model} --root_dir=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds3TilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds3_FileMappingDict.p" 

#1024 lung with different magnification
#x5
#nparam="--last_layer_size=${lls} --cuda --calc_val_auc --augment  --init=xavier --dropout=0.1 --imgSize=${im_size} --nonlinearity=leaky --model_type=${model} --root_dir=/beegfs/sb3923/DeepCancer/alldata/Lungx1024x5TilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/Lungx1024x5TilesSorted/Lungx1024x5_FileMappingDict.p" 
#x10
nparam="--last_layer_size=${lls} --cuda --calc_val_auc --augment  --init=xavier --dropout=0.1 --imgSize=${im_size} --nonlinearity=leaky --model_type=${model} --root_dir=/beegfs/sb3923/DeepCancer/alldata/Lungx1024x10TilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/Lungx1024x10TilesSorted/Lungx1024x10_FileMappingDict.p" 
#x20
#nparam="--last_layer_size=${lls} --cuda --calc_val_auc --augment  --init=xavier --dropout=0.1 --imgSize=${im_size} --nonlinearity=leaky --model_type=${model} --root_dir=/beegfs/sb3923/DeepCancer/alldata/Lungx1024x20TilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/Lungx1024x20TilesSorted/Lungx1024x20_FileMappingDict.p" 



nexp="/scratch/sb3923/experiments/${exp_name}"
output="/scratch/sb3923/logs/${exp_name}.log" 

python3 -u /scratch/sb3923/PathCNN/train.py $nparam --experiment $nexp > $output
