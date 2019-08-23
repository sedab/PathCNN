#!/bin/bash
#
#SBATCH --job-name=train
#SBATCH --gres=gpu:p100:1
##SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --mem=15GB
#SBATCH --output=outputs/%A.out
#SBATCH --error=outputs/%A.err

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


module purge
module load python/gpu/3.6.5 

exp_name="exp2"
model="alexnet" #
im_size = "224"

nparam="--cuda --calc_val_auc --augment  --init=xavier --dropout=0.1 --imgSize=${im_size} --nonlinearity=leaky --model_type=${model} --root_dir=/beegfs/sb3923/DeepCancer/alldata/LungTilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/LungTilesSorted/Lung_FileMappingDict.p" 
#kidney
#nparam="--cuda --calc_val_auc  --augment --init=xavier --dropout=0.1 --imgSize=${im_size} --nonlinearity=leaky --model_type=${model}  --root_dir=/beegfs/sb3923/DeepCancer/alldata/KidneyTilesSorted/ --num_class=4 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/KidneyTilesSorted/Kidney_FileMappingDict.p"
#breast
#nparam="--cuda --calc_val_auc  --augment --init=xavier --dropout=0.1 --imgSize=${im_size} --nonlinearity=leaky --model_type=${model}  --root_dir=/beegfs/sb3923/DeepCancer/alldata/BreastTilesSorted/ --num_class=2 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/BreastTilesSorted/Breast_FileMappingDict.p"


#downsampled lung
#ds1
#nparam="--cuda  --augment --dropout=0.1 --imgSize=${im_size} --nonlinearity=leaky --model_type=${model} --init=xavier  --calc_val_auc --root_dir=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds1TilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds1_FileMappingDict.p" 
#ds2
#nparam="--cuda  --augment --dropout=0.1 --imgSize=${im_size} --nonlinearity=leaky --model_type=${model} --init=xavier  --calc_val_auc --root_dir=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds2TilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds2_FileMappingDict.p" 
#ds3
#nparam="--cuda  --augment --dropout=0.1 --imgSize=${im_size} --nonlinearity=leaky --model_type=${model} --init=xavier  --calc_val_auc --root_dir=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds3TilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds3_FileMappingDict.p" 



nexp="/gpfs/scratch/bilals01/test-repo/experiments/${exp_name}"

output="/gpfs/scratch/bilals01/test-repo/logs/${exp_name}.log" 

python3 -u /gpfs/scratch/bilals01/test-repo/PathCNN/train.py $nparam --experiment $nexp > $output
