#!/bin/bash
# Resource Request
#SBATCH --partition=cpu_dev
#SBATCH --job-name=gauto_conv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=3:00:00
#SBATCH --output=outputs/cpu_train1_%A_%a.out
#SBATCH --error=outputs/cpu_train1_%A_%a.err

##!/bin/bash
##SBATCH --partition=gpu4_short
##SBATCH --ntasks=8
##SBATCH --cpus-per-task=1
##SBATCH --job-name=multiple_PCNN
##SBATCH --gres=gpu:4
##SBATCH --output=outputs/rq_train1_%A_%a.out
##SBATCH --error=outputs/rq_train1_%A_%a.err
##SBATCH --mem=100GB

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

#input params
exp_name="exp8"
test_val="test"
im_size="224" 
model="alexnet"


nparam="--root_dir=/beegfs/sb3923/DeepCancer/alldata/LungTilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/LungTilesSorted/Lung_FileMappingDict.p --val=${test_val} --imgSize=im_size --model_type=model" 
#kidney
#nparam="--root_dir=/beegfs/sb3923/DeepCancer/alldata/KidneyTilesSorted/ --num_class=4 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/KidneyTilesSorted/Kidney_FileMappingDict.p --val=${test_val} --imgSize=im_size --model_type=model"
#breast
#nparam="--root_dir=/beegfs/sb3923/DeepCancer/alldata/BreastTilesSorted/ --num_class=2 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/BreastTilesSorted/Breast_FileMappingDict.p --val=${test_val} --imgSize=im_size --model_type=model"


#downsampled lung, you probably need the full dataset
#ds1
#nparam="--root_dir=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds1TilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds1_FileMappingDict.p --val=${test_val} --imgSize=im_size --model_type=model" 
#ds2
#nparam="--root_dir=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds2TilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds2_FileMappingDict.p --val=${test_val} --imgSize=im_size --model_type=model" 
#ds3
#nparam="--root_dir=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds3TilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds3_FileMappingDict.p --val=${test_val} --imgSize=im_size --model_type=model" 



nexp="/gpfs/scratch/bilals01/test-repo/experiments/${exp_name}"

out="/gpfs/scratch/bilals01/test-repo/logs/${test_val}"

if [ ! -d $out ]; then
    mkdir -p $out;
fi

# check if next checkpoint available
declare -i count=1
declare -i step=1 

while true; do
    echo $count
    PathToEpoch="${nexp}/checkpoints/"
    Cmodel="epoch_$count.pth"
    output="${out}/${test_val}_${exp_name}_${Cmodel}.log"
    echo $PathToEpoch
    echo $Cmodel
    echo $output
    if [ -f $PathToEpoch/$Cmodel ]; then
        python3 -u test.py  --experiment $nexp  --model $Cmodel $nparam > $output
    else
        break
    fi
    count=`expr "$count" + "$step"`
done
