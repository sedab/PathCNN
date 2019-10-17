#!/bin/bash
#
#SBATCH --job-name=m_test
##SBATCH --gres=gpu:p100:1
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --mem=100GB
#SBATCH --output=outputs/test_%A.out
#SBATCH --error=outputs/test_%A.err
#
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
module load python3/intel/3.5.3
module load pytorch/python3.5/0.2.0_3
module load torchvision/python3.5/0.1.9

#input params
exp_name="breast_sub_basal_vs_rest_7layers_tr"
test_val="test"
im_size="299" 
model="7layers"
tlog="/scratch/sb3923/logs/${exp_name}.log" 

#breast subtypes
nparam="--root_dir=/beegfs/sb3923/DeepCancer/alldata/BreastSubtypes_basal_vs_rest_no_normal/BreastTilesSorted/ --num_class=2 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/BreastSubtypes_basal_vs_rest_no_normal/Breast_FileMappingDict.p --train_log=${tlog} --val=${test_val} --imgSize=${im_size} --model_type=${model}"
  
#nparam="--root_dir=/beegfs/sb3923/DeepCancer/alldata/BreastSubtypes_no_normal/BreastTilesSorted/ --num_class=4 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/BreastSubtypes_no_normal/Breast_FileMappingDict.p --train_log=${tlog} --val=${test_val} --imgSize=${im_size} --model_type=${model}"

nexp="/scratch/sb3923/experiments/${exp_name}"
out="/scratch/sb3923/logs/${test_val}"


#if [ ! -d $out ]; then
#    mkdir -p $out;
#fi

# check if next checkpoint available
declare -i count=1
declare -i step=1 

while true; do
    echo $count
    PathToEpoch="${nexp}/checkpoints/"
    Cmodel="epoch_$count.pth"
    output="${out}/tsne_${test_val}_${exp_name}_${Cmodel}.log"
    echo $PathToEpoch
    echo $Cmodel
    echo $output
    if [ -f $PathToEpoch/$Cmodel ]; then
        python3 -u tsne.py  --experiment $nexp  --model $Cmodel $nparam > $output
    else
        break
    fi
    count=`expr "$count" + "$step"`
done
