#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --gres=gpu:p100:1
##SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --mem=50GB
#SBATCH --output=outputs/train_%A.out
#SBATCH --error=outputs/train_%A.err

##### below is for on nyulmc hpc: bigpurple #####
##### above is for on nyu hpc: prince #####

##!/bin/bash
##SBATCH --partition=gpu8_medium
##SBATCH --ntasks=8
##SBATCH --cpus-per-task=1
##SBATCH --job-name=test_PCNN
##SBATCH --gres=gpu:1
##SBATCH --output=outputs/rq_train1_%A_%a.out
##SBATCH --error=outputs/rq_train1_%A_%a.err
##SBATCH --mem=200GB

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


#input params
exp_name="lung_pathcnn_6layers_tr"
im_size="299" #224, 299
model="6layers"
model_cp="epoch_1.pth"
test_val="test" 
tlog="/scratch/sb3923/logs/${exp_name}.log" 

nparam="--model=${model_cp} --imgSize=${im_size} --root_dir=/beegfs/sb3923/DeepCancer/alldata/LungTilesSorted/ --num_class=3 --tile_dict_path=/beegfs/sb3923/DeepCancer/alldata/LungTilesSorted/Lung_FileMappingDict.p --train_log=${tlog} --val=${test_val} --model_type=${model}"

nexp="/scratch/sb3923/experiments/${exp_name}"
output="/scratch/sb3923/logs/time_${exp_name}_${test_val}_${model_cp}.log" 

python3 -u /scratch/sb3923/PathCNN/test.py $nparam --experiment $nexp > $output
