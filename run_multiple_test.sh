#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --job-name=TstSe0
#SBATCH --gres=gpu:4
#SBATCH --mem=320GB
#SBATCH --output=rq_tst_%A.out
#SBATCH --error=rq_test_%A.err

module load python/gpu/3.6.5 
cd /gpfs/scratch/coudrn01/NN_test/Seda/training/deep-cancer/

# check if next checkpoint available
declare -i count = 500 
declare -i step = 500

while true; do
    echo count
    python3 -u test_v02.py --data 'frozen' --experiment 'ds1'  --model "step_$count.pth" 
    count = `expr "$count" + "$step"`
end
