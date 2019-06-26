#!/bin/bash
#SBATCH --job-name=redmagic_gmm
#SBATCH --output=job/arrayJob_%A_%a.out
#SBATCH --error=job/arrayJob_%A_%a.err
#SBATCH --array=0
#SBATCH --time=24:00:00
#SBATCH -p iric,hns,normal
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
PARRAY=("False" "True")
echo "My SLURM_ARRAY_TASK_ID is " $SLURM_ARRAY_TASK_ID
which python
srun python ../code/run_diagoostic.py --paramfile ../yamlfiles/test_0625.yaml
