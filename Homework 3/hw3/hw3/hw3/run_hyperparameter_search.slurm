#!/bin/bash
#SBATCH --job-name=roberta_hyperparameter_search
#SBATCH --open-mode=append
#SBATCH --output=/scratch/sons01/output/%j_%x.out
#SBATCH --error=/scratch/sons01/error/%j_%x.err
#SBATCH --export=ALL
#SBATCH --account=ds_ga_1012_2022sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --mem=20G
#SBATCH -c 4

#SBATCH --mail-user=<sons01@nyu.edu>
#SBATCH --mail-type=ALL

singularity exec --nv --bind /scratch/sons01 --overlay /scratch/sons01/overlay-25GB-500K.ext3:ro /scratch/sons01/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python /scratch/sons01/hw3/run_hyperparameter_search.py /scratch/sons01/data/BoolQ
"
