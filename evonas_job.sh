#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=Testjob_evonas
#SBATCH --export=NONE
#SBATCH --mail-user=mateo.avila@fau.de
#SBATCH --mail-type=ALL

unset SLURM_EXPORT_ENV
module load python/3.10-anaconda
#module load cuda/11.6.1
source /home/woody/iwb3/iwb3021h/evolutionary_nas/.venv/bin/activate
cd /home/woody/iwb3/iwb3021h/evolutionary_nas/
srun python3 -W "ignore" main.py 


deactivate