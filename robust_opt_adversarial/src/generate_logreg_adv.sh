#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH -o noise.%j.txt
#SBATCH -e noise.%j.err
#SBATCH -t 09:01:00
#SBATCH --mem 8000
#SBATCH -J noise
python generate_logreg_adv.py --seed=$1
