#!/bin/bash
#SBATCH -p seasdgx1
#SBATCH -o noise.%j.txt
#SBATCH -e noise.%j.err
#SBATCH -t 09:01:00
#SBATCH --mem 16000
#SBATCH -J noise
python generate_logreg_adv.py --seed=$1 &
