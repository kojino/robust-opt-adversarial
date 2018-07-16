#!/bin/bash
for seed in {0..999}
do
    sbatch -c '#!/bin/bash
    #SBATCH -p shared,general,serial_requeue
    #SBATCH -o noise.%j.txt
    #SBATCH -e noise.%j.err
    #SBATCH -t 09:01:00
    #SBATCH --mem 1000
    #SBATCH -J noise
    python noise_against_logreg.py --seed='${seed}
done
