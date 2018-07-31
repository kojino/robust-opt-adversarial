#!/bin/bash
for seed in {12..13}
do
    echo $seed
    sbatch generate_logreg_adv.sh $seed
done
