#!/bin/bash
for seed in {16..100}
do
    echo $seed
    sbatch generate_logreg_adv.sh $seed
done
