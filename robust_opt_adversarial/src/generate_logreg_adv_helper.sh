#!/bin/bash
for seed in {10..11}
do
    echo $seed
    bash generate_logreg_adv.sh $seed
done
