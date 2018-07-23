#!/bin/bash
for seed in {87..99}
do
    echo $seed
    bash generate_logreg_adv.sh $seed
done
