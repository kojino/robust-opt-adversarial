#!/bin/bash
for seed in {0..99}
do
    echo $seed
    bash noise_against_logreg_helper.sh $seed
done
