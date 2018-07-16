#!/bin/bash
for seed in {0..999}
do
    echo $seed
    bash noise_against_logreg_helper.sh $seed
done
