#!/bin/bash

dir="scripts/ssimat-synthkd-cifar100/"

# Array of training scripts
scripts=("train2.sh")

# Run each script sequentially in the background
for script in "${scripts[@]}"; do
    echo "Running ${script}"
    echo "Start time: $(date)" >> "${dir}log/${script}.log"
    nohup bash "${dir}$script" >> "${dir}log/${script}.log" 2>&1 &
    pid=$!
    wait $pid  # Wait for the current training to finish before proceeding
done

echo "All trainings completed."