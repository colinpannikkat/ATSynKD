#!/bin/bash

dir="scripts/ssimkd-tinyimagenet/"

# Array of training scripts
# scripts=("train5.sh")
scripts=("train6.sh" "train7.sh" "train8.sh")

# Run each script sequentially in the background
for script in "${scripts[@]}"; do
    echo "Running ${script}"
    echo "Start time: $(date)" >> "${dir}log/${script}.log"
    nohup bash "${dir}$script" >> "${dir}log/${script}.log" 2>&1 &
    pid=$!
    wait $pid  # Wait for the current training to finish before proceeding
done

echo "All trainings completed."