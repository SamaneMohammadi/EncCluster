#!/bin/bash

# Define the number of clients and GPUs
NUM_CLIENTS=30
NUM_GPUS=2

# Define baselines
baselines=("bfuse")

# Define round and skewness pairs
declare -A round_skewness_pairs
#round_skewness_pairs[1000]=0.2
round_skewness_pairs[3000]=10.0

# Iterate through each pair of NUM_ROUNDS and SKEWNESS
for NUM_ROUNDS in "${!round_skewness_pairs[@]}"
do
    SKEWNESS=${round_skewness_pairs[$NUM_ROUNDS]}
    echo "Running experiments for NUM_ROUNDS=$NUM_ROUNDS and SKEWNESS=$SKEWNESS"

    # Iterate through each baseline
    for BASELINE in "${baselines[@]}"
    do
        echo "Running experiment on $BASELINE"
        cd "$BASELINE" || exit

        # Calculate gRAM_per_client with 4 decimal places
        gRAM_per_client=$(echo "scale=4; $NUM_GPUS/$NUM_CLIENTS" | bc)

        # Execute experiment
        python main.py \
            --num_clients=$NUM_CLIENTS \
            --gRAM_per_client=$gRAM_per_client \
            --num_rounds=$NUM_ROUNDS \
            --model='mlpmixer' \
            --dataset='cifar10' \
            --skewness=$SKEWNESS \
            --participation=1.0

        cd .. || exit
        sleep 1
    done
done

echo "All experiments completed."
