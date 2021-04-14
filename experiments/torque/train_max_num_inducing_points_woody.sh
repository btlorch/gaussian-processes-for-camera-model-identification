#!/usr/bin/env bash

declare -a MAX_NUM_INDUCING_POINTS_CANDIDATES=(512 256 128 64 32 16 8 4)

for max_num_inducing_points in "${MAX_NUM_INDUCING_POINTS_CANDIDATES[@]}"; do
  # Train each GPC 5 times
  for i in {1..5}; do
    qsub.tinygpu -v "MAX_NUM_INDUCING_POINTS=$max_num_inducing_points,SEED=$i,MODEL_SELECTION_SEED=91058,TORCH_SEED=42" train_single_gpc_woody.sh
  done
done
