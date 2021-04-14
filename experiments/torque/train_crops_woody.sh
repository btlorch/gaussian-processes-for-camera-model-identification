#!/usr/bin/env bash

declare -a FEATURES_FILES=(\
  "2020_10_28-dresden_spam_features_crop_full_resolution.h5" \
  "2020_10_28-dresden_spam_features_crop_1024.h5" \
  "2020_10_28-dresden_spam_features_crop_512.h5" \
  "2020_10_28-dresden_spam_features_crop_256.h5" \
  "2020_10_29-dresden_spam_features_crop_128.h5")

for FEATURES_FILE in "${FEATURES_FILES[@]}"; do
  # Train with 5 different training-test splits
  for i in {1..5}; do
    qsub.tinygpu -v "FEATURES_FILE=$FEATURES_FILE,MODEL_SELECTION_SEED=91058,SEED=$i,TORCH_SEED=42" train_single_gpc_woody.sh
  done
done
