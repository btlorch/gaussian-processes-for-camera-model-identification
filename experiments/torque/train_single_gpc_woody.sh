#!/bin/bash -l
#
# Torque job script to run on TinyGPU cluster
#
# allocate 1 node for 24 hours
#PBS -l nodes=1:ppn=4:gtx1080ti,walltime=24:00:00
#
# job name
#PBS -N train_spam_gpc
#
# stdout and stderr files
#PBS -o train_spam_gpc-$PBS_JOBID.out -e train_spam_gpc-$PBS_JOBID.err
#
# Retain stdout and stderr files on the execution host. This allows them to be viewed while the job is executing. These files will be stored in the user's home directory.
#PBS -k oe
#
# If not defined already, set default values for hyper-parameters that can be overwritten
# "qsub.tinygpu -v "MAX_NUM_INDUCING_POINTS=XXX ..."
# Data
FEATURES_FILE="${FEATURES_FILE:=2020_10_28-dresden_spam_features_crop_full_resolution.h5}"
FRACTION_MOTIVES="${FRACTION_MOTIVES:=0.8}"
SEED="${SEED:=91058}"
MODEL_SELECTION_SEED="${MODEL_SELECTION_SEED:=}"

# Model
MAX_NUM_INDUCING_POINTS="${MAX_NUM_INDUCING_POINTS:=512}"
LEARN_INDUCING_LOCATIONS="${LEARN_INDUCING_LOCATIONS:=True}"
LENGTHSCALE_PRIOR_ALPHA="${LENGTHSCALE_PRIOR_ALPHA:=}"
LENGTHSCALE_PRIOR_BETA="${LENGTHSCALE_PRIOR_BETA:=}"
ENABLE_ARD="${ENABLE_ARD:=False}"

# Training
TORCH_SEED="${TORCH_SEED:=}"

# Static
PROJECT_DIR=${HOME}/i1/gaussian-processes-for-camera-model-identification
MODEL_DIR="${HPCVAULT}/camera_model_identification/models"

# Extract leading number from job id
JOB_ID=$(echo "$PBS_JOBID" | grep -E -o "^[0-9]+")

module load python/3.8-anaconda
module load cuda/10.2

# Copy data over to SSD
DATA_FILE_HDD="${HPCVAULT}/camera_model_identification/${FEATURES_FILE}"
DATA_FILE_SSD="${TMPDIR}/${FEATURES_FILE}"
echo "Copying features file to scratch SSD ${DATA_FILE_SSD}"
cp  ${DATA_FILE_HDD} ${DATA_FILE_SSD}

conda activate gpc

cd ${PROJECT_DIR}/experiments || return

PYTHONPATH=$PROJECT_DIR python train_spam_gpc.py \
  --dresden_dir "${HPCVAULT}/ddimgdb" \
  --features_file "${DATA_FILE_SSD}" \
  --fraction_motives "${FRACTION_MOTIVES}" \
  --num_known_models 10 \
  --seed "${SEED}" \
  --model_selection_seed "${MODEL_SELECTION_SEED}" \
  --max_num_inducing_points "${MAX_NUM_INDUCING_POINTS}" \
  --learn_inducing_locations "${LEARN_INDUCING_LOCATIONS}" \
  --lengthscale_prior_alpha "${LENGTHSCALE_PRIOR_ALPHA}" \
  --lengthscale_prior_beta "${LENGTHSCALE_PRIOR_BETA}" \
  --enable_ard "${ENABLE_ARD}" \
  --logdir "${MODEL_DIR}" \
  --logdir_suffix "seed_${SEED}_max_num_inducing_points_${MAX_NUM_INDUCING_POINTS}_job_${JOB_ID}" \
  --torch_seed "${TORCH_SEED}"

conda deactivate
