#!/bin/bash

set -e

CUDA_DEVICE=0
EXP_ROOT="./runs"

TEXT_MODELS=(
    bert-lstm
    clip
)

MOTION_MODELS=(
    bidir-gru
    dgstgcn
    movit-bodyparts-timemask
    movit-full-timemask
    upper-lower-gru
)

LOSSES=(
    # contrastive-fixed
    info-nce
)

DATASETS=(
    human-ml3d
    kit-mocap
)

REPS=(
  0
  #1
  #2
)

DATA_REP=cont_6d_plus_rifke
SPACE_DIM=256

BEST_METRIC=all

for TEXT_MODEL in ${TEXT_MODELS[@]}; do
    for MOTION_MODEL in ${MOTION_MODELS[@]}; do
        for LOSS in ${LOSSES[@]}; do
            for DATASET in ${DATASETS[@]}; do
                for REP in ${REPS[@]}; do
                    EXP_NAME=data=${DATASET}/motion_model=${MOTION_MODEL}/optim=${LOSS}/text_model=${TEXT_MODEL}/data_rep=${DATA_REP}/space-dim=${SPACE_DIM}/run-${REP}
                    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE HYDRA_FULL_ERROR=1 conda run --no-capture-output -n t2m python inference.py ${EXP_ROOT}/${EXP_NAME} --best_on_metric ${BEST_METRIC} --set test || true
                done
            done
        done
    done
done

