#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:?"Usage: bash scripts/preprocess.sh DATA_ROOT"}
mkdir -p "$DATA_ROOT/nnUNet_raw" "$DATA_ROOT/nnUNet_preprocessed" "$DATA_ROOT/nnUNet_results"
DATA_ROOT=$(cd "$DATA_ROOT" && pwd)

export nnUNet_raw="$DATA_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$DATA_ROOT/nnUNet_preprocessed"
export nnUNet_results="$DATA_ROOT/nnUNet_results"

nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres --verify_dataset_integrity
