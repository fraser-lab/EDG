#!/bin/bash

PHENIX_ENV="/home/misc/software/phenix/phenix-1.21-5190/phenix_env.sh" # TODO: set path to phenix environment file
if [ -f $PHENIX_ENV ]; then
    source $PHENIX_ENV
else
    echo "Error: PHENIX environment file not found at $PHENIX_ENV"
    exit 1
fi

model_file=$1
mtz_file=$2
mtz_labels=$3
ignore_conflicts=$4
mask_atoms=$5
wrapping=$6
selection=$7
cwd=$8

cd $cwd

phenix.map_box \
    "$model_file" \
    "$mtz_file" \
    "label=$mtz_labels" \
    "ignore_symmetry_conflicts=$ignore_conflicts" \
    "mask_atoms=$mask_atoms" \
    "wrapping=$wrapping" \
    "selection=$selection"