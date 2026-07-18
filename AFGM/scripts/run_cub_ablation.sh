#!/usr/bin/env bash

# Full AFGM
python train_afgm.py --texture_enhance --phased_training

# Disable fixed biological prior
python train_afgm.py --no_texture_enhance --phased_training

# Branch ablations
python train_afgm.py --ablate S
python train_afgm.py --ablate T
python train_afgm.py --ablate C
