# Avian-inspired Feature Gating Model Based on HVE

Code for the Avian-inspired Feature Gating Model (AFGM) built on the Humanoid Vision Engine (HVE) framework.

The implementation files are provided in the `AFGM/` folder.

Original HVE repository:  
https://github.com/gyhandy/Humanoid-Vision-Engine

## Quick start

```bash
cd AFGM
pip install -r requirements.txt
python train_afgm.py

## Overview

AFGM extends the HVE framework by introducing:

- sample-dependent dynamic feature gating;
- sparse gate regularization;
- a fixed biological gating prior;
- component ablation experiments;
- CUB-200-2011 and dog fine-grained recognition experiments.

The model uses frozen shape, texture, and color branch encoders and trains the adaptive gating module and classifier head.

## Repository structure

```text
AFGM-HVE/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── train_afgm.py
├── util/
│   ├── tools.py
│   ├── data_loader.py
│   └── attention.py
├── configs/
├── scripts/
├── data/
│   ├── README.md
│   ├── splits/
│   └── image_lists/
├── results/
├── checkpoints/
└── outputs/
```

## Important note

The uploaded `train_afgm.py` depends on the HVE utility files and the modified AFGM attention/gating module. To run this repository, please add the following files under `util/`:

```text
util/tools.py
util/data_loader.py
util/attention.py
```

These files should include the functions/classes used by `train_afgm.py`:

```python
load_resnet18
get_latent_output
get_Dataloader
attention
```

## Dataset sources

This repository does not redistribute the original or processed image datasets.

### HVE framework and feature-biased datasets

The AFGM implementation is built on the Humanoid Vision Engine (HVE) framework:

https://github.com/gyhandy/Humanoid-Vision-Engine

The shape-, texture-, and color-biased datasets used for HVE-style feature representation should be obtained from the original HVE source.

### CUB-200-2011

The CUB-200-2011 foreground-segmented inputs used in our experiments were generated following the preprocessing procedure described in the HVE framework.

The processed foreground-segmented CUB images are not redistributed in this repository due to copyright restrictions of the original images.

### StanfordExtra V12 dog dataset

The dog fine-grained recognition experiments were based on StanfordExtra V12 and Stanford Dogs/ImageNetDogs.

StanfordExtra official repository:  
https://github.com/benjiebob/StanfordExtra

StanfordExtra annotation release form:  
https://forms.gle/sRtbicgxsWvRtRmUA

Stanford Dogs / ImageNetDogs official page:  
http://vision.stanford.edu/aditya86/ImageNetDogs/

Stanford Dogs image archive:  
http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar

## Installation

```bash
pip install -r requirements.txt
```

## Example usage

Replace the paths below with your local paths to the HVE-preprocessed feature images and pretrained branch encoders.

```bash
python train_afgm.py \
  --root_shape data/CUB/feature_images/shape \
  --root_texture data/CUB/feature_images/texture \
  --root_color data/CUB/feature_images/color \
  --shape_model checkpoints/hve/shape_resnet18.pth \
  --texture_model checkpoints/hve/texture_resnet18.pth \
  --color_model checkpoints/hve/color_resnet18.pth \
  --save_model_dir outputs/AFGM \
  --epoch 50 \
  --batch_size 64 \
  --seed 1024
```

## Component ablation

Disable the fixed biological gating prior:

```bash
python train_afgm.py --no_texture_enhance
```

Perform feature-branch ablation:

```bash
python train_afgm.py --ablate S
python train_afgm.py --ablate T
python train_afgm.py --ablate C
```

## Outputs

The training script saves:

- `config.json`
- `metrics_epoch.csv`
- `log.txt`
- `training_summary.json`
- `best.pth`
- `last.pth`

Large checkpoints are not included in this repository by default. If model weights are released, upload them separately through GitHub Releases or Zenodo.

## Acknowledgement

This implementation is built on the HVE framework. Please cite or acknowledge the original HVE work when using this repository.

If you use this repository, please also cite the corresponding AFGM manuscript.
