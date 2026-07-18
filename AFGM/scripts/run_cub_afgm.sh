#!/usr/bin/env bash
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
