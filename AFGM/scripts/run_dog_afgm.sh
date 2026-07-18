#!/usr/bin/env bash
python train_afgm.py \
  --root_shape data/Dog/feature_images/shape \
  --root_texture data/Dog/feature_images/texture \
  --root_color data/Dog/feature_images/color \
  --shape_model checkpoints/hve/dog_shape_resnet18.pth \
  --texture_model checkpoints/hve/dog_texture_resnet18.pth \
  --color_model checkpoints/hve/dog_color_resnet18.pth \
  --save_model_dir outputs/AFGM_dog \
  --epoch 50 \
  --batch_size 64 \
  --seed 1024
