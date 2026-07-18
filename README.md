# AFGM based on HVE
Code for the Avian-inspired Feature Gating Model built on the HVE framework.

# AFGM-HVE

This repository contains the implementation of the Avian-inspired Feature Gating Model (AFGM), built on the Humanoid Vision Engine (HVE) framework.

The original HVE framework is available from the official HVE repository:
https://github.com/gyhandy/Humanoid-Vision-Engine

# Dataset sources

This repository does not redistribute the original or processed image datasets. 
Please obtain the datasets from their original authorized sources.

## feature-biased datasets

https://github.com/gyhandy/Humanoid-Vision-Engine

The shape-, texture-, and color-biased datasets used in this study follow the HVE framework and should be obtained from the original HVE source.

## CUB-200-2011

The CUB-200-2011 dataset should be obtained from the official Caltech-UCSD Birds-200-2011 source. 
The processed foreground-segmented CUB images are not redistributed in this repository due to copyright restrictions of the original images.
The foreground-segmented CUB-200-2011 inputs used in our experiments were generated following the preprocessing procedure described in the HVE framework.

## StanfordExtraDog dataset

StanfordExtra official repository:
https://github.com/benjiebob/StanfordExtra

StanfordExtra annotation release form:
https://forms.gle/sRtbicgxsWvRtRmUA

Stanford Dogs / ImageNetDogs official page:
http://vision.stanford.edu/aditya86/ImageNetDogs/

Stanford Dogs image archive:
http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar

## Files provided in this repository

This repository includes:
- AFGM model implementation;
- dynamic gating module;
- sparse gate regularization;
- fixed biological gating prior;
- CUB-200-2011 experiment scripts;
- dog fine-grained recognition experiment scripts;
- component ablation scripts;
- configuration files.


