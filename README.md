# AFGM 

Code for the Avian-inspired Feature Gating Model (AFGM) built on the Humanoid Vision Engine (HVE) framework.

This repository contains the implementation of AFGM, including dynamic feature gating, sparse gate regularization, and a fixed biological gating prior.

The original HVE framework is available from the official HVE repository:  
https://github.com/gyhandy/Humanoid-Vision-Engine

## Dataset sources

This repository does not redistribute the original or processed image datasets.  
Please obtain the datasets from their original authorized sources.

### Feature-biased datasets

The shape-, texture-, and color-biased datasets used in this study follow the HVE framework and should be obtained from the original HVE source:

https://github.com/gyhandy/Humanoid-Vision-Engine

### CUB-200-2011

The CUB-200-2011 dataset should be obtained from the official Caltech-UCSD Birds-200-2011 source.

The foreground-segmented CUB-200-2011 inputs used in our experiments were generated following the preprocessing procedure described in the HVE framework. The processed foreground-segmented CUB images are not redistributed in this repository due to copyright restrictions of the original images.

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

## Files provided in this repository

This repository includes:

- AFGM model implementation;
- dynamic gating module;
- sparse gate regularization;
- fixed biological gating prior;
- CUB-200-2011 experiment scripts;
- dog fine-grained recognition experiment scripts;
- component ablation scripts;
- configuration files;
- result tables used in the manuscript.

The original and processed image files are not included.v
