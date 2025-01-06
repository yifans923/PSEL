# PSEL: Pseudo Label Learning for Partial Point Cloud Registration

This repository contains the official implementation for "Pseudo Label Learning for Partial Point Cloud Registration"

## Introduction
 We propose a novel pseudo label learning strategy that utilizes the complementarity between these two tasks to learn overlapping regions and correspondences. This strategy generates two types of pseudo labels for each task based on the distance between pairs of aligned
point clouds. During the learning process, errors in pseudo labels are gradually reduced as the accuracy of the overlap areas and correspondences improves.

## Get Started

### Requirement
- python >= 3.6
- PyTorch >= 1.8.0
- CUDA == 10.6

The code has been trained and tested with Python 3.8, PyTorch 1.12.0 and CUDA 10.6 on Ubuntu 20.04 and RTX 4090.


### Evaluation
```
cd model_utils
python test_model.py
```
### Checkpoints

We have uploaded the best model checkpoint weights in the repository, and you can change the path to load the model weights in 'test_madel.py'


### Training
```
cd model_utils
python main.py
```

## Acknowledgement
Our code refers to [PointNet](https://github.com/fxia22/pointnet.pytorch), [DCP](https://github.com/WangYueFt/dcp) and [RORNet]([https://github.com/vinits5/masknet](https://github.com/superYuezhang/RORNet/tree/main)). We want to thank the above open-source projects.


