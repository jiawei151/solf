# SOLF
Implementation of our technical report [Weight-Output Bi-Directional Constraint for Continual Learning](https://www.researchgate.net/publication/369901519_Weight-Output_Bi-Directional_Constraint_for_Continual_Learning). 

#### Dataset
* ImageNet100
Refer to [ImageNet100_Split](https://github.com/arthurdouillard/incremental_learning.pytorch/tree/master/imagenet_split)


## Dependencies

* python 3.7.9
* pytorch >1.0 (>1.7 recommended)
* sacred 0.8.2
* torchvision 0.10.1
* cuda 11.2

## Training

For CIFAR100, run:
```
sh scripts/run.sh
```
For ImageNet-100, run:
```
sh scripts/run_imagenet.sh
```

## Acknowledgement
Thanks for the great code base from https://github.com/Rhyssiyan/DER-ClassIL.pytorch.

