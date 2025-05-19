# ThermalFaceRecognition
Face Recognition system based on custom ResNet for thermal images, involving the dataset of Visual Pairs provided by SpeakingFaces available at https://github.com/IS2AI/SpeakingFaces.

## DATASET
Download the SpeakingFaces dataset and organize folders as shown below. For each identity we require a different folder. First digits of images name give the ID of the participant.
```python create_subdir.py```
Then we can split the dataset into train-test-val according to 70-20-10, keeping the datafolders organization.
```python split.py```

```python split.py```

## HOW TO USE
```
dataset_split/
├── train/
│   ├── 1/
│   │   ├── 1_1_1.png
│   │   └── ...
│   ├── 2/
│   │   ├── 2_1_1.png
│   │   └── ...
│   └── ...
├── test/
│   ├── 1/
│   │   ├── 1_1_9.png
│   │   └── ...
│   ├── 2/
│   │   ├── 2_1_9.png
│   │   └── ...
│   └── ...
└── val/
    ├── 1/
    │   ├── 1_1_17.png
    │   └── ...
    ├── 2/
    │   ├── 2_1_17.png
    │   └── ...
    └── ...
```

## NETWORK DETAILS
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
CustomResNet                             [32, 142]                 --
├─Conv2d: 1-1                            [32, 64, 56, 56]          3,136
├─BatchNorm2d: 1-2                       [32, 64, 56, 56]          128
├─MaxPool2d: 1-3                         [32, 64, 28, 28]          --
├─Sequential: 1-4                        [32, 64, 28, 28]          --
│    └─BasicBlock: 2-1                   [32, 64, 28, 28]          --
│    │    └─Conv2d: 3-1                  [32, 64, 28, 28]          36,864
│    │    └─BatchNorm2d: 3-2             [32, 64, 28, 28]          128
│    │    └─Conv2d: 3-3                  [32, 64, 28, 28]          36,864
│    │    └─BatchNorm2d: 3-4             [32, 64, 28, 28]          128
│    │    └─Sequential: 3-5              [32, 64, 28, 28]          --
│    └─BasicBlock: 2-2                   [32, 64, 28, 28]          --
│    │    └─Conv2d: 3-6                  [32, 64, 28, 28]          36,864
│    │    └─BatchNorm2d: 3-7             [32, 64, 28, 28]          128
│    │    └─Conv2d: 3-8                  [32, 64, 28, 28]          36,864
│    │    └─BatchNorm2d: 3-9             [32, 64, 28, 28]          128
│    │    └─Sequential: 3-10             [32, 64, 28, 28]          --
├─Sequential: 1-5                        [32, 128, 14, 14]         --
│    └─BasicBlock: 2-3                   [32, 128, 14, 14]         --
│    │    └─Conv2d: 3-11                 [32, 128, 14, 14]         73,728
│    │    └─BatchNorm2d: 3-12            [32, 128, 14, 14]         256
│    │    └─Conv2d: 3-13                 [32, 128, 14, 14]         147,456
│    │    └─BatchNorm2d: 3-14            [32, 128, 14, 14]         256
│    │    └─Sequential: 3-15             [32, 128, 14, 14]         8,448
│    └─BasicBlock: 2-4                   [32, 128, 14, 14]         --
│    │    └─Conv2d: 3-16                 [32, 128, 14, 14]         147,456
│    │    └─BatchNorm2d: 3-17            [32, 128, 14, 14]         256
│    │    └─Conv2d: 3-18                 [32, 128, 14, 14]         147,456
│    │    └─BatchNorm2d: 3-19            [32, 128, 14, 14]         256
│    │    └─Sequential: 3-20             [32, 128, 14, 14]         --
├─Sequential: 1-6                        [32, 256, 7, 7]           --
│    └─BasicBlock: 2-5                   [32, 256, 7, 7]           --
│    │    └─Conv2d: 3-21                 [32, 256, 7, 7]           294,912
│    │    └─BatchNorm2d: 3-22            [32, 256, 7, 7]           512
│    │    └─Conv2d: 3-23                 [32, 256, 7, 7]           589,824
│    │    └─BatchNorm2d: 3-24            [32, 256, 7, 7]           512
│    │    └─Sequential: 3-25             [32, 256, 7, 7]           33,280
│    └─BasicBlock: 2-6                   [32, 256, 7, 7]           --
│    │    └─Conv2d: 3-26                 [32, 256, 7, 7]           589,824
│    │    └─BatchNorm2d: 3-27            [32, 256, 7, 7]           512
│    │    └─Conv2d: 3-28                 [32, 256, 7, 7]           589,824
│    │    └─BatchNorm2d: 3-29            [32, 256, 7, 7]           512
│    │    └─Sequential: 3-30             [32, 256, 7, 7]           --
├─AdaptiveAvgPool2d: 1-7                 [32, 256, 1, 1]           --
├─Linear: 1-8                            [32, 142]                 36,494
