# ConvFFN-Flow
Non-contact river surface velocity estimation using a lightweight optical flow framework and realistic river imagery
## Method Overview
ConvFFN-Flow is built upon a recurrent optical flow estimation framework and introduces the following key features:
- A **Convolutional Feed-Forward Network (ConvFFN)** to enhance local feature aggregation and motion representation  
- Lightweight architectural design for efficient inference and reduced computational cost  
- Improved robustness in weak-texture regions and specular reflection areas commonly observed in river surfaces  
- Compatibility with standard optical flow benchmarks as well as realistic river imagery  
---

## Environment Requirements

The code has been tested with the following configuration:

```bash
Python >= 3.11
PyTorch >= 2.1
CUDA >= 12.1
<img src="">
## Requirements
The code has been tested with PyTorch 2.1 and Cuda 12.1.
```Shell
conda create --name ConvFFN-Flow
conda activate ConvFFN-Flow
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

```
## Pretrained Models

We provide pretrained ConvFFN-Flow models for realistic river surface motion estimation.
All official pretrained weights are released via GitHub Releases:

[https://github.com/nxdsun/ConvFFN-Flow/releases]

Multiple model variants trained under different configurations are available.
Please refer to the release notes for details of each model.

## Demos
Pretrained models can be downloaded by running
```Shell
./download_models.sh
```
or downloaded from [google drive](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing)

You can demo a trained model on a sequence of frames
```Shell
python demo.py --model=models/raft-things.pth --path=demo-frames
```

## Required Data
To evaluate/train RAFT, you will need to download the required datasets. 
* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/) (optional)


By default `datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder

```Shell
├── datasets
    ├── RIVER-Dataset
        ├── test
        ├── training
## Evaluation
You can evaluate a trained model using `evaluate.py`
```Shell
python evaluate.py --model=models/raft-things.pth --dataset=sintel --mixed_precision
```

## Training
We used the following training schedule in our paper (2 GPUs). Training logs will be written to the `runs` which can be visualized using tensorboard
```Shell
./train_standard.sh
```


