# ConvFFN-Flow
Non-contact river surface velocity estimation using a lightweight optical flow framework and realistic river imagery
## Method Overview
ConvFFN-Flow is built upon a recurrent optical flow estimation framework and introduces the following key features:
- A **Convolutional Feed-Forward Network (ConvFFN)** to enhance local feature aggregation and motion representation  
- Lightweight architectural design for efficient inference and reduced computational cost  
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



You can demo a trained model on a sequence of frames
```Shell
python demo_test.py \ 
  --model checkpoints/raft-convext.pth --path E:/CRG --output_path E:/CRG_demo/convfft
```

By default `datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder

```Shell
├── datasets
    ├── RIVER-Dataset
        ├── test
        ├── training
        ├── validation
## Evaluation
You can evaluate a trained model using `evaluate.py`
```Shell
python evaluate.py --model=models/raft-convext.pth --dataset=River_data --mixed_precision
```

## Training
We used the following training schedule in our paper (1 GPUs). Training logs will be written to the `runs` which can be visualized using tensorboard
```Shell
 python -u train_convext.py --name raft-convext --stage River_data --validation River_data --gpus 0 --num_steps 250000 --batch_size 8 --lr 0.0001 --image_size 400 400 --wdecay 0.0001 --mixed_precision
```


