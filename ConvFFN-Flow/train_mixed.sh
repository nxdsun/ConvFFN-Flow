#!/bin/bash
mkdir -p checkpoints
python -u train.py --name raft-chairs --stage chairs --validation chairs --gpus 0 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision 
python -u train.py --name raft-things --stage things --validation sintel --restore_ckpt checkpoints/raft-chairs.pth --gpus 0 --num_steps 120000 --batch_size 5 --lr 0.0001 --image_size 400 720 --wdecay 0.0001 --mixed_precision
python -u train.py --name raft-sintel --stage sintel --validation sintel --restore_ckpt checkpoints/raft-things.pth --gpus 0 --num_steps 120000 --batch_size 5 --lr 0.0001 --image_size 368 768 --wdecay 0.00001 --gamma=0.85 --mixed_precision
python -u train.py --name raft-kitti  --stage kitti --validation kitti --restore_ckpt checkpoints/raft-sintel.pth --gpus 0 --num_steps 50000 --batch_size 5 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85 --mixed_precision
python -u train.py --name raft-river --stage River_data --validation River_data --gpus 0 --num_steps 250000 --batch_size 8 --lr 0.00025 --image_size 400 400--wdecay 0.0001 --mixed_precision






python -u train.py --name raft-river --stage River_data --validation River_data --gpus 0 --num_steps 250000 --batch_size 4 --lr 0.0004 --image_size 400 400 --wdecay 0.0001 --mixed_precision


python -u train_convext.py --name raft-river --stage River_data --validation River_data --gpus 0 --num_steps 250000 --batch_size 4 --lr 0.0004 --image_size 400 400 --wdecay 0.0001 --mixed_precision



python -u train.py --name raft_fuliye --stage River_data --validation River_data --gpus 0 --num_steps 250000 --batch_size 4 --lr 0.0003 --image_size 400 400 --wdecay 0.0001

runs/May17_21-52-05_DESKTOP-78H7GA4