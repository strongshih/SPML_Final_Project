#!/bin/bash

CUDA_VISIBLE_DEVICES="3" python3 ./train_MIT_robust_dataset.py \
    --model resnet_32 \
    --nz 128 \
    --ngf 64 \
    --ndf 64 \
    --nclass 10 \
    --batch_size 64 \
    --start_width 4 \
    --dataset cifar10 \
    --root ./cifar10_data \
    --img_width 32 \
    --iter_d 5 \
    --out_f ckpt.adv-5.128px-cifar10 \
    --ngpu 1 \
    --starting_epoch 0 \
    --max_epoch 200 \
    --lr 0.0002 \
    --adv_steps 5 \
    --epsilon 0.0625 \
    --our_loss # Our ACGAN
