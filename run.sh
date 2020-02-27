#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python3 run.py train --train-path=./data/penn/train.txt --valid-path=./data/penn/valid.txt --hidden-size 1024 
elif [ "$1" = "baseline_debug" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --train-path=./data/penn/train.txt --valid-path=./data/penn/valid.txt --hidden-size 100
else
	echo "Invalid Option Selected"
fi
