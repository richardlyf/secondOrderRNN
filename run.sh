#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python3 run.py train --train-path=./data/penn/train.txt --valid-path=./data/penn/valid.txt --hidden-size 1024 
else
	echo "Invalid Option Selected"
fi
