#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python3 run.py train --train-path=./data/penn/train.txt --valid-path=./data/penn/valid.txt --hidden-size 1024 
elif [ "$1" = "baseline_debug" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --train-path=./data/penn/train.txt --valid-path=./data/penn/valid.txt --hidden-size 100
elif [ "$1" = "parens_m4" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --train-path=./data/mbounded-dyck-k/m4/train.formal.txt --valid-path=./data/mbounded-dyck-k/m4/dev.formal.txt --hidden-size 12 --embedding-dim 12 --log parens_m4 --log-every 300 --epochs 30
else
	echo "Invalid Option Selected"
fi
