#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python3 run.py train --train-path=./data/penn/train.txt --valid-path=./data/penn/valid.txt --hidden-size 1024 
elif [ "$1" = "debug" ]; then
    CUDA_VISIBLE_DEVICES=0 python run.py --train-path=./data/mbounded-dyck-k/m4/train.formal.txt --valid-path=./data/mbounded-dyck-k/m4/dev.formal.txt --hidden-size 30 --embedding-dim 12              
elif [ "$1" = "parens_m4" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --train-path=./data/mbounded-dyck-k/m4/train.formal.txt --valid-path=./data/mbounded-dyck-k/m4/dev.formal.txt --hidden-size 30 --embedding-dim 12 --log parens_m4 --log-every 10 --epochs 60 --lr 1e-4 --model mLSTM
else
	echo "Invalid Option Selected"
fi
