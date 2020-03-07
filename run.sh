#!/bin/bash

if [ "$1" = "parens_m4" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --train-path=./data/mbounded-dyck-k/m4/train.formal.txt --valid-path=./data/mbounded-dyck-k/m4/dev.formal.txt --hidden-size 30 --embedding-size 12 --log parens_m4 --log-every 10 --epochs 60 --lr 1e-4 --batch-size 1 --dropout 0
elif [ "$1" = "test_m4" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --mode test --train-path=./data/mbounded-dyck-k/m4/train.formal.txt --valid-path=./data/mbounded-dyck-k/m4/dev.formal.txt --test-path ./data/mbounded-dyck-k/m4/test.formal.txt --checkpoint ./log/parens_m4_batch1_Y2020_M3_D3_h13_m33_lr0.0001/checkpoints/best_val_ppl.pth --hidden-size 30 --embedding-size 12 --batch-size 1 --dropout 0
elif [ "$1" = "test_m6" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --mode test --train-path=./data/mbounded-dyck-k/m6/train.formal.txt --valid-path=./data/mbounded-dyck-k/m6/dev.formal.txt --test-path ./data/mbounded-dyck-k/m6/test.formal.txt --checkpoint ./log/parens_m6_batch1_Y2020_M3_D3_h13_m35_lr0.0001/checkpoints/best_val_ppl.pth --hidden-size 30 --embedding-size 18 --batch-size 1 --dropout 0
elif [ "$1" = "test_m8" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --mode test --train-path=./data/mbounded-dyck-k/m8/train.formal.txt --valid-path=./data/mbounded-dyck-k/m8/dev.formal.txt --test-path ./data/mbounded-dyck-k/m8/test.formal.txt --checkpoint ./log/parens_m8_batch1_Y2020_M3_D3_h13_m35_lr0.0001/checkpoints/best_val_ppl.pth --hidden-size 30 --embedding-size 24 --batch-size 1 --dropout 0
elif [ "$1" = "test_m4_mlstm" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --mode test --model mLSTM --train-path=./data/mbounded-dyck-k/m4/train.formal.txt --valid-path=./data/mbounded-dyck-k/m4/dev.formal.txt --test-path ./data/mbounded-dyck-k/m4/test.formal.txt --checkpoint ./log/parens_m4_mLSTM_Y2020_M3_D3_h16_m58_lr0.0001/checkpoints/best_val_ppl.pth --hidden-size 30 --embedding-size 12 --batch-size 1 --dropout 0
elif [ "$1" = "train_m4_mlstm" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model mLSTM --train-path=./data/mbounded-dyck-k/m4/train.formal.txt --valid-path=./data/mbounded-dyck-k/m4/dev.formal.txt --hidden-size 30 --embedding-size 12 --batch-size 1 --dropout 0 --lr 1e-4 --log parens_m4_mLSTM --log-every 10
else
	echo "Invalid Option Selected"
fi
