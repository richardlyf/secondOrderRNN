#!/bin/bash

if [ "$1" = "parens_m4" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --train-path=./data/mbounded-dyck-k/m4/train.formal.txt --valid-path=./data/mbounded-dyck-k/m4/dev.formal.txt --hidden-size 12 --embedding-size 30 --log parens_m4 --log-every 10 --epochs 60 --lr 1e-4 --batch-size 1 --dropout 0 --is-stream 0
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
elif [ "$1" = "train_penn" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model ptb_lstm --train-path=./data/penn/train.txt --valid-path=./data/penn/valid.txt --hidden-size 600 --embedding-size 300 --batch-size 64 --dropout 0 --lr 1e-4 --is-stream 1 --bptt 70 --log-every 10 --log penn_lstm_batch64_70bptt --epochs 1000
elif [ "$1" = "test_penn" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --mode test --model ptb_lstm --train-path=./data/penn/train.txt --valid-path=./data/penn/valid.txt --test-path=./data/penn/test.txt --checkpoint ./log/penn_lstm_batch64_70bptt_Y2020_M3_D10_h4_m31_lr0.0001/checkpoints/best_val_ppl.pth --hidden-size 600 --embedding-size 300 --batch-size 64 --dropout 0 --lr 1e-4 --is-stream 1 --bptt 70
elif [ "$1" = "debug_penn" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model ptb_lstm --train-path=./data/penn/tiny_train.txt --valid-path=./data/penn/valid.txt --test-path=./data/penn/test.txt --hidden-size 600 --embedding-size 300 --batch-size 64 --dropout 0 --lr 1e-4 --is-stream 1 --bptt 70
elif [ "$1" = "wiki_stats" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --mode test --model ptb_lstm --train-path=./data/wikitext-2/train.txt --valid-path=./data/wikitext-2/valid.txt --test-path LM_syneval/EMNLP2018/templates/all_test_sents.txt --checkpoint ./final/wiki2_baseline/best_val_ppl.pth --hidden-size 600 --embedding-size 300 --batch-size 1 --dropout 0.5 --lr 1e-4 --is-stream 0 --bptt 70 --stats-output-file rnn.output
else
	echo "Invalid Option Selected"
fi
