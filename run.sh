#!/bin/bash

##########################
### Parentheses m4
##########################
if [ "$1" = "train_m4_baseline" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model baseline_lstm --train-path=./data/mbounded-dyck-k/m4/train.formal.txt --valid-path=./data/mbounded-dyck-k/m4/dev.formal.txt --hidden-size 12 --embedding-size 30 --log baseline_m4_batch10 --log-every 10 --epochs 1000 --lr 1e-4 --batch-size 10 --dropout 0 --num-layers 1 --is-stream 0
elif [ "$1" = "test_m4_baseline" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model baseline_lstm --train-path=./data/mbounded-dyck-k/m4/train.formal.txt --valid-path=./data/mbounded-dyck-k/m4/dev.formal.txt --test-path ./data/mbounded-dyck-k/m4/test.formal.txt  --hidden-size 12 --embedding-size 30 --batch-size 10 --dropout 0 --num-layers 1 --is-stream 0 --epochs 1000 --mode test --checkpoint ./log/baseline_m4_batch10_Y2020_M3_D15_h11_m11_lr0.0001/checkpoints/best_val_ppl.pth
elif [ "$1" = "train_m4_attention" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model attention --train-path=./data/mbounded-dyck-k/m4/train.formal.txt --valid-path=./data/mbounded-dyck-k/m4/dev.formal.txt --hidden-size 12 --embedding-size 30 --batch-size 10 --dropout 0 --lr 1e-4 --log attention_m4_batch10 --log-every 10
elif [ "$1" = "test_m4_attention" ]; then
    CUDA_VISIBLE_DEVICES=0 python3  --model attention --train-path=./data/mbounded-dyck-k/m4/train.formal.txt --valid-path=./data/mbounded-dyck-k/m4/dev.formal.txt --test-path ./data/mbounded-dyck-k/m4/test.formal.txt --hidden-size 12 --embedding-size 30 --batch-size 10 --dropout 0 --lr 1e-4 --mode test --checkpoint ./log/final/attention_m4_batch10_Y2020_M3_D15_h11_m11_lr0.0001/checkpoints/best_val_ppl.pth

##########################
### Parentheses m6
##########################
elif [ "$1" = "train_m6_baseline" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model baseline_lstm --train-path=./data/mbounded-dyck-k/m6/train.formal.txt --valid-path=./data/mbounded-dyck-k/m6/dev.formal.txt --hidden-size 18 --embedding-size 30 --log baseline_m6_batch10 --log-every 10 --epochs 1000 --lr 1e-4 --batch-size 10 --dropout 0 --num-layers 1 --is-stream 0
elif [ "$1" = "test_m6_baseline" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model baseline_lstm --train-path=./data/mbounded-dyck-k/m6/train.formal.txt --valid-path=./data/mbounded-dyck-k/m6/dev.formal.txt --test-path ./data/mbounded-dyck-k/m6/test.formal.txt  --hidden-size 18 --embedding-size 30 --batch-size 10 --dropout 0 --num-layers 1 --is-stream 0 --mode test --checkpoint ./log/final/baseline_m6_batch10_Y2020_M3_D13_h21_m5_lr0.0001/checkpoints/best_val_ppl.pth
elif [ "$1" = "train_m6_attention" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model attention --train-path=./data/mbounded-dyck-k/m6/train.formal.txt --valid-path=./data/mbounded-dyck-k/m6/dev.formal.txt --hidden-size 18 --embedding-size 30 --batch-size 10 --dropout 0 --lr 1e-4 --log attention_m4_batch10 --log-every 10 --is-stream 0
elif [ "$1" = "test_m6_attention" ]; then
    CUDA_VISIBLE_DEVICES=0 python3  --model attention --train-path=./data/mbounded-dyck-k/m6/train.formal.txt --valid-path=./data/mbounded-dyck-k/m6/dev.formal.txt --test-path ./data/mbounded-dyck-k/m6/test.formal.txt --hidden-size 18 --embedding-size 30 --batch-size 10 --dropout 0 --lr 1e-4 --is-stream 0 --mode test --checkpoint ./log/final/attention_m6_batch10_Y2020_M3_D13_h15_m22_lr0.0001/checkpoints/best_val_ppl.pth

##########################
### Parentheses m8
##########################
elif [ "$1" = "train_m8_baseline" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model baseline_lstm --train-path=./data/mbounded-dyck-k/m8/train.formal.txt --valid-path=./data/mbounded-dyck-k/m8/dev.formal.txt --hidden-size 24 --embedding-size 30 --log baseline_m6_batch10 --log-every 10 --epochs 1000 --lr 1e-4 --batch-size 10 --dropout 0 --num-layers 1 --is-stream 0
elif [ "$1" = "test_m8_baseline" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model baseline_lstm --train-path=./data/mbounded-dyck-k/m8/train.formal.txt --valid-path=./data/mbounded-dyck-k/m8/dev.formal.txt --test-path ./data/mbounded-dyck-k/m8/test.formal.txt  --hidden-size 24 --embedding-size 30 --batch-size 10 --dropout 0 --num-layers 1 --is-stream 0 --mode test --checkpoint ./log/final/baseline_m8_batch10_Y2020_M3_D13_h15_m4_lr0.0001/checkpoints/best_val_ppl.pth
elif [ "$1" = "train_m8_attention" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model attention --train-path=./data/mbounded-dyck-k/m8/train.formal.txt --valid-path=./data/mbounded-dyck-k/m8/dev.formal.txt --hidden-size 24 --embedding-size 30 --batch-size 10 --dropout 0 --lr 1e-4 --log attention_m4_batch10 --log-every 10 --is-stream 0
elif [ "$1" = "test_m8_attention" ]; then
    CUDA_VISIBLE_DEVICES=0 python3  --model attention --train-path=./data/mbounded-dyck-k/m8/train.formal.txt --valid-path=./data/mbounded-dyck-k/m8/dev.formal.txt --test-path ./data/mbounded-dyck-k/m8/test.formal.txt --hidden-size 24 --embedding-size 30 --batch-size 10 --dropout 0 --is-stream 0 --lr 1e-4 --mode test --checkpoint ./log/final/attention_m8_batch10_Y2020_M3_D13_h15_m13_lr0.0001/checkpoints/best_val_ppl.pth 
##########################
### Penn Treebank
##########################

elif [ "$1" = "train_penn_baseline" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model stream_lstm --train-path=./data/penn/train.txt --valid-path=./data/penn/valid.txt --hidden-size 600 --embedding-size 300 --batch-size 64 --bptt 70 --dropout 0.5 --lr 1e-4 --is-stream 1  --log-every 10 --log baseline_penn_batch64_bptt70 --epochs 1000
elif [ "$1" = "test_penn_baseline" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model stream_lstm --train-path=./data/penn/train.txt --valid-path=./data/penn/valid.txt --test-path=./data/penn/test.txt --hidden-size 600 --embedding-size 300 --batch-size 64 --bptt 70 --dropout 0.5 --lr 1e-4 --is-stream 1 --mode test --checkpoint ./log/final/baseline_penn_batch64_bptt70_Y2020_M3_D13_h15_m31_lr0.0001/checkpoints/best_val_ppl.pth
elif [ "$1" = "train_penn_2cell" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model attention --train-path=./data/penn/train.txt --valid-path=./data/penn/valid.txt --hidden-size 600 --embedding-size 300 --batch-size 64 --bptt 70 --dropout 0.5 --lr 1e-4 --is-stream 1 --log-every 10 --log attention_penn_batch64_bptt70_2cell --epochs 1000 --second-order-size 2
elif [ "$1" = "test_penn_2cell" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model attention --train-path=./data/penn/train.txt --valid-path=./data/penn/valid.txt --test-path=./data/penn/test.txt --hidden-size 600 --embedding-size 300 --batch-size 64 --bptt 70 --dropout 0.5 --lr 1e-4 --is-stream 1  --epochs 1000 --second-order-size 2 --mode test --checkpoint ./log/final/attention_penn_batch64_bptt70_2cell_Y2020_M3_D15_h17_m19_lr0.0001/checkpoints/best_val_ppl.pth
elif [ "$1" = "train_penn_5cell" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model attention --train-path=./data/penn/train.txt --valid-path=./data/penn/valid.txt --hidden-size 600 --embedding-size 300 --batch-size 64 --bptt 70 --dropout 0.5 --lr 1e-4 --is-stream 1 --log-every 10 --log attention_penn_batch64_bptt70_5cell --epochs 1000 --second-order-size 5
elif [ "$1" = "test_penn_5cell" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model attention --train-path=./data/penn/train.txt --valid-path=./data/penn/valid.txt --hidden-size 600 --embedding-size 300 --batch-size 64 --bptt 70 --dropout 0.5 --lr 1e-4 --is-stream 1 --epochs 1000 --second-order-size 5 --mode test --checkpoint ./log/final/attention_penn_batch64_bptt70_5cell_Y2020_M3_D13_h15_m33_lr0.0001/checkpoints/best_val_ppl.pth
    
##########################
### WikiText2
##########################

elif [ "$1" = "train_wiki_baseline" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model stream_lstm --train-path=./data/wikitext-2/train.txt --valid-path=./data/wikitext-2/valid.txt --hidden-size 600 --embedding-size 300 --batch-size 64 --bptt 70 --dropout 0.5 --lr 1e-4 --is-stream 1  --log-every 10 --log baseline_wiki_batch64_bptt70 --epochs 1000
elif [ "$1" = "test_wiki_baseline" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model stream_lstm --train-path=./data/wikitext-2/train.txt --valid-path=./data/wikitext-2/valid.txt --test-path=./data/wikitext-2/test.txt --hidden-size 600 --embedding-size 300 --batch-size 64 --bptt 70 --dropout 0.5 --lr 1e-4 --is-stream 1 --mode test --checkpoint ./log/final//checkpoints/best_val_ppl.pth
elif [ "$1" = "train_wiki_2cell" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model attention --train-path=./data/wikitext-2/train.txt --valid-path=./data/wikitext-2/valid.txt --hidden-size 600 --embedding-size 300 --batch-size 64 --bptt 70 --dropout 0.5 --lr 1e-4 --is-stream 1 --log-every 10 --log attention_wiki_batch64_bptt70_2cell --epochs 1000 --second-order-size 2
elif [ "$1" = "test_wiki_2cell" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model attention --train-path=./data/wikitext-2/train.txt --valid-path=./data/wikitext-2/valid.txt --test-path=./data/wikitext-2/test.txt --hidden-size 600 --embedding-size 300 --batch-size 64 --bptt 70 --dropout 0.5 --lr 1e-4 --is-stream 1  --epochs 1000 --second-order-size 2 --mode test --checkpoint ./log/final/attention_wiki_batch64_70bptt_2cell_Y2020_M3_D15_h23_m45_lr0.0001/checkpoints/best_val_ppl.pth
elif [ "$1" = "train_wiki_5cell" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model attention --train-path=./data/wikitext-2/train.txt --valid-path=./data/wikitext-2/valid.txt --hidden-size 600 --embedding-size 300 --batch-size 64 --bptt 70 --dropout 0.5 --lr 1e-4 --is-stream 1 --log-every 10 --log attention_penn_batch64_bptt70_5cell --epochs 1000 --second-order-size 5
elif [ "$1" = "test_wiki_5cell" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model attention --train-path=./data/wikitext-2/train.txt --valid-path=./data/wikitext-2/valid.txt --hidden-size 600 --embedding-size 300 --batch-size 64 --bptt 70 --dropout 0.5 --lr 1e-4 --is-stream 1 --epochs 1000 --second-order-size 5 --mode test --checkpoint ./log/final/attention_wiki_batch64_70bptt_5cell_Y2020_M3_D15_h22_m12_lr0.0001/checkpoints/best_val_ppl.pth
else
	echo "Invalid Option Selected"
fi
