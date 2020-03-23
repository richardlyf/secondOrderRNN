# Modeling Long-Distance Dependencies with Second-Order LSTMs
Stanford CS224N Natural Language Processing with Deep Learning

We present our results in this [paper](Paper.pdf).

---

## Datasets

mbounded-dyck-k parenthesis dataset is included in the `data/mbounded-dyck-k` directory.

To acquire Penn Treebank and Wiki2 datasets:
```
sh getdata.sh
```

## Training and testing the language model

`run.sh` includes all training and testing commands.

## Flags Clarifications
- When `--log <LOG_NAME>` is specified, the model parameters, checkpoints, and tensorboard log will be saved under `/log/LOG_NAME`. Without the flag, no logs will be recorded.
- If `--is-stream 0` is specified, we treat each line in the dataset as a separate sample sentence. `<start>` and `<end>` tokens are added to each sentence and all sentences are padded to the same length then batched. The number of samples in the dataset must be divisible by the batch size. This option is used for the parenthesis dataset and syntax evaluation dataset.
- If `--is-stream 1` is specified, we treat the entire dataset as one continuous stream. The last hidden state of a batch is used to initialize the LSTM for the next batch. The dataset to padded to a multiple of batch size and bptt. The backprop through time flag `--bptt` determines the sequence length of each batch. This option is used for natural language datasets like Penn Treebank and Wiki2.

## Analysis
- To show the principal angles between sub-spaces (PABS) for pairs of hidden-to-hidden weight matrices, run `compare_weights.py` with a model checkpoint.
- To evaluate the syntax of generated texts, see the [documentation](/LM_syneval/README.md) in `LM_syneval`.
- **(Experimental)** To generate heatmaps, go to the `heatmap` branch and run:
```
python3 run.py --mode train --model attention --train-path=./data/wikitext-2/train.txt --valid-path=./data/wikitext-2/valid.txt --checkpoint ./LOG_PATH/best_val_ppl.pth --hidden-size 600 --embedding-size 300 --batch-size 10 --bptt 70 --dropout 0.5 --lr 1e-4 --is-stream 1 --epochs 1000 --second-order-size 5 > ./data/heatmaps/sample_att_wiki_5cell.txt
```
- Mode should be set to train because we need temperature set to high to be able to mark how confident each word is assigned to each cell. The batch info and attention info is printed and piped to an output file.
We use the output file to generate heatmaps.
```
python3 utils/heatmap.py --input data/heatmaps/sample_att_wiki_5cell.txt --output data/heatmaps/ --prefix att_wiki_5cell
```

## Future Work
We have modified the baseline and attention model to work with RNN instead of LSTM in the `rnn` branch. However we did not have time to verify the correctness of the code and aggregate results for those models.

## Sanitychecks
- `test_dataset.py` tests the validity of the dataset object
- `test_eval.py` tests the validity of the evaluation metrics
- `test_model.py` tests the validity of the models
