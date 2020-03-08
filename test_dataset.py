from dataset import *

def test_penntreebank():
    path = "data/penn/train.txt"
    vocab = Vocab(file_path="data/penn/train.json")
    treebank = PennTreebankDataset(path, batch_size = 64, bptt=70)
    # print the first line of each batch to confirm that they are contiguous
    for batch_idx, (x, y) in enumerate(treebank):
    	print(" ".join([vocab.id2word[w] for w in x[0]]))
    	if batch_idx == 3: break

test_penntreebank()
