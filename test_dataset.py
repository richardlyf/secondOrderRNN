from dataset import *

def test_penntreebank():
    print ("-"*80)
    path = "data/penn/train.txt"

    # CONSTANTS
    batch_size = 64
    bptt = 70    
    words_per_batch = batch_size * bptt
    n_batch = 199

    # OPTIONS
    n_seq = 10 # how many sequences to compare
    batch_num = 5 # which batch to test (1 through n_batch)
    
    # read in first nseq*bptt words in the batch
    words = []
    start = batch_num * bptt * n_batch
    end = start + (n_seq * bptt)
    with open(path, 'r') as file:
        while len(words) < end:
            words += file.readline().split()
    reference = " ".join(words[start:end])

    print("Running Sanity Check: Dataset")
    treebank = PennTreebankDataset(path, batch_size=batch_size, bptt=bptt)
    vocab = Vocab(file_path="data/penn/train.json")
    words = []
    for i in range(0, batch_size*n_seq, batch_size):
        x, y = treebank[i + batch_num]
        words += [vocab.id2word[w] for w in x]
    compare= " ".join(words)
    assert(compare == reference)

    print("All Sanity Checks Passed!")
    print ("-"*80)

if __name__ == '__main__':
    test_penntreebank()
