from dataset import *

def test_penntreebank():
    print ("-"*80)
    path = "data/penn/train.txt"
    batch_size = 64
    bptt = 70    

    words = []
    # read in first 3*bptt words in the dataset
    with open(path, 'r') as file:
        while len(words) < 3*bptt:
            words += file.readline().split()

    reference = " ".join(words[0:3*bptt])

    print("Running Sanity Check: IterableDataset")
    treebank1 = PennTreebankDataset(path, batch_size=batch_size, bptt=bptt)
    vocab = Vocab(file_path="data/penn/train.json")
    # print the first line of each batch to confirm that they are contiguous
    words = []
    for batch_idx, (x, y) in enumerate(treebank1):
        words += [vocab.id2word[w] for w in x[0]]
        if batch_idx == 2: break
    compare1 = " ".join(words)
    assert(compare1 == reference)

    print("Running Sanity Check: Dataset")
    treebank2 = PennTreebankDataset2(path, batch_size=batch_size, bptt=bptt)
    words = []
    for i in range(0, batch_size*3, batch_size):
        x, y = treebank2[i]
        words += [vocab.id2word[w] for w in x]
    compare2 = " ".join(words)
    assert(compare2 == reference)

    print("All Sanity Checks Passed!")
    print ("-"*80)

if __name__ == '__main__':
    test_penntreebank()
