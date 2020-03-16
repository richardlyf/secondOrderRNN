from model.dataset import *
import math
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def test_penntreebank():
    print ("-"*80)
    path = "data/penn/train.txt"

    # CONSTANTS
    batch_size = 64
    bptt = 70    
    print("Running Sanity Check: Dataset")

    # Open treebank text file
    words = []
    with open(path, 'r') as f:
        for line in f:
            words += line.split()

    treebank = CustomDataset(path, batch_size=batch_size, bptt=bptt, is_stream=True)
    dataloader = DataLoader(treebank, batch_size=batch_size, shuffle=False, num_workers=4)
    vocab = Vocab(file_path="data/penn/train_stream.json")
    
    total_num_lines = math.ceil(len(words) / bptt)
    num_batches = math.ceil(total_num_lines / batch_size)
    print("Total number of batches: ", num_batches)

    words += ['<pad>'] * (num_batches * bptt * batch_size - len(words))

    """
    if batch_size = 3
    x <
    x -
    x 
    x <
    x -
    x 
    x <
    x -
    x
    """
    for i, (batch_x, batch_y) in enumerate(tqdm(dataloader)):
        # x -> (64, bptt)
        # x[0] -> first bptt words in file = 0 * ceil(total_num_lines / batch_size)
        # x[1] -> 1 * ceil(total_num_lines / batch_size)
        # For each line of length 70 in the batch
        for batch_row_idx, (x, y) in enumerate(zip(batch_x, batch_y)):
            x = x.tolist()
            y = y.tolist()
            x = [vocab.id2word[index] for index in x]
            y = [vocab.id2word[index] for index in y]

            seq_len = bptt
            if i == num_batches - 1:
                seq_len = bptt - 1

            gt_x = words[(batch_row_idx * num_batches + i) * bptt: (batch_row_idx * num_batches + i) * bptt + seq_len]
            gt_y = words[(batch_row_idx * num_batches + i) * bptt + 1: (batch_row_idx * num_batches + i) * bptt + seq_len + 1]
 
            assert (x == gt_x), "\n At batch {}, row {}, \n Expected: \n {} \n\n got: \n {}".format(i, batch_row_idx, x, gt_x) 
            assert (y == gt_y), "\n At batch {}, row {}, \n Expected: \n {} \n\n got: \n {}".format(i, batch_row_idx, y, gt_y) 


    print("All Sanity Checks Passed!")
    print ("-"*80)


if __name__ == '__main__':
    test_penntreebank()
