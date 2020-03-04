from dataset import *
import sys
import numpy as np
import torch
import time
from model import LSTMLanguageModel
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def test(dataset):
    batch_size = 1
    vocab = dataset.get_vocab()
    criterion = nn.NLLLoss(ignore_index=vocab.pad_id)
    model = LSTMLanguageModel(vocab, hidden_dim=30, batch_size=batch_size, embedding_dim=12)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)    

    x, y = next(iter(dataloader))
    # y = y[:, :5]
    # y[:, 4] = 0
    # print(x)
    # print(y)
    y = y.view(-1)
    y_pred = model(x)
    loss = criterion(y_pred, y)
    print(loss)
    
    print("-"*80)

    batch_size = 40
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    x, y = next(iter(dataloader))
    # y = y[:, :5]
    # y[:, 4] = 0
    # print(x)
    # print(y)
    y = y.view(-1)
    y_pred = model(x)
    loss = criterion(y_pred, y)
    print(loss)
    
    # assert (np.allclose(y_pred, y_pred1[:len(y_pred)])), "{}".format(y_pred - y_pred1[:len(y_pred)])



        



if __name__ == "__main__":
    dataset = ParensDataset("./data/mbounded-dyck-k/m4/train.formal.txt")
    test(dataset)