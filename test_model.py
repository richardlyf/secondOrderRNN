import sys
import numpy as np
import torch
import time
import torch.nn as nn
from utils.utils import *
from torch.utils.data import Dataset, DataLoader
from model.model import *
from model.dataset import *
from model.attentionSecondOrderLSTM import AttentionSecondOrderLSTM


def test(dataset):
    batch_size = 1
    vocab = dataset.get_vocab()
    criterion = nn.NLLLoss(ignore_index=vocab.pad_id)
    model = LSTMLanguageModel(vocab, hidden_size=30, batch_size=batch_size, embed_size=12)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)  
    model.lstm = AttentionSecondOrderLSTM(second_order_size=2, input_size=12, hidden_size=30)
    

    x, y = next(iter(dataloader))
    # y = y[:, :5]
    # y[:, 4] = 0
    # print(x)
    # print(y)
    y = y.view(-1)
    y_pred, _ = model(x, 0.0001)
    loss = criterion(y_pred, y)
    print(loss)


    # x, y = next(iter(dataloader))
    # # y = y[:, :5]
    # # y[:, 4] = 0
    # # print(x)
    # # print(y)
    # y = y.view(-1)
    # y_pred = model(x)
    # loss = criterion(y_pred, y)
    # print(loss)
    
    # assert (np.allclose(y_pred, y_pred1[:len(y_pred)])), "{}".format(y_pred - y_pred1[:len(y_pred)])


if __name__ == "__main__":
    dataset = CustomDataset("./data/mbounded-dyck-k/m4/train.formal.txt")
    test(dataset)