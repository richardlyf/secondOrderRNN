from dataset import *
import sys
import numpy as np
import torch
import time
from model import *
import torch.nn as nn
from utils import *
from torch.utils.data import Dataset, DataLoader


def test(dataset):
    batch_size = 1
    vocab = dataset.get_vocab()
    criterion = nn.NLLLoss(ignore_index=vocab.pad_id)
    model = LSTMLanguageModel(vocab, hidden_dim=30, batch_size=batch_size, embedding_dim=12)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)  

    model = load_checkpoint("log/parens_m4_batch1_Y2020_M3_D3_h13_m33_lr0.0001/checkpoints/lr0.0001_epoch0.pth", model, "cpu")  
    print(model.state_dict())


    assignments = {0: [0, 1, 2, 3, 4 ,5]}
    model = LSTMLanguageModel2(vocab, hidden_dim=30, batch_size=batch_size, embedding_dim=12, assignments=assignments, num_cells=1)
    model = load_checkpoint("log/parens_m4_mLSTM_1cell_Y2020_M3_D4_h13_m30_lr0.0001/checkpoints/lr0.0001_epoch0.pth", model, "cpu")  
    print(model.state_dict())

    # x, y = next(iter(dataloader))
    # # y = y[:, :5]
    # # y[:, 4] = 0
    # # print(x)
    # # print(y)
    # y = y.view(-1)
    # y_pred = model(x)
    # loss = criterion(y_pred, y)
    # print(loss)
    
    print("-"*80)

    batch_size = 40
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

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
    dataset = ParensDataset("./data/mbounded-dyck-k/m4/train.formal.txt")
    test(dataset)