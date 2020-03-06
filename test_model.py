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



####### TEST mLSTM but only one cell ###########
## TODO Edit this to have one cell only

class test_LSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, device, bias=True):
        super(test_LSTM, self).__init__()
        # Checking proper set up
        if embed_size < 1 or hidden_size < 1:
            raise ValueError("'num_cells', 'embd_dim', and 'hidden_layers' must be >= 1")

        # Initialize a single LSTMCell
        self.lstm_cell = nn.LSTMCell(embed_size, hidden_size, bias=bias)

        # Preserve arguments
        self.vocab = vocab
        self.hidden_size = hidden_size 

    def forward(self, input, input_embed, dec_states=None):
        """
        Fun stuff going on here. The difficulty lies in not being able to apply just one of our LSTM cells to 
        an entire sample, but still wanting to batch. So, we construct a few different pieces
        to keep track of the hidden and cell state of each sample in the batch at every time step.
        Each time step follows this process:
        (1) Break the current time step into different mini batches based on the LSTMCell they 
            will be applied to. 
        (2) Assemble the previous states for the mini_batch to feed into the LSTM.
        (3) Run the specific LSTMCell on the mini_batch
        (4) Place the outputs into their appropriate location in dec_states
        -------------------------------
        @param input Tensor(batch_size, seq_len): input of size (seq_len, batch_size)
        @param input_embed Tensor(seq_len, batch_size, embed_size): Embedding of the entire input. Input to LSTMCells
        @param dec_states Tensor(batch_size, 2, hidden_size): hidden state and cell state used to initalize the LSTMCells

        @return combined_outputs Tensor(seq_len, batch_size, hidden_size): Output tensor of size (batch_size, 2 * hidden_size)
        """
        input = input.cpu().detach().numpy()
        batch_size, seq_len = input.shape
        
        # dec_states dim=0 will hold the two Tensors representing the 
        # sentences current hidden and cell states at each time step
        if dec_states == None:
            dec_states = torch.zeros((batch_size, 2, self.hidden_size), device=input_embed.device)

        # outputs to be returned at the end of the function and used to make prediction
        combined_outputs = torch.empty((seq_len, batch_size, self.hidden_size), device=input_embed.device)
        
        # initialize hidden and cell
        hidden, cell = torch.split(dec_states, 1, dim=1)
        # (sub_batch_size, hidden_size)
        hidden = torch.squeeze(hidden, dim=1)
        cell = torch.squeeze(cell, dim=1)

        for seq_idx in range(seq_len):
            # All embeddings at the current time-step
            # (batch_size, embed_size)
            idx_input_embeddings = input_embed[seq_idx]
            # update hidden and cell
            hidden, cell = self.lstm_cell(idx_input_embeddings, (hidden, cell))
            # upload combined outputs
            combined_outputs[seq_idx, :] = hidden

        return combined_outputs, (hidden, cell)