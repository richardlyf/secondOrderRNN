import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#Debug
from dataset import *

class customCellBase(nn.Module):
    """
    Generic cell base that handles error checking and weight initialization
    Based on nn.RNNCellBase: 
    https://github.com/pytorch/pytorch/blob/ace2b4f37f26b8d7782dd6f1ce7e3738f8dc0dec/torch/nn/modules/rnn.py#L741
    """
    def __init__(self, input_size, hidden_size):
        super(customCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        # type: (Tensor, Tensor, str) -> None
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            print(weight.size())
            weight.uniform_(-stdv, stdv)


class AttentionSecondOrderLSTMCell(customCellBase):
    """
    For second order LSTM, we apply different weight matrix W to different inputs
    Each weight matrix W is repesented by an individual LSTMCell.
    Each LSTMCell has a corresponding attention vector V_i that is multiplied with
    the hidden state to compute the attention score.    
    """
    def __init__(self, vocab, secondOrderSize, input_size, hidden_size, bias=True, **kwargs):
        super(AttentionSecondOrderLSTMCell, self).__init__(input_size, hidden_size)

        self.secondOrderLSTMCells = []
        self.attentionVectors = []
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for i in range(secondOrderSize):
            self.secondOrderLSTMCells.append(nn.LSTMCell(input_size, hidden_size, bias))
            self.attentionVectors.append(nn.Parameter(torch.Tensor(hidden_size).uniform_(-stdv, stdv)))
        self.test = nn.Parameter(torch.Tensor(hidden_size).uniform_(-stdv, stdv))
        self.softmax = nn.Softmax(dim=1)
            
    def temperature_softmax(self, x, temperature):
        """
        When the temperature is 1 (hot), this function behaves like a normal softmax
        When the temperature is close to 0 (cold), this function puts all probability mass
        on only one of the secondOrder cells
        When the temperature is 0.1, the softmax result is effectively one-hot
        There no lower bound to how close the temperature can get to 0

        @param x: (batch_size, secondOrderSize)
        @param temperature: [1 - 0)
        """
        return self.softmax(x / temperature)

    def forward(self, input, dec_states=None):
        self.check_forward_input(input)


class AttentionSecondOrderLSTM(nn.Module):
    """
    Second order LSTM that uses SecondOrderLSTMCell
    """
    def __init__(self):
        pass

    def forward(self, x):
        pass


# TESTs
train_dataset = ParensDataset("./data/mbounded-dyck-k/m4/train.formal.txt")
vocab = train_dataset.get_vocab()
attlstm = AttentionSecondOrderLSTMCell(vocab, 2, input_size=5, hidden_size=6)
for x in attlstm.parameters():
    print(x)