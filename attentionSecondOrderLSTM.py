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
    def __init__(self, vocab, second_order_size, input_size, hidden_size, bias=True, **kwargs):
        super(AttentionSecondOrderLSTMCell, self).__init__(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.second_order_size = second_order_size

        self.secondOrderLSTMCells = nn.ModuleList([nn.LSTMCell(input_size, hidden_size, bias)\
            for i in range(second_order_size)])
        # Each cell has an attention vector V_i of size hidden_size so together they're a matrix 
        # of size (hidden_size, second_order_size) which is essentially a linear layer with no bias
        self.attentionScores = torch.nn.Linear(hidden_size, second_order_size, bias=False)
        self.softmax = nn.Softmax(dim=1)
            
    def temperature_softmax(self, x, temperature):
        """
        When the temperature is 1 (hot), this function behaves like a normal softmax
        When the temperature is close to 0 (cold), this function puts all probability mass
        on only one of the secondOrder cells
        When the temperature is 0.1, the softmax result is effectively one-hot
        There no lower bound to how close the temperature can get to 0

        @param x: (batch_size, second_order_size)
        @param temperature: [1 - 0)
        """
        return self.softmax(x / temperature)

    def forward(self, input, dec_states=None, temperature):
        """
        We compute attention using the hidden state h_t. Initially the temperature should be high, so that the attention
        score is widely distributed. The input embedding is passed through each LSTMCell to obtain second_order_size amount
        of updated hidden states h_{t+1}. The updated hidden states are then weighted by the attention distribution and 
        summed to form a single next hidden state. As training goes on, the temperature should decrease, so the attention
        distribution would only favor one of the LSTMCell's output and the updated hidden state would effectively be the output
        hidden state of that LSTMCell.

        At time sequence t, given hidden state h_t, we first compute the attention score 
        attscore_t,i = h_t,i * V_i so that attscore_t has shape (second_order_size, ); i = {1, ... , second_order_size}
        We then compute the attention distribution alpha_t = temperature_softmax(attscore_t)
        The attention distribution is used to weight the output hidden states of LSTMCells. We get a single updated hidden state
        h_{t+1} = \sum_{i=1}^{second_order_size} alpha_t,i * h_{t+1},i
        The cell state is updated in the same fashion where the final c_{t+1} is the weighted average of new cells states

        @param input Tensor(batch_size, embed_size): Input embedding of a batch for one time sequence
        @param dec_states Tuple(Tensor(batch_size, hidden_size), *): Tuple of hidden state and cell state
        with the same shape
        @param temperature [1 - 0): Should decrease as the model continues to train. See documentation on temperature_softmax above
        @return updated_states Tuple(Tensor(batch_size, hidden_size), *): Tuple of updated hidden and cell state
        """
        self.check_forward_input(input)
        batch_size, embed_size = input.shape

        if dec_states is None:
            hidden = torch.zeros(batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
            cell = torch.zeros(batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
            dec_states = (hidden, cell)

        # (batch_size, hidden_size)
        hidden_state, cell_state = dec_states
        # Compute attention score of cells
        # (batch_size, second_order_size)
        attscore = self.attentionScores(hidden_state)
        # Compute attention distribution
        alpha = self.temperature_softmax(attscore, temperature)

        # Compute updated hidden and cell state using each LSTMCell
        updated_hidden = torch.zeros(batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
        updated_cell = torch.zeros(batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
        for cell_idx in range(self.second_order_size):
            lstm_hidden, lstm_cell = self.secondOrderLSTMCells[cell_idx](input, dec_states)
            updated_hidden = alpha[cell_idx] * lstm_hidden
            updated_cell = alpha[cell_idx] * lstm_cell






class AttentionSecondOrderLSTM(nn.Module):
    """
    Second order LSTM that uses SecondOrderLSTMCell
    """
    def __init__(self):
        pass

    def forward(self, input):
        """
        @param input Tensor(batch_size, seq_len, embed_size)
        """
        pass


# TESTs
train_dataset = ParensDataset("./data/mbounded-dyck-k/m4/train.formal.txt")
vocab = train_dataset.get_vocab()
attlstm = AttentionSecondOrderLSTMCell(vocab, 2, input_size=5, hidden_size=6)
for x in attlstm.parameters():
    print(x)