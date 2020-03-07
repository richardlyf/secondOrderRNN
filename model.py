import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from paren_mLSTM import paren_mLSTM
from paren_mLSTM import test_LSTM

def ModelChooser(model_name, **kwargs):
    """
    This function takes in a model name and returns its corresponding model
    """
    if model_name == "baseline_lstm":
        return LSTMLanguageModel(**kwargs)
    if model_name == "mLSTM":
        # Group by a paren and b paren
        assignments = {
            0: [0, 2, 4],
            1: [1, 3, 5]
        }
        kwargs["assignments"] = assignments
        return AssignmentLanguageModel(**kwargs)
    if model_name == "test_lstm":
        # Group by a paren and b paren
        assignments = {
            0: [0, 2, 4],
            1: [1, 3, 5]
        }
        kwargs["assignments"] = assignments
        kwargs["num_cells"] = 1
        return TESTLanguageModel(**kwargs)


class LSTMLanguageModel(nn.Module):
    """ simple LSTM neural network language model """     
    def __init__(self, vocab, hidden_size=100, embed_size=12, dropout_rate=0.5, num_layers=1, **kwargs):
        super(LSTMLanguageModel, self).__init__()
        
        vocab_size = len(vocab)
        self.embeddings = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(
            input_size=embed_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True)

        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.drop = nn.Dropout(p=dropout_rate)
        
    def forward(self, x):
        """
        Predict, return hidden state so it can be used to intialize the next hidden state 
        @param x: (batch_size, sequence_length)
        """
        # (batch_size, sequence_length, embed_size)
        embedded = self.embeddings(x)
        # (batch_size, sequence_length, embed_size)
        lstm_output, hdn = self.lstm(embedded)
        # (batch_size * sequence_length, hidden_size)
        reshaped = lstm_output.reshape(-1, lstm_output.size(2))
        decoded = self.linear(reshaped)
        # (batch_size * sequence_length, vocab_size)
        logits = F.log_softmax(decoded, dim=1)
                
        return logits


class AssignmentLanguageModel(nn.Module):
    """ second order LSTM language model with explicit assignment """     
    def __init__(self, vocab, hidden_size=100, embed_size=12, dropout_rate=0.5, \
                device=None, assignments=None, num_cells=2, **kwargs):
        super(AssignmentLanguageModel, self).__init__()
        
        vocab_size = len(vocab)
        self.embeddings = nn.Embedding(vocab_size, embed_size)

        # Here we introduce the mLSTM
        self.lstm = paren_mLSTM(
            embed_size=embed_size, 
            hidden_size=hidden_size, 
            assignments=assignments, 
            num_cells=num_cells, 
            device=device)

        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """
        Predict, return hidden state so it can be used to intialize the next hidden state 
        @param x: (batch_size, sequence_length)
        """
        embedded = self.embeddings(x)
        lstm_output, hdn = self.lstm(x, embedded)
        reshaped = lstm_output.view(-1, lstm_output.size(2))
        decoded = self.linear(reshaped)
        # (batch_size * sequence_length, vocab_size)
        logits = F.log_softmax(decoded, dim=1)
                
        return logits


class TESTLanguageModel(nn.Module):
    """ language model for testing parens_mLSTM """     
    def __init__(self, vocab, hidden_size=100, embed_size=12, device=None, assignments=None, num_cells=2, **kwargs):
        super(TESTLanguageModel, self).__init__()
        
        vocab_size = len(vocab)
        self.embeddings = nn.Embedding(vocab_size, embed_size)

        # Here we introduce the mLSTM
        self.lstm = test_LSTM(
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab=vocab,
            assignments=assignments,
            num_cells=num_cells,
            device=device)

        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        

    def forward(self, x):
        """
        Predict, return hidden state so it can be used to intialize the next hidden state 
        @param x: (batch_size, sequence_length)
        """
        embedded = self.embeddings(x)
        lstm_output, hdn = self.lstm(x, embedded)
        reshaped = lstm_output.view(-1, lstm_output.size(2))
        decoded = self.linear(reshaped)
        # (batch_size * sequence_length, vocab_size)
        logits = F.log_softmax(decoded, dim=1)
                
        return logits
