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
        assignments = {
            0: [0, 2, 3],
            1: [1, 4, 5]
        }
        kwargs["assignments"] = assignments
        return LSTMLanguageModel2(**kwargs)
    if model_name == "test_lstm":
        return TESTLanguageModel(**kwargs)


class LSTMLanguageModel(nn.Module):
    """ simple LSTM neural network language model """     
    def __init__(self, vocab, hidden_dim=100, batch_size=10, embedding_dim=12, dropout_rate=0.5, num_layers=1, **kwargs):
        super(LSTMLanguageModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        
        vocab_size = len(vocab)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(
            input_size = embedding_dim, 
            hidden_size = self.hidden_dim, 
            num_layers = num_layers,
            dropout = dropout_rate,
            batch_first=True)

        self.linear = nn.Linear(
            in_features = self.hidden_dim, 
            out_features = vocab_size)

        self.drop = nn.Dropout(p=dropout_rate)
        

    def forward(self, x, train=True):
        """
        Predict, return hidden state so it can be used to intialize the next hidden state 
        @param x: (batch_size, sequence_length)
        """
        # (batch_size, sequence_length, embedding_dim)
        embedded = self.embeddings(x)
        # embedded = self.drop(embedded) if train else embedded
        
        # (batch_size, sequence_length, embedding_dim)
        lstm_output, hdn = self.lstm(embedded)

        # (batch_size * sequence_length, hidden_size)
        reshaped = lstm_output.reshape(-1, lstm_output.size(2))
        # dropped = self.drop(reshaped) if train else reshaped
        
        decoded = self.linear(reshaped)
        # (batch_size * sequence_length, vocab_size)
        logits = F.log_softmax(decoded, dim=1)
                
        return logits


class LSTMLanguageModel2(nn.Module):
    """ simple LSTM neural network language model """     
    def __init__(self, vocab, hidden_dim=100, batch_size=10, embedding_dim=12, dropout_rate=0.5, \
                num_layers=1, device=None, assignments=None, num_cells=2, **kwargs):

        super(LSTMLanguageModel2, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        
        vocab_size = len(vocab)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Here we introduce the mLSTM
        self.lstm = paren_mLSTM(
            embed_size=embedding_dim,
            hidden_size=hidden_dim,
            vocab=vocab,
            assignments=assignments,
            num_cells=num_cells,
            device=device)

        self.linear = nn.Linear(
            in_features = self.hidden_dim, 
            out_features = vocab_size)

        self.drop = nn.Dropout(p=dropout_rate)
        

    def forward(self, x, train=True):
        """
        Predict, return hidden state so it can be used to intialize the next hidden state 
        @param x: (batch_size, sequence_length)
        """
        embedded = self.embeddings(x)
        # embedded = self.drop(embedded) if train else embedded
        
        lstm_output, hdn = self.lstm(x, embedded)

        reshaped = lstm_output.view(-1, lstm_output.size(2))
        # dropped = self.drop(reshaped) if train else reshaped
        
        decoded = self.linear(reshaped)
        # (batch_size * sequence_length, vocab_size)
        logits = F.log_softmax(decoded, dim=1)
                
        return logits


class TESTLanguageModel(nn.Module):
    """ simple LSTM neural network language model """     
    def __init__(self, vocab, hidden_dim=100, batch_size=10, embedding_dim=12, device=None, **kwargs):

        super(TESTLanguageModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        
        vocab_size = len(vocab)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Here we introduce the mLSTM
        self.lstm = test_LSTM(
            embed_size=embedding_dim,
            hidden_size=hidden_dim,
            vocab=vocab,
            device=device)

        self.linear = nn.Linear(
            in_features = self.hidden_dim, 
            out_features = vocab_size)
        

    def forward(self, x, train=True):
        """
        Predict, return hidden state so it can be used to intialize the next hidden state 
        @param x: (batch_size, sequence_length)
        """
        embedded = self.embeddings(x)
        # embedded = self.drop(embedded) if train else embedded
        
        lstm_output, hdn = self.lstm(x, embedded)

        reshaped = lstm_output.view(-1, lstm_output.size(2))
        # dropped = self.drop(reshaped) if train else reshaped
        
        decoded = self.linear(reshaped)
        # (batch_size * sequence_length, vocab_size)
        logits = F.log_softmax(decoded, dim=1)
                
        return logits
