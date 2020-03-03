import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def ModelChooser(model_name, **kwargs):
    """
    This function takes in a model name and returns its corresponding model
    """
    if model_name == "baseline_lstm":
        return LSTMLanguageModel(**kwargs)


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
            dropout = dropout_rate)

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
        # (sequence_length, batch_size, embedding_dim) to fit LSTM input shape requirement
        embedded = torch.transpose(embedded, 0, 1).contiguous()
        
        lstm_output, hdn = self.lstm(embedded)
        reshaped = lstm_output.view(-1, lstm_output.size(2))
        # dropped = self.drop(reshaped) if train else reshaped
        
        decoded = self.linear(reshaped)
        # (batch_size * sequence_length, vocab_size)
        logits = F.log_softmax(decoded, dim=1)
                
        return logits