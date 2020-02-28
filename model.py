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
    def __init__(self, TEXT, hidden_dim=100, batch_size=10, embedding_dim=12, dropout_rate=0.5, is_parens=True, **kwargs):
        super(LSTMLanguageModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        
        vocab_size = len(TEXT.vocab)
        if not is_parens:
            embedding_dim = TEXT.vocab.vectors.shape[1]
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        if not is_parens:
            self.embeddings.weight.data.copy_(TEXT.vocab.vectors)
        
        self.lstm = nn.LSTM(
            input_size = embedding_dim, 
            hidden_size = self.hidden_dim, 
            num_layers = 2,
            dropout = dropout_rate)

        self.linear = nn.Linear(
            in_features = self.hidden_dim, 
            out_features = vocab_size)

        self.drop = nn.Dropout(p = dropout_rate)


    def init_hidden(self):
        direction = 2 if self.lstm.bidirectional else 1
        return (
            Variable(torch.zeros(
                direction*self.lstm.num_layers, 
                self.batch_size, 
                self.hidden_dim)), 
            Variable(torch.zeros(
                direction*self.lstm.num_layers, 
                self.batch_size, 
                self.hidden_dim)))


    def detach_hidden(self, hidden):
        """ util function to keep down number of graphs """
        return tuple([h.detach() for h in hidden])
        

    def forward(self, x, hidden, train=True):
        """ predict, return hidden state so it can be used to intialize the next hidden state """
        embedded = self.embeddings(x)
        embedded = self.drop(embedded) if train else embedded
        
        lstm_output, hdn = self.lstm(embedded, hidden)
        reshaped = lstm_output.view(-1, lstm_output.size(2))
        dropped = self.drop(reshaped) if train else reshaped
        
        decoded = self.linear(dropped)
        
        logits = F.log_softmax(decoded, dim = 1)
                
        return logits, self.detach_hidden(hdn)    