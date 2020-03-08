import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from paren_mLSTM import paren_mLSTM, test_LSTM
from attentionSecondOrderLSTM import AttentionSecondOrderLSTM
from dataset import get_glove


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
    if model_name == "attention":
        return AttentionLanguageModel(**kwargs)


    if model_name == "test_lstm":
        # Group by a paren and b paren
        assignments = {
            0: [0, 2, 4],
            1: [1, 3, 5]
        }
        kwargs["assignments"] = assignments
        kwargs["num_cells"] = 1
        return TESTLanguageModel(**kwargs)
    if model_name == "ptb_lstm":
        kwargs["embed_path"] = "data/.vector_cache/miniglove.txt"
        return LSTMNaturalLanguageModel(**kwargs)

# Updated to return hidden state, to be used for PTB baseline, but could be incorporated
# into all models
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

    def forward(self, x, init_state):
        """
        @param x: (batch_size, sequence_length)
        @param init_state: Tuple(Tensor, Tensor)
            each Tensor (batch_size, hidden_size)
        """
        # (batch_size, sequence_length, embed_size)
        embedded = self.embeddings(x)
        # (batch_size, sequence_length, embed_size)
        lstm_output, ret_state = self.lstm(embedded, init_state)
        # (batch_size * sequence_length, hidden_size)
        reshaped = lstm_output.reshape(-1, lstm_output.size(2))
        decoded = self.linear(reshaped)
        # (batch_size * sequence_length, vocab_size)
        logits = F.log_softmax(decoded, dim=1)
                
        return logits, ret_state



class LSTMNaturalLanguageModel(nn.Module):
    """ simple LSTM neural network language model """     
    def __init__(self, vocab, hidden_size=100, embed_size=300, dropout_rate=0.5, num_layers=1, embed_path=None, device=None, batch_size=None,**kwargs):
        super(LSTMNaturalLanguageModel, self).__init__()
        
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        vocab_size = len(vocab)

        # load glove embeddings and initialize 
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        glove_embeddings = get_glove(embed_path, vocab)
        glove_tensor = torch.tensor(glove_embeddings, dtype=torch.float32, device=device)
        self.embeddings.weight = nn.Parameter(glove_tensor)
        self.embeddings.weight.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=embed_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True)

        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.drop = nn.Dropout(p=dropout_rate)

    def init_lstm_state(self, device):
        zero_hidden = torch.zeros((
            self.num_layers, 
            self.batch_size, 
            self.hidden_size), device=device) 
        zero_cell = torch.zeros((
            self.num_layers, 
            self.batch_size, 
            self.hidden_size), device=device) 
        return (zero_hidden, zero_cell)

    def detach_hidden(self, hidden):
        """ util function to keep down number of graphs """
        return tuple([h.detach() for h in hidden])

    def forward(self, x, init_state):
        """
        @param x: (batch_size, sequence_length)
        @param init_state: Tuple(Tensor, Tensor)
            each Tensor (batch_size, hidden_size)
        """
        # (batch_size, sequence_length, embed_size)
        embedded = self.embeddings(x)
        # (batch_size, sequence_length, embed_size)
        lstm_output, ret_state = self.lstm(embedded, init_state)
        # (batch_size * sequence_length, hidden_size)
        reshaped = lstm_output.reshape(-1, lstm_output.size(2))
        decoded = self.linear(reshaped)
        # (batch_size * sequence_length, vocab_size)
        logits = F.log_softmax(decoded, dim=1)
                
        return logits, self.detach_hidden(ret_state)


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
        @param x: (batch_size, sequence_length)
        """
        embedded = self.embeddings(x)
        lstm_output, hdn = self.lstm(x, embedded)
        reshaped = lstm_output.view(-1, lstm_output.size(2))
        decoded = self.linear(reshaped)
        # (batch_size * sequence_length, vocab_size)
        logits = F.log_softmax(decoded, dim=1)
                
        return logits


class AttentionLanguageModel(nn.Module):
    """ second order attention language model """     
    def __init__(self, vocab, second_order_size=2, hidden_size=100, embed_size=12, device=None, temp_decay=0.9, temp_decay_interval=None, **kwargs):
        """
        @param temp_decay: Everything temperature decays, the temperature is multiplied by this value
        @param temp_decay_interval: The temperature will decay after forward() is called temp_decay_interval times. Should be set to
        len(dataloader) when training. When testing, should be set to None so temperature doesn't decay.
        """
        super(AttentionLanguageModel, self).__init__()
        
        vocab_size = len(vocab)
        self.embeddings = nn.Embedding(vocab_size, embed_size)

        self.lstm = AttentionSecondOrderLSTM(
            second_order_size=second_order_size, 
            input_size=embed_size, 
            hidden_size=hidden_size)

        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        
        self.counter = 0
        self.train_temperature = 1
        self.test_temperature = 1e-5
        self.temp_decay = temp_decay
        self.temp_decay_interval = temp_decay_interval
        
    def forward(self, x):
        """
        Predict, return hidden state so it can be used to intialize the next hidden state 
        @param x: (batch_size, sequence_length)
        """
        if self.training:
            temperature = self.train_temperature
            self.counter += 1
            # At the end of every epoch decrease temperature
            assert (self.temp_decay_interval is not None), "Did not set temp_decay_interval for training"
            if self.counter == self.temp_decay_interval:
                self.counter = 0
                self.train_temperature *= self.temp_decay
                print("Temperature decayed to ", self.train_temperature)
        # For eval temperature should be set to default low
        else:
            temperature = self.test_temperature

        embedded = self.embeddings(x)
        lstm_output, hdn = self.lstm(embedded, temperature)
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
