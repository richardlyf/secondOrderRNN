import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.paren_mLSTM import paren_mLSTM
from model.attentionSecondOrderLSTM import AttentionSecondOrderLSTM
from model.attentionSecondOrderRNN import AttentionSecondOrderRNN
from model.dataset import get_glove


def ModelChooser(model_name, **kwargs):
    """
    This function takes in a model name and returns its corresponding model

    Note: currently pre-trained embeddings are not supported for attention
    or assignment models; to use pre-trained embeddings for a baseline model,
    fill in the embed_path parameter in kwargs
    """
    if model_name == "baseline_lstm":
        # fill in path to pretrained vector embeddings here
        # if embed_path is left empty, model will train without embeddings
        # kwargs["embed_path"] = "data/vectors/glove.840B.300d.txt"
        return LSTMLanguageModel(**kwargs)
    if model_name == "baseline_rnn":
        # fill in path to pretrained vector embeddings here
        # if embed_path is left empty, model will train without embeddings
        # kwargs["embed_path"] = "data/vectors/glove.840B.300d.txt"
        return RNNLanguageModel(**kwargs)
    if model_name == "attention_lstm":
        return AttentionLSTMLanguageModel(**kwargs)
    if model_name == "attention_rnn":
        return AttentionRNNLanguageModel(**kwargs)
    if model_name == "assignent_lstm":
        # Group by a paren and b paren
        assignments = {
            0: [2, 3, 4, 6],
            1: [0, 1, 5, 7]
        }
        kwargs["assignments"] = assignments
        return AssignmentLSTMLanguageModel(**kwargs)

class LanguageModelBase(nn.Module):
    """
    Contains functions needed by all models for the penn tree bank dataset
    """
    def __init__(self, batch_size, hidden_size, num_layers=None):
        super(LanguageModelBase, self).__init__()
        if num_layers is None:
            self.hidden_shape = (batch_size, hidden_size)
        else:
            self.hidden_shape = (num_layers, batch_size, hidden_size)

    def detach_hidden(self, hidden):
        """
        Detaches the lstm states before returning them in forward()
        """
        if isinstance(hidden, torch.Tensor):
            return hidden.detach()
        else:
            return tuple([h.detach() for h in hidden])


class LSTMLanguageModelBase(LanguageModelBase):
    """
    Contains functions needed by all models for the penn tree bank dataset
    """
    def __init__(self, batch_size, hidden_size, num_layers=None):
        super(LSTMLanguageModelBase, self).__init__()

    def init_lstm_state(self, device):
        """
        Initialize hidden state and cell state to zeros
        """
        zero_hidden = torch.zeros(self.hidden_shape, device=device) 
        zero_cell = torch.zeros(self.hidden_shape, device=device) 
        return (zero_hidden, zero_cell)

    def generate_context_state(self, device):
        """
        Initialize hidden state and cell state to random
        """
        context_hidden = torch.randn(self.hidden_shape, device=device) 
        context_cell = torch.randn(self.hidden_shape, device=device) 
        return (context_hidden, context_cell)


class RNNLanguageModelBase(LanguageModelBase):
    """
    Contains functions needed by all models for the penn tree bank dataset
    """
    def __init__(self, batch_size, hidden_size, num_layers=None):
        super(RNNLanguageModelBase, self).__init__()

    def init_lstm_state(self, device):
        """
        Initialize hidden state to zeros
        """
        return torch.zeros(self.hidden_shape, device=device) 

    def generate_context_state(self, device):
        """
        Initialize hidden state to random
        """
        return torch.randn(self.hidden_shape, device=device) 


class LSTMLanguageModel(LSTMLanguageModelBase):
    """ simple LSTM neural network language model """     
    def __init__(self, vocab, batch_size=10, hidden_size=100, embed_size=300, dropout_rate=0.5, \
            num_layers=1, embed_path=None, device=None, **kwargs):
        super(LSTMLanguageModel, self).__init__(batch_size, hidden_size, num_layers)
        
        vocab_size = len(vocab)
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        # load glove embeddings and initialize 
        if embed_path is not None:
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

    def forward(self, x, init_state):
        """
        @param x: (batch_size, sequence_length)
        @param init_state: Tuple(Tensor, Tensor)
            each Tensor (batch_size, hidden_size)
        """
        # (batch_size, sequence_length, embed_size)
        embedded = self.embeddings(x)
        embedded = self.drop(embedded)
        # (batch_size, sequence_length, embed_size)
        lstm_output, ret_state = self.lstm(embedded, init_state)
        # (batch_size * sequence_length, hidden_size)
        reshaped = lstm_output.reshape(-1, lstm_output.size(2))
        reshaped = self.drop(reshaped)
        decoded = self.linear(reshaped)
        # (batch_size * sequence_length, vocab_size)
        logits = F.log_softmax(decoded, dim=1)
                
        return logits, self.detach_hidden(ret_state)


class RNNLanguageModel(RNNLanguageModelBase):
    """ simple RNN neural network language model """     
    def __init__(self, vocab, batch_size=10, hidden_size=100, embed_size=300, dropout_rate=0.5, \
            num_layers=1, embed_path=None, device=None, **kwargs):
        super(RNNLanguageModel, self).__init__(batch_size, hidden_size, num_layers)
        
        vocab_size = len(vocab)
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        # load glove embeddings and initialize 
        if embed_path is not None:
            glove_embeddings = get_glove(embed_path, vocab)
            glove_tensor = torch.tensor(glove_embeddings, dtype=torch.float32, device=device)
            self.embeddings.weight = nn.Parameter(glove_tensor)
            self.embeddings.weight.requires_grad = False

        self.rnn = nn.RNN(
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
        embedded = self.drop(embedded)
        # (batch_size, sequence_length, embed_size)
        rnn_output, ret_state = self.rnn(embedded, init_state)
        # (batch_size * sequence_length, hidden_size)
        reshaped = rnn_output.reshape(-1, lstm_output.size(2))
        reshaped = self.drop(reshaped)
        decoded = self.linear(reshaped)
        # (batch_size * sequence_length, vocab_size)
        logits = F.log_softmax(decoded, dim=1)
                
        return logits, self.detach_hidden(ret_state)


class AssignmentLSTMLanguageModel(LSTMLanguageModelBase):
    """ second order LSTM language model with explicit assignment """     
    def __init__(self, vocab, batch_size=10, hidden_size=100, embed_size=12, dropout_rate=0.5, \
            device=None, assignments=None, num_cells=2, **kwargs):
        super(AssignmentLanguageModel, self).__init__(batch_size, hidden_size)
        
        vocab_size = len(vocab)
        self.embeddings = nn.Embedding(vocab_size, embed_size)

        # Here we introduce the assignmentLSTM
        self.lstm = paren_mLSTM(
            embed_size=embed_size, 
            hidden_size=hidden_size, 
            assignments=assignments, 
            num_cells=num_cells, 
            device=device)

        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, x, init_state):
        """
        @param x: (batch_size, sequence_length)
        """
        embedded = self.embeddings(x)
        embedded = self.drop(embedded)
        lstm_output, ret_state = self.lstm(x, embedded, init_state)
        reshaped = lstm_output.view(-1, lstm_output.size(2))
        reshaped = self.drop(reshaped)
        decoded = self.linear(reshaped)
        # (batch_size * sequence_length, vocab_size)
        logits = F.log_softmax(decoded, dim=1)
                
        return logits, self.detach_hidden(ret_state)


class AttentionLSTMLanguageModel(LSTMLanguageModelBase):
    """ second order attention language model """     
    def __init__(self, vocab, second_order_size=2, batch_size=10, hidden_size=100, embed_size=12, \
            device=None, temp_decay=0.9, temp_decay_interval=None, dropout_rate=0.5, **kwargs):
        """
        @param temp_decay: Everything temperature decays, the temperature is multiplied by this value
        @param temp_decay_interval: The temperature will decay after forward() is called temp_decay_interval times. Should be set to
        len(dataloader) when training. When testing, should be set to None so temperature doesn't decay.
        """
        super(AttentionLSTMLanguageModel, self).__init__(batch_size, hidden_size)
        
        vocab_size = len(vocab)
        self.embeddings = nn.Embedding(vocab_size, embed_size)

        self.lstm = AttentionSecondOrderLSTM(
            second_order_size=second_order_size, 
            input_size=embed_size, 
            hidden_size=hidden_size)

        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.drop = nn.Dropout(p=dropout_rate)
        
        self.counter = 0
        self.train_temperature = 1
        self.test_temperature = 1e-5
        self.temp_decay = temp_decay
        self.temp_decay_interval = temp_decay_interval
        
    def forward(self, x, init_state):
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
        embedded = self.drop(embedded)
        lstm_output, ret_state = self.lstm(embedded, temperature, init_state)
        reshaped = lstm_output.view(-1, lstm_output.size(2))
        reshaped = self.drop(reshaped)
        decoded = self.linear(reshaped)
        # (batch_size * sequence_length, vocab_size)
        logits = F.log_softmax(decoded, dim=1)
                
        return logits, self.detach_hidden(ret_state)



class AttentionRNNLanguageModel(RNNLanguageModelBase):
    """ second order attention language model """     
    def __init__(self, vocab, second_order_size=2, batch_size=10, hidden_size=100, embed_size=12, \
            device=None, temp_decay=0.9, temp_decay_interval=None, dropout_rate=0.5, **kwargs):
        """
        @param temp_decay: Everything temperature decays, the temperature is multiplied by this value
        @param temp_decay_interval: The temperature will decay after forward() is called temp_decay_interval times. Should be set to
        len(dataloader) when training. When testing, should be set to None so temperature doesn't decay.
        """
        super(AttentionRNNLanguageModel, self).__init__(batch_size, hidden_size)
        
        vocab_size = len(vocab)
        self.embeddings = nn.Embedding(vocab_size, embed_size)

        self.rnn = AttentionSecondOrderRNN(
            second_order_size=second_order_size, 
            input_size=embed_size, 
            hidden_size=hidden_size)

        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.drop = nn.Dropout(p=dropout_rate)
        
        self.counter = 0
        self.train_temperature = 1
        self.test_temperature = 1e-5
        self.temp_decay = temp_decay
        self.temp_decay_interval = temp_decay_interval
        
    def forward(self, x, init_state):
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
        embedded = self.drop(embedded)
        rnn_output, ret_state = self.rnn(embedded, temperature, init_state)
        reshaped = rnn_output.view(-1, lstm_output.size(2))
        reshaped = self.drop(reshaped)
        decoded = self.linear(reshaped)
        # (batch_size * sequence_length, vocab_size)
        logits = F.log_softmax(decoded, dim=1)
                
        return logits, self.detach_hidden(ret_state)