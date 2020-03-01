import torch
import torch.nn as nn

class paren_mLSTM(nn.Module):

    # TODO: look up how to use the Torch Text object
    def __init__(emb_dim, hidden_size, text, device, bias=True, num_cells=2, input_assignment=None):
        """
        -Quite similar to the initilization of a regular LSTMCell, but with two additional steps. 
        - First, multiple internal LSTMCells are initialized, each representing a different weight matrix. 
        - Then, a method of assigning inputs to each LSTM will be passed in if "assignment" is set to "explicit".
        ---------------------------------------------------------------------------------------- 
        @param emb_dim (int): Embedding size (dimensionality) of word, needs to match torchtext embedding dim
        @param hidden_Size (int): Hidden size (dimensionality) 
        @param texts (torchtext): torchtext object containing the language  
               should contain vocab object
        @param device (device): device argument where to save tensors
        @param bias (bool): Include bias or not in forward step
        @param num_cells (int): Number of LSTMCells. 
        @param assignment (string): Either 'explicit' or 'learned', denotes 
               decision strategy to apply inputs to specific LSTMCells.
        @param input_assignment (dictionary{string : int}): maps input strings to an indice
               that represents which LSTMCell gets applied to the input                                                           

        """
        # Checking proper set up
        if num_cells < 1 or emb_dim < 1 or hidden_size < 1:
            raise ValueError("'num_cells', 'embd_dim', and 'hidden_layers' must be >= 1")

        if input_assignments != None:
            max_idx = max(list(input_assignments.values()))
            if max_idx > num_cells - 1:
                raise ValueError("invalid value in 'input_assignments' larger than number of LSTMCells"

        # Initialize num_cells numbers of LSTMCells
        self.lstm = []
        for i in range(num_cells):
            self.lstm.append(nn.LSTMCell(emb_dim, hidden_size, bias=bias))

        # Initialize parentheses embeddings
        vocab = text.vocab
        self.text = text
        self.embeddings = nn.Embedding(len(vocab), emb_dim)
        self.embeddings.weight.data.copy_(vocab.vectors)

        # Set assignments 
        self.assignments = input_assignment 
        
        
    def forward(input, hidden_state=None, train=True):
        """
        @param input (tensor of strings): input of size (batch_size, sen_len)
        @param hidden_state (tuple(tensor, tensor)): 
        """
        pass

