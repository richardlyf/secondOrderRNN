import torch
import torch.nn as nn

class paren_mLSTM(nn.Module):

    # TODO: look up how to use the Torch Text object
    def __init__(self, emb_dim, hidden_size, vocab, device, bias=True, num_cells=2, input_assignment=None):
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
        super(paren_mLSTM, self).__init__()
        # Checking proper set up
        if num_cells < 1 or emb_dim < 1 or hidden_size < 1:
            raise ValueError("'num_cells', 'embd_dim', and 'hidden_layers' must be >= 1")

        if input_assignment != None:
            max_idx = max(list(input_assignment.values()))
            if max_idx > num_cells - 1:
                raise ValueError("invalid value in 'input_assignment' larger than number of LSTMCells")

        # Initialize num_cells numbers of LSTMCells
        self.lstm_list = []
        for i in range(num_cells):
            self.lstm_list.append(nn.LSTMCell(emb_dim, hidden_size, bias=bias))

        # Initialize parentheses embeddings
        self.embeddings = nn.Embedding(len(vocab), emb_dim) #TODO: add pad_idx?

        # Preserve arguments 
        self.assignments = input_assignment 
        self.hidden_size = hidden_size 
        self.num_cells = num_cells

    def forward(self, input, dec_states=None, train=True):
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
        @param input (Tensor of ints): input of size (batch_size, sen_len)
        @param dec_states (List[(Tensor, Tensor)]) input_assignment
        @param train (boolean): allows for dropout TODO

        @ return combined_outputs(Tensor): Output tensor of size (batch_size, 2 * hidden_size)
        """
        batch_size, sen_len = input.shape
        input_embed = self.embeddings(input)
        
        # dec_states dim=0 will hold the two Tensors representing the 
        # sentences current hidden and cell states at each time step
        if dec_states == None:
            dec_states = torch.zeros((batch_size, 2, self.hidden_size))

        # Calculate which cell each sentence in the batch should be applied to for the entire sentence length
        mini_batch_indices = self.construct_mini_batch_indices(input)

        for i in range(sen_len):
            # All embeddings at the current time-step
            time_step_embeddings = torch.index_select(input_embed, 1, torch.tensor([i]))

            # need to break into mini batches
            mini_batches = [None] * self.num_cells #may not need to declare these outside of the loop
            mini_dec_states = [None] * self.num_cells
            for j in range(len(mini_batches)):
                curr_indices = mini_batch_indices[i][j]
                mini_batches[j] = torch.squeeze(torch.index_select(time_step_embeddings, 0, curr_indices), dim=1)
                
                # need to group past hidden and cell states with the mini_batch
                mini_dec_states[j] = torch.index_select(dec_states, 0, curr_indices) 
                # TODO: put these dec_states into tuple form
                mini_hidden = torch.squeeze(torch.index_select(mini_dec_states[j], 1, torch.tensor([0])), dim=1)
                mini_cell  = torch.squeeze(torch.index_select(mini_dec_states[j], 1, torch.tensor([1])), dim=1)
                # need to run mLSTM cell on each mini_batch
                hidden_outputs, cell_outputs  = self.lstm_list[j](mini_batches[j], (mini_hidden, mini_cell))
                
                for k in range(curr_indices.shape[0]): #this might be 1
                    curr_hidden = torch.index_select(hidden_outputs, 0, torch.tensor([k])) 
                    curr_cell = torch.index_select(cell_outputs, 0, torch.tensor([k])) 
                    dec_states[curr_indices[k]] =  torch.cat((curr_hidden, curr_cell), 0)

        return dec_states.reshape((batch_size, 2 * self.hidden_size))
        

    def construct_mini_batch_indices(self, input):
        """
        Used to divide batches into minibatches along the length of an entire input. 
        ---------------------------
        @param input (Tensor of ints): Provides the information to be used with 
            self.assignments that maps an input string at a given time step
            to the LSTMCell to which that input will be applied. (batch_size, sen_len)

        @return indices (List[List[Tensor]] The look up tensor to be used as the 
                third argument to torch.selex_index for each time step and LSTMCell. 
        """
        batch, sen_len = input.shape
        indices = [] 

        for i in range(sen_len):
            indices.append([])
            tokens = torch.index_select(input, 1, torch.tensor([i])) 
            for j in range(self.num_cells):
                indices[i].append(torch.tensor([i for i in range(len(tokens)) if self.assignments[int(tokens[i])] == j]).long())

        return indices










