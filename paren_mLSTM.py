import torch
import torch.nn as nn
import numpy as np

class paren_mLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, assignments, device, bias=True, num_cells=2):
        """
        - Quite similar to the initilization of a regular LSTMCell, but with two additional steps. 
        - First, multiple internal LSTMCells are initialized, each representing a different weight matrix. 
        - Then, a method of assigning inputs to each LSTM will be passed in as "assignment".
        ---------------------------------------------------------------------------------------- 
        @param embed_size (int): Size of word embedding
        @param hidden_Size (int): Size of hidden state
        @param assignments (dict[int -> List(int)]): A map from LSTMCell index to a list of vocab word indicies
        that should be assigned to that LSTMCell
        @param device: gpu or cpu
        @param bias (bool): Include bias or not in forward step
        @param num_cells (int): Number of LSTMCells. 
        """
        super(paren_mLSTM, self).__init__()
        # Checking proper set up
        if num_cells < 1 or embed_size < 1 or hidden_size < 1:
            raise ValueError("'num_cells', 'embd_dim', and 'hidden_layers' must be >= 1")

        if assignments != None:
            max_idx = max(list(assignments.keys()))
            if max_idx > num_cells - 1:
                raise ValueError("invalid value in 'assignments' larger than number of LSTMCells")

        # Initialize num_cells numbers of LSTMCells
        self.lstm_cells = nn.ModuleList([nn.LSTMCell(embed_size, hidden_size, bias=bias) for i in range(num_cells)])
        self.assignments = assignments
        self.hidden_size = hidden_size 
        self.num_cells = num_cells

    def forward(self, input, input_embed, lstm_states=None):
        """
        Fun stuff going on here. The difficulty lies in not being able to apply just one of our LSTM cells to 
        an entire sample, but still wanting to batch. So, we construct a few different pieces
        to keep track of the hidden and cell state of each sample in the batch at every time step.
        Each time step follows this process:
        (1) Break the current time step into different mini batches based on the LSTMCell they 
            will be applied to. 
        (2) Assemble the previous states for the mini_batch to feed into the LSTM.
        (3) Run the specific LSTMCell on the mini_batch
        (4) Place the outputs into their appropriate location in lstm_states
        -------------------------------
        @param input Tensor(batch_size, seq_len): input of size (batch_size, seq_len)
        @param input_embed Tensor(batch_size, seq_len, embed_size): Embedding of the entire input. Input to LSTMCells
        @param lstm_states Tensor(batch_size, 2, hidden_size): hidden state and cell state used to initalize the LSTMCells
        @return combined_outputs Tensor(batch_size, seq_len, hidden_size): Output of all hidden states for each time sequence
        @return last_dec_state: Tuple of the last hidden state and cell state in the time sequence
        """
        input = input.cpu().detach().numpy()
        batch_size, seq_len = input.shape
        
        if lstm_states == None:
            hidden = torch.zeros((batch_size, self.hidden_size), device=input_embed.device)
            cell = torch.zeros((batch_size, self.hidden_size), device=input_embed.device)
            lstm_states = (hidden, cell)

        # outputs to be returned at the end of the function and used to make prediction
        combined_outputs = torch.empty((batch_size, seq_len, self.hidden_size), device=input_embed.device)

        for seq_idx in range(seq_len):
            # initialize hidden and cell
            hidden, cell = lstm_states
            new_hidden = torch.zeros((batch_size, self.hidden_size), device=input_embed.device)
            new_cell = torch.zeros((batch_size, self.hidden_size), device=input_embed.device)

            # (batch_size, embed_size) Embeddings at the current time-step
            input_embed_t = input_embed[:, seq_idx]
            # (batch_size,) Input character indices at the current time-step
            input_t = input[:, seq_idx]

            # For each LSTMCell, compute their sub-batches and sub-hidden states
            for cell_idx in range(self.num_cells):
                sub_batch_indices = np.where(np.isin(input_t, self.assignments[cell_idx]))
                # (sub_batch_size, embed_size)
                sub_batch_embeddings = input_embed_t[sub_batch_indices]
                # Skip this cell if nothing is assigned to it. Sub-batch is empty
                if (sub_batch_embeddings.size(0) == 0):
                    continue
                # (sub_batch_size, hidden_size)
                sub_hidden = hidden[sub_batch_indices]
                sub_cell = cell[sub_batch_indices]
                # Update hidden and cell
                sub_hidden, sub_cell = self.lstm_cells[cell_idx](sub_batch_embeddings, (sub_hidden, sub_cell))
                new_hidden[sub_batch_indices] = sub_hidden
                new_cell[sub_batch_indices] = sub_cell
            # Update lstm states and store new hidden state to combined_outputs
            lstm_states = (new_hidden, new_cell)
            combined_outputs[:, seq_idx] = new_hidden

        return combined_outputs, (new_hidden, new_cell)


# TEST COMPLETED
# THE FOLLOWING MODEL SERVED ITS PURPOSE AND SHOULD NO LONGER BE NEEDED

class test_LSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, assignments, device, bias=True, num_cells=2):
        super(test_LSTM, self).__init__()
        # Checking proper set up
        if embed_size < 1 or hidden_size < 1:
            raise ValueError("'num_cells', 'embd_dim', and 'hidden_layers' must be >= 1")

        # Initialize a single LSTMCell
        self.lstm_cell = nn.ModuleList([nn.LSTMCell(embed_size, hidden_size, bias=bias) for i in range(num_cells)])

        # Preserve arguments
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.assignments = assignments
        self.num_cells = num_cells

    def forward(self, input, input_embed, lstm_states=None):
        input = input.cpu().detach().numpy()
        batch_size, seq_len = input.shape
        
        # lstm_states dim=0 will hold the two Tensors representing the 
        # sentences current hidden and cell states at each time step
        if lstm_states == None:
            hidden = torch.zeros((batch_size, self.hidden_size), device=input_embed.device)
            cell = torch.zeros((batch_size, self.hidden_size), device=input_embed.device)
            lstm_states = (hidden, cell)

        # outputs to be returned at the end of the function and used to make prediction
        combined_outputs = torch.empty((batch_size, seq_len, self.hidden_size), device=input_embed.device)

        for seq_idx in range(seq_len):
            # initialize hidden and cell
            hidden, cell = lstm_states
            new_hidden = torch.zeros((batch_size, self.hidden_size), device=input_embed.device)
            new_cell = torch.zeros((batch_size, self.hidden_size), device=input_embed.device)

            # All embeddings at the current time-step
            # (batch_size, embed_size)
            input_embed_t = input_embed[:, seq_idx]
            # (batch_size,)
            input_t = input[:, seq_idx]

            # For each LSTMCell, compute their sub-batches and sub-hidden states
            for cell_idx in range(self.num_cells):
                sub_batch_indices = np.where(np.isin(input_t, self.assignments[cell_idx]))
                # (sub_batch_size, embed_size)
                sub_batch_embeddings = input_embed_t[sub_batch_indices]
                # Skip this cell if nothing is assigned to it. Sub-batch is empty
                if (sub_batch_embeddings.size(0) == 0):
                    continue
                sub_hidden = hidden[sub_batch_indices]
                sub_cell = cell[sub_batch_indices]

                # update hidden and cell
                sub_hidden, sub_cell = self.lstm_cell[cell_idx](sub_batch_embeddings, (sub_hidden, sub_cell))
                new_hidden[sub_batch_indices] = sub_hidden
                new_cell[sub_batch_indices] = sub_cell
            lstm_states = (new_hidden, new_cell)
            # upload combined outputs
            combined_outputs[:, seq_idx] = new_hidden

        return combined_outputs, (new_hidden, new_cell)