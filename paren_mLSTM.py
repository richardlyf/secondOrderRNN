import torch
import torch.nn as nn
import numpy as np

class paren_mLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, assignments, device, bias=True, num_cells=2):
        """
        - Quite similar to the initilization of a regular LSTMCell, but with two additional steps. 
        - First, multiple internal LSTMCells are initialized, each representing a different weight matrix. 
        - Then, a method of assigning inputs to each LSTM will be passed in if "assignment" is set to "explicit".
        ---------------------------------------------------------------------------------------- 
        @param embed_size (int): Size of word embedding
        @param hidden_Size (int): Size of hidden state
        @param vocab: Vocab object containing all the vocab to index mappings
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
        self.lstm_list = nn.ModuleList([nn.LSTMCell(embed_size, hidden_size, bias=bias) for i in range(num_cells)])

        # Preserve arguments
        self.vocab = vocab
        self.assignments = assignments
        self.hidden_size = hidden_size 
        self.num_cells = num_cells

    def forward(self, input, input_embed, dec_states=None):
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
        @param input Tensor(batch_size, seq_len): input of size (seq_len, batch_size)
        @param input_embed Tensor(seq_len, batch_size, embed_size): Embedding of the entire input. Input to LSTMCells
        @param dec_states Tensor(batch_size, 2, hidden_size): hidden state and cell state used to initalize the LSTMCells

        @return combined_outputs Tensor(seq_len, batch_size, hidden_size): Output tensor of size (batch_size, 2 * hidden_size)
        """
        input = input.cpu().detach().numpy()
        batch_size, seq_len = input.shape
        
        # dec_states dim=0 will hold the two Tensors representing the 
        # sentences current hidden and cell states at each time step
        if dec_states == None:
            dec_states = torch.zeros((batch_size, 2, self.hidden_size), device=input_embed.device)

        # outputs to be returned at the end of the function and used to make prediction
        combined_outputs = torch.empty((seq_len, batch_size, self.hidden_size), device=input_embed.device)

        for seq_idx in range(seq_len):
            # All embeddings at the current time-step
            # (batch_size,)
            idx_input = input[:, seq_idx]
            # (batch_size, embed_size)
            idx_input_embeddings = input_embed[seq_idx]

            # For each LSTMCell, compute their sub-batches and sub-hidden states
            for cell_idx in range(self.num_cells):
                sub_batch_indices = np.where(np.isin(idx_input, self.assignments[cell_idx]))
                # (sub_batch_size, embed_size)
                sub_batch_embeddings = idx_input_embeddings[sub_batch_indices]
                # Skip this cell if nothing is assigned to it. Sub-batch is empty
                if (sub_batch_embeddings.size(0) == 0):
                    continue
                # (sub_batch_size, 2, hidden_size)
                sub_dec_states = dec_states[sub_batch_indices]
                # (sub_batch_size, 1, hidden_size)
                sub_hidden, sub_cell = torch.split(sub_dec_states, 1, dim=1)
                # (sub_batch_size, hidden_size)
                sub_hidden = torch.squeeze(sub_hidden, dim=1)
                sub_cell = torch.squeeze(sub_cell, dim=1)

                sub_hidden, sub_cell = self.lstm_list[cell_idx](sub_batch_embeddings, (sub_hidden, sub_cell))
                # Update corresponding hidden states
                dec_states[sub_batch_indices] = torch.stack([sub_hidden, sub_cell], dim=1)
                dec_states = dec_states.detach()
                combined_outputs[seq_idx, sub_batch_indices] = sub_hidden

        return combined_outputs, (dec_states[:, 0], dec_states[:, 1])


# TEST

class test_LSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, device, bias=True):
        super(test_LSTM, self).__init__()
        # Checking proper set up
        if embed_size < 1 or hidden_size < 1:
            raise ValueError("'num_cells', 'embd_dim', and 'hidden_layers' must be >= 1")

        # Initialize a single LSTMCell
        self.lstm_cell = nn.ModuleList([nn.LSTMCell(embed_size, hidden_size, bias=bias)])

        # Preserve arguments
        self.vocab = vocab
        self.hidden_size = hidden_size 

    def forward(self, input, input_embed, dec_states=None):
        input = input.cpu().detach().numpy()
        batch_size, seq_len = input.shape
        
        # dec_states dim=0 will hold the two Tensors representing the 
        # sentences current hidden and cell states at each time step
        if dec_states == None:
            dec_states = torch.zeros((batch_size, 2, self.hidden_size), device=input_embed.device)

        # outputs to be returned at the end of the function and used to make prediction
        combined_outputs = torch.empty((seq_len, batch_size, self.hidden_size), device=input_embed.device)

        for seq_idx in range(seq_len):
            # initialize hidden and cell
            hidden, cell = torch.split(dec_states, 1, dim=1)
            # (sub_batch_size, hidden_size)
            hidden = torch.squeeze(hidden, dim=1)
            cell = torch.squeeze(cell, dim=1)

            # All embeddings at the current time-step
            # (batch_size, embed_size)
            idx_input_embeddings = input_embed[seq_idx]
            # update hidden and cell
            hidden, cell = self.lstm_cell[0](idx_input_embeddings, (hidden, cell))
            dec_states = torch.stack([hidden, cell], dim=1)
            # upload combined outputs
            combined_outputs[seq_idx, :] = hidden

        return combined_outputs, (hidden, cell)