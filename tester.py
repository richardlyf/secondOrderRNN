from paren_mLSTM import paren_mLSTM
import torch
import torch.nn as nn
import random

# Set up mLSTM
embed_dim = 5
hidden_size = 3
vocab = [i for i in range(6)]
input_assignments = dict((i, i % 2) for i in vocab)

mLSTM = paren_mLSTM(embed_dim, hidden_size, vocab, "device", input_assignment=input_assignments)


batch_size = 1
sen_len = 4 

inputs = [random.randint(0,5) for i in range(batch_size * sen_len)]
inputs = torch.tensor(inputs).reshape((batch_size, sen_len))

print(inputs)
outputs = mLSTM.forward(inputs)
print('outputs', outputs)
print('outputs shape', outputs.shape)
