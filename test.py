from eval import *
from dataset import *
import sys
import numpy as np
import torch

def test_distances(vocab):
    print ("-"*80)
    print("Running Sanity Check: Distances Between Parens")
    
    # Test 1
    test = ['(a','a)', '(b', 'b)', '(a', 'a)', '(b', 'b)', '(a', 'a)']
    test_idx = vocab.words2indices(test)
    exp_dist = [1, None, 1, None, 1, None, 1, None, 1]
    dist = get_distances(y=test_idx[1:], init=test_idx[0])
    assert(exp_dist == dist)

    # Test 2
    test = ['(b','(a','(b', '(a', '(b', 'b)', 'a)', 'b)', 'a)', 'b)']
    test_idx = vocab.words2indices(test)
    exp_dist = [None, None, None, None, 1, 3, 5, 7, 9]
    dist = get_distances(y=test_idx[1:], init=test_idx[0])
    assert(exp_dist == dist)

    # Test 3
    test = ['(a','(a', 'a)', '(b', '(a', 'a)', 'b)', '(b', 'b)', 'a)']
    test_idx = vocab.words2indices(test)
    exp_dist = [None, 1, None, None, 1, 3, None, 1, 9]
    dist = get_distances(y=test_idx[1:], init=test_idx[0])
    assert(exp_dist == dist)

    print("All Sanity Checks Passed!")
    print ("-"*80)

def test_ldpa(vocab):
    print ("-"*80)
    print("Running Sanity Check: LDPA Calculation")
    
    batch_size = 3
    sent_len = 9
    vocab_size = len(vocab)
    test1 = vocab.words2indices(['(a','a)', '(b', 'b)', '(a', 'a)', '(b', 'b)', '(a', 'a)'])
    test2 = vocab.words2indices(['(b','(a','(b', '(a', '(b', 'b)', 'a)', 'b)', 'a)', 'b)'])
    test3 = vocab.words2indices(['(a','(a', 'a)', '(b', '(a', 'a)', 'b)', '(b', 'b)', 'a)'])
    y = test1[1:] + test2[1:] + test3[1:]
    init = [test1[0], test2[0], test3[0]]

    y_pred = torch.ones((batch_size * sent_len, vocab_size)) / 4
    y_pred[:, 0:2] = 0.0

    # Test 1: 0%
    ldpa = LDPA(y=torch.tensor(y), y_pred=y_pred, init=init, batch_size=batch_size, thresh=0.8)
    assert(all([ldpa[i][1] == 0 for i in range(len(ldpa))]))

    # Test 2: 100%
    ldpa = LDPA(y=torch.tensor(y), y_pred=y_pred, init=init, batch_size=batch_size, thresh=0.5)
    assert(all([ldpa[i][1] / ldpa[i][0] == 1 for i in range(len(ldpa)) if ldpa[i][0]]))

    # Test 3: mix 
    y_pred[:, 5] = 0.1
    ldpa = LDPA(y=torch.tensor(y), y_pred=y_pred, init=init, batch_size=batch_size, thresh=0.5)
    ratios = [ldpa[i][1] / ldpa[i][0] for i in range(len(ldpa)) if ldpa[i][0]]
    assert(ratios == [5/9, 1/2, 0, 1, 1/2])

    print("All Sanity Checks Passed!")
    print ("-"*80)

if __name__ == "__main__":
    vocab = Vocab(file_path="data/mbounded-dyck-k/m4/train.formal.json")
    if sys.argv[1] == 'dist':
        test_distances(vocab)
    if sys.argv[1] == 'ldpa':
        test_ldpa(vocab)
