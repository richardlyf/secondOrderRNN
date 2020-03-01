from eval import *
from dataset import *
import sys
import numpy as np

def test_distances():
    print ("-"*80)
    print("Running Sanity Check: Distances Between Parens")
    vocab = Vocab(file_path="data/mbounded-dyck-k/m4/train.formal.json")

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

if __name__ == "__main__":
    if sys.argv[1] == 'dist':
        test_distances()
