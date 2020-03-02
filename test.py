from eval import *
from dataset import *
import sys
import numpy as np
import torch
import time

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
    print("Running Sanity Check: get_LDPA_counts Calculation")
    
    batch_size = 3
    sent_len = 9
    vocab_size = len(vocab)
    test1 = vocab.words2indices(['(a','a)', '(b', 'b)', '(a', 'a)', '(b', 'b)', '(a', 'a)'])
    test2 = vocab.words2indices(['(b','(a','(b', '(a', '(b', 'b)', 'a)', 'b)', 'a)', 'b)'])
    test3 = vocab.words2indices(['(a','(a', 'a)', '(b', '(a', 'a)', 'b)', '(b', 'b)', 'a)'])
    y = torch.tensor(test1[1:] + test2[1:] + test3[1:])
    init = torch.tensor([test1[0], test2[0], test3[0]])

    y_pred = torch.ones((batch_size * sent_len, vocab_size)) / 4
    y_pred[:, 0:2] = 0.0

    start = time.time()
    # Test 1: 0%
    ldpa = get_LDPA_counts(y=y, y_pred=y_pred, init=init, batch_size=batch_size, max_dist=sent_len, thresh=0.8)
    assert(all([ldpa[i][1] == 0 for i in range(len(ldpa))]))

    # Test 2: 100%
    ldpa = get_LDPA_counts(y=y, y_pred=y_pred, init=init, batch_size=batch_size, max_dist=sent_len, thresh=0.5)
    assert(all([ldpa[i][1] / ldpa[i][0] == 1 for i in range(len(ldpa)) if ldpa[i][0]]))

    # Test 3: mix 
    y_pred[:, 5] = 0.1
    ldpa = get_LDPA_counts(y=y, y_pred=y_pred, init=init, batch_size=batch_size, max_dist=sent_len, thresh=0.5)
    ratios = [ldpa[i][1] / ldpa[i][0] for i in range(len(ldpa)) if ldpa[i][0]]
    assert(ratios == [5/9, 1/2, 0, 1, 1/2]), "Expected {} \n Got {}".format([5/9, 1/2, 0, 1, 1/2], ratios)

    # Test 4: robust random
    batch_size = 2
    sent_len = 3
    test1 = vocab.words2indices(['(a','a)', '(b', 'b)'])
    test2 = vocab.words2indices(['(a','(b', 'b)', 'a)'])
    y = torch.tensor(test1[1:] + test2[1:])
    init = torch.tensor([test1[0], test2[0]])
    y_pred = torch.ones((batch_size * sent_len, vocab_size)) * 0.001
    y_pred[0, vocab.word2id['a)']] = 1 # at dist 1, 1 above thresh
    y_pred[2, vocab.word2id['a)']] = 1 # at dist 1, 0 above thresh
    y_pred[4, vocab.word2id['b)']] = 1 # at dist 1, 1 above thresh
    y_pred[5, vocab.word2id['a)']] = 1 # at dist 3, 1 above thresh
    ldpa = get_LDPA_counts(y=y, y_pred=np.log(y_pred), init=init, batch_size=batch_size, max_dist=sent_len, thresh=0.8)
    assert(ldpa[1][0] == 3), "Expect 3 close at dist 1, got {}".format(ldpa[1][0]) # 3 closed at dist 1
    assert(ldpa[1][1] == 2), "Expect 2 above thresh at dist 1, got {}".format(ldpa[1][1]) # 2 of which above threshold
    assert(ldpa[3][0] == 1), "Expect 1 close at dist 3, got {}".format(ldpa[3][0]) # 1 closed at dist 3
    assert(ldpa[3][1] == 1), "Expect 1 above thresh at dist 3, got {}".format(ldpa[3][1]) # 1 of which above threshold

    # Test 5: validate wcpa
    val_dataset = ParensDataset("./data/mbounded-dyck-k/m4/dev.formal.txt")
    x, y = val_dataset[0]
    max_sents_len = len(x)
    total_ldpa_counts = np.zeros((max_sents_len + 1, 2), dtype=np.int64)
    
    # fabricate y_pred for 1st sample of batch 1
    y_pred = np.ones((len(y), vocab_size)) * 0.001
    y_pred[0, 4] = 1 # dist 1 above thresh
    y_pred[4, 5] = 1 # dist 1 above thresh
    y_pred[13, 5] = 1 # dist 1 not above thresh
    first_chars = torch.tensor([x[0]])
    y = torch.tensor(y)
    y_pred = torch.tensor(np.log(y_pred))
    ldpa_counts = get_LDPA_counts(y=y, y_pred=y_pred, init=first_chars, batch_size=1, max_dist=max_sents_len)
    total_ldpa_counts += ldpa_counts

    # ground truth counts
    gt_ldpa_counts = np.zeros((max_sents_len + 1, 2), dtype=np.int64)
    gt_ldpa_counts[1, 0] = 32
    gt_ldpa_counts[3, 0] = 11
    gt_ldpa_counts[5, 0] = 3
    gt_ldpa_counts[9, 0] = 2
    gt_ldpa_counts[11, 0] = 2
    gt_ldpa_counts[13, 0] = 1
    gt_ldpa_counts[23, 0] = 1
    gt_ldpa_counts[25, 0] = 1
    gt_ldpa_counts[31, 0] = 1
    gt_ldpa_counts[1, 1] = 2
    assert(np.array_equal(gt_ldpa_counts[:, 0], total_ldpa_counts[:, 0])), \
        "first field not equal! \n gt_ldpa_counts {} \n total_ldpa_counts {} \n".format(gt_ldpa_counts[:, 0], total_ldpa_counts[:, 0])
    assert(np.array_equal(gt_ldpa_counts[:, 1], total_ldpa_counts[:, 1])), \
        "second field not equal! \n gt_ldpa_counts {} \n total_ldpa_counts {} \n".format(gt_ldpa_counts[:, 1], total_ldpa_counts[:, 1])

    # fabricate y_pred for 2nd sample of batch 1
    x, y = val_dataset[1]
    y_pred = np.ones((len(y), vocab_size)) * 0.001
    y_pred[2, 4] = 1 # dist 1 above thresh
    y_pred[4, 4] = 1 # dist 1 above thresh
    y_pred[7, 5] = 1 # dist 1 above thresh
    y_pred[8, 4] = 1 # dist 3 above thresh
    y_pred[44, 4] = 1 # dist 45 above thresh
    first_chars = torch.tensor([x[0]])
    y = torch.tensor(y)
    y_pred = torch.tensor(np.log(y_pred))
    ldpa_counts = get_LDPA_counts(y=y, y_pred=y_pred, init=first_chars, batch_size=1, max_dist=max_sents_len)
    total_ldpa_counts += ldpa_counts

    # ground truth counts
    gt_ldpa_counts[1, 0] += 28
    gt_ldpa_counts[3, 0] += 6
    gt_ldpa_counts[5, 0] += 5
    gt_ldpa_counts[7, 0] += 1
    gt_ldpa_counts[9, 0] += 2
    gt_ldpa_counts[23, 0] += 1
    gt_ldpa_counts[33, 0] += 1
    gt_ldpa_counts[45, 0] += 1

    gt_ldpa_counts[1, 1] += 3
    gt_ldpa_counts[3, 1] += 1
    gt_ldpa_counts[45, 1] += 1
    assert(np.array_equal(gt_ldpa_counts[:, 0], total_ldpa_counts[:, 0])), \
        "first field not equal! \n gt_ldpa_counts {} \n total_ldpa_counts {} \n".format(gt_ldpa_counts[:, 0], total_ldpa_counts[:, 0])
    assert(np.array_equal(gt_ldpa_counts[:, 1], total_ldpa_counts[:, 1])), \
        "second field not equal! \n gt_ldpa_counts {} \n total_ldpa_counts {} \n".format(gt_ldpa_counts[:, 1], total_ldpa_counts[:, 1])

    valid_dist = np.where(total_ldpa_counts[:, 0] > 0)
    ldpa = total_ldpa_counts[valid_dist, 1] / total_ldpa_counts[valid_dist, 0]
    gt_ldpa = np.array([1/12, 1/17, 1])
    ldpa = ldpa[np.nonzero(ldpa)]
    assert(np.array_equal(ldpa, gt_ldpa)), "gt_ldpa {} \n ldpa {}".format(gt_ldpa, ldpa)

    # time test
    for i in range(2000):
        ldpa_counts = get_LDPA_counts(y=y, y_pred=y_pred, init=first_chars, batch_size=1, max_dist=max_sents_len)

    print("All Sanity Checks Passed! Took time: ", time.time() - start)
    print ("-"*80)

if __name__ == "__main__":
    vocab = Vocab(file_path="data/mbounded-dyck-k/m4/train.formal.json")
    if sys.argv[1] == 'dist':
        test_distances(vocab)
    if sys.argv[1] == 'ldpa':
        test_ldpa(vocab)