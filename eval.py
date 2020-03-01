import itertools
import numpy as np
import torch

def get_distances(y, init, close_idx=[4, 5], open_idx=[2, 3], pad_idx=0):
    """
    Calculate distance between pairs of open and close parentheses.
    The entire sentence is passed in using y and init because the target sentence
    doesn't have access to the first character and the first character is stripped
    from the input x.

    @param y (list[int]): sentence of parentheses without the first character
    @param init (int): the first character of the sentence
    @param close_idx (list[int]): indices of unique close parentheses in vocabulary
    @param open_idx (list[int]): indices of unique open parentheses in vocabulary
    @return dists (list[int]): None for open parenthesis, distance to corresponding
        open parenthesis for the close parnethesis
    """
    # map open-to-close and close-to-open indices
    oc = dict(zip(open_idx, close_idx)) 
    co = dict(zip(close_idx, open_idx))
    # initialize k stacks, one for each type of parenthesis
    stacks = {open_idx[i]: [] for i in range(len(open_idx))}
    # the first character of the sentence is figuratively given index -1 since y
    # starts with the second character at index 0
    stacks[init] += [-1]
    # initalize empty matrix of distances
    dists = [None] * len(y)
    # for each close parenthesis, calculate accuracy of prediction
    for i, p in enumerate(y):
        # stop at pad token
        if p == pad_idx:
            return dists
        if p in oc.keys():
            # add open parenthesis to stack
            stacks[p].append(i)
        # handle close parenthesis
        elif p in co.keys():
            # pop from corresonding stack
            start = stacks[co[p]].pop()
            dists[i] = i - start
    return dists


def get_LDPA_counts(y, y_pred, init, batch_size, max_dist, close_idx=[4, 5], open_idx=[2, 3], thresh=0.8):
    """
    The closing distance is the distance between a close paren and its corresponding open paren
    For each closing distance, count how many parens in the batch close at that distance.
    For each closing distance, count how many close parens are correctly predicted with a probability 
    mass > thresh between all close paren predictions.
    The two counts can be used to compute the long-distance prediction accuracy.

    @param y Tensor(batch_size * seq_len, ): A flattened batch of target sentence
    @param y_pred Tensor(batch_size * seq_len, vocab_size): Prediction distribution for each word in 
    target sentence
    @param init Tensor(batch_size, ): A batch of first characters of each sentence. The target sentence 
    doesn't contain the first character of that sentence
    @param batch_size
    @param max_dist (int): The maximum sentence length, which is also the maximum distance between open 
    and closed parentheses

    @return ldpa_counts (max_dist + 1, 2): Array holding [num close parens, num prediction above thresh]
    for each closing distance
    """

    # split y into batches
    targets = np.array_split(y.tolist(), batch_size)
    init = init.tolist()

    # create an array that holds [num close parens, num predictions above prob threshold] at each distance
    # max_dist is inclusive
    ldpa_counts = np.zeros((max_dist + 1, 2), dtype = np.int64)

    for i, target in enumerate(targets):
        # calculate distances
        dists = get_distances(target, init[i], close_idx, open_idx)

        # count frequency of each distance
        distances, counts = np.unique([d for d in dists if d], return_counts=True)
        for dist, count in zip(list(distances), list(counts)):
            ldpa_counts[dist, 0] += count

        # calculate long distance prediction accuracy
        for idx, char in enumerate(target):
            if char in close_idx:
                # Computes the probability weight of the correct closed bracket relative to probability of other closed brackets
                pred_prob = torch.exp(y_pred[idx + i * max_dist, char])
                close_prob = torch.sum(torch.exp(y_pred[idx + i * max_dist, close_idx]))
                frac =  pred_prob / close_prob
                if frac >= thresh:
                    ldpa_counts[dists[idx], 1] += 1
    return ldpa_counts