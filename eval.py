import itertools
import numpy as np
import torch

def get_distances(y, init, close_idx=[3, 5], open_idx=[2, 4], pad_idx=0):
    """
    Calculate distance between pairs of open and close parentheses

    @param y (list[int]): sentence of parentheses
    @param close_idx (list[int]): indices of close parentheses in vocabulary
    @param open_idx (list[int]): indices of open parentheses in vocabulary
    @return dists (list[int]): None for open parenthesis, distance to corresponding
        open parenthesis for the close parnethesis
    """
    # map open-to-close and close-to-open indices
    oc = dict(zip(open_idx, close_idx)) 
    co = dict(zip(close_idx, open_idx))
    # initialize k stacks, one for each type of parenthesis
    stacks = {open_idx[i]: [] for i in range(len(open_idx))}
    stacks[init] += [-1]
    # initalize empty matrix of distances
    dists = [None]*len(y)
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


def LDPA(y, y_pred, init, batch_size, close_idx=[3, 5], open_idx = [2, 4], thresh=0.8, pad_idx=0):
    """
    Calculate long distance prediction accuracy for a batch of sentences
    """

    # split y into chunks on padding character
    targets = np.array_split(y.tolist(), batch_size)

    # calculate max distance
    max_dist = len(targets[0])
    ldpa = {i : [0, 0] for i in range(max_dist)}

    for i, target in enumerate(targets):
        # calculate distances
        dists = get_distances(target, init[i], close_idx=[3, 5], open_idx = [2, 4])

        # count frequency of each distance
        values, counts = np.unique([d for d in dists if d], return_counts=True)
        for v, c in zip(list(values), list(counts)):
            ldpa[v][0] += c

        # calculate long distance prediction accuracy
        for idx, char in enumerate(target):
            if char in close_idx:
                frac = torch.exp(y_pred[idx, char]) / torch.sum(torch.exp(y_pred[idx, close_idx]))
                if frac >= thresh:
                    ldpa[dists[idx]][1] += 1
    return ldpa