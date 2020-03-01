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


def LDPA(y, y_pred, init, batch_size, close_idx=[4, 5], open_idx=[2, 3], thresh=0.8, pad_idx=0):
    """
    Calculate long distance prediction accuracy for a batch of sentences

    @param y Tensor(batch_size * seq_len, ): A flattened batch of target sentence
    @param y_pred Tensor(batch_size * seq_len, vocab_size): Prediction distribution for each word in target sentence
    @param init Tensor(batch_size, ): A batch of first characters of each sentence. The target sentence doesn't contain
    the first character of that sentence
    """

    # split y into batches
    targets = np.array_split(y.tolist(), batch_size)
    init = init.tolist()

    # calculate max distance
    max_dist = len(targets[0])
    # Create a dictionary mapping distance to [num close parens at this distance, num predictions above prob threshold at this distance]
    # max_dist is inclusive
    ldpa = {d : [0, 0] for d in range(max_dist + 1)}

    for i, target in enumerate(targets):
        # calculate distances
        dists = get_distances(target, init[i], close_idx=[4, 5], open_idx=[2, 3])

        # count frequency of each distance
        distances, counts = np.unique([d for d in dists if d], return_counts=True)
        for dist, count in zip(list(distances), list(counts)):
            ldpa[dist][0] += count

        # calculate long distance prediction accuracy
        for idx, char in enumerate(target):
            if char in close_idx:
                # Computes the probability weight of the correct closed bracket relative to probability of other closed brackets
                frac = y_pred[idx + i * max_dist, char] / torch.sum(y_pred[idx + i * max_dist, close_idx])
                if frac >= thresh:
                    ldpa[dists[idx]][1] += 1
    return ldpa