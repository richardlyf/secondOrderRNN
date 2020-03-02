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
    # map close-to-open indices
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
        if p in open_idx:
            # add open parenthesis to stack
            stacks[p].append(i)
        # handle close parenthesis
        elif p in close_idx:
            # pop from corresonding stack
            start = stacks[co[p]].pop()
            dists[i] = i - start
    return dists


def get_LDPA_counts(y, y_pred, init, batch_size, max_dist, close_idx=[4, 5], open_idx=[2, 3], thresh=0.8, pad_idx=0):
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
    y = y.detach().numpy()
    targets = np.array_split(y, batch_size)
    init = init.tolist()

    # create an array that holds [num close parens, num predictions above prob threshold] at each distance
    # max_dist is inclusive
    ldpa_counts = np.zeros((max_dist + 1, 2), dtype=np.int64)
    all_closing_dists = []

    for i, target in enumerate(targets):
        # calculate distances
        closing_dists = get_distances(target, init[i], close_idx, open_idx, pad_idx)
        all_closing_dists.append(closing_dists)

    # (batch_size * sequence_length)
    all_closing_dists = np.asarray(all_closing_dists).reshape(-1)
    # Take out all the None's corresponding to open parens
    all_closing_dists = all_closing_dists[all_closing_dists != None]

    # get indices of all closed parens
    close_paren_indices = np.where(np.isin(y, close_idx))[0]
    # get all actual close parens
    close_parens = y[close_paren_indices]
    # get probability of predicting the actual close paren
    pred_prob = torch.exp(y_pred[close_paren_indices, close_parens])
    # get the sum of the probability of prediction any close paren
    close_prob = torch.sum(torch.exp(y_pred[:, close_idx][close_paren_indices, :]), dim=1)
    # probability mass on the correct close paren (batch_size * sequence_legnth,)
    frac = pred_prob / close_prob

    # Store counts of close parens at each distance
    total_dists, total_counts = np.unique(all_closing_dists, return_counts=True)
    total_dists = total_dists.astype(int)
    ldpa_counts[total_dists, 0] = total_counts

    # Store counts of predictions above threshold
    dists_above_thresh = all_closing_dists[np.where(frac >= thresh)[0]]
    above_dists, above_counts = np.unique(dists_above_thresh, return_counts=True)
    above_dists = above_dists.astype(int)
    # If dists is not empty. It's empty if nothing is above threshold
    if len(above_dists) != 0:
        ldpa_counts[above_dists, 1] = above_counts

    return ldpa_counts