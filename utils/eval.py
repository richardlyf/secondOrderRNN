import itertools
import numpy as np
import torch
import torch.nn.functional as F

def validate(model, criterion, val_dataset, is_stream, device, \
        vocab=None, stats_output_file=None):
    """
    Computes both perplexity and WCPA on the validation set
    Aggregate version of validate_ppl and validate_wcpa so we loop through
    validation set only once and save training time.

    If stats_output_file is not None, we will be collecting statistics on the test sentences
    and store them in stats_output_file
    """
    x, y = next(iter(val_dataset))
    batch_size, max_sents_len = x.size()
    total_ldpa_counts = np.zeros((max_sents_len, 2), dtype=np.int64)
    aggregate_loss = []
    # syntax stats
    if stats_output_file is not None:
        f = open(stats_output_file, "w")

    # initialize hidden state
    init_state = model.init_lstm_state(device=device)
    for sentence_id, batch in enumerate(val_dataset):
        x, y = batch
        x = x.to(device)
        y = y.view(-1).to(device)
        
        y_p, ret_state = model(x, init_state)
        init_state = ret_state if is_stream else init_state
        # syntax stats
        if stats_output_file is not None:
            record_complexity(y_p, y, sentence_id, vocab, f=f)
        # ppl
        loss = criterion(y_p, y)
        aggregate_loss.append(loss.item())
        # wcpa
        if not is_stream:
            ldpa_counts = get_LDPA_counts(y=y, y_pred=y_p, batch_size=batch_size,
                max_dist=max_sents_len)
            total_ldpa_counts += ldpa_counts

    # close file
    if stats_output_file is not None:
        f.close()
    # val_loss and ppl
    val_loss = np.mean(aggregate_loss)
    val_ppl = np.exp(val_loss)
    # wcpa and ldpa
    if not is_stream:
        valid_dist = np.where(total_ldpa_counts[:, 0] > 0)
        ldpa = total_ldpa_counts[valid_dist, 1] / total_ldpa_counts[valid_dist, 0]  
        wcpa = np.min(ldpa)
    if is_stream:
        return val_ppl, val_loss, -1, None
    else:
        return val_ppl, val_loss, wcpa, (valid_dist, ldpa)


def validate_ppl(model, criterion, val_dataset, device):
    """
    Runs through the validation set and computes the perplexity of the model
    """
    aggregate_loss = []
    for batch in val_dataset:
        x, y = batch
        x = x.to(device)
        y = y.view(-1).to(device)
        y_p = model(x)
        
        loss = criterion(y_p, y)
        aggregate_loss.append(loss.item())        
    val_loss = np.mean(aggregate_loss)
    val_ppl = np.exp(val_loss)
    return val_ppl, val_loss
    

def validate_wcpa(model, val_dataset, device):
    """
    Computes worst-case long-distance prediction accuracy (WCPA) of the model
    """
    # max sentence length is the second dimension of x in val_dataset
    x, y = next(iter(val_dataset))
    batch_size, max_sents_len = x.size()
    # initialize storage for long distance prediction accuracy
    total_ldpa_counts = np.zeros((max_sents_len, 2), dtype=np.int64)

    for batch in val_dataset:
        x, y = batch
        x = x.to(device)
        y = y.view(-1).to(device)
        y_p = model(x)

        # calculate counts for LDPA metric
        ldpa_counts = get_LDPA_counts(y=y, y_pred=y_p, batch_size=batch_size,
            max_dist=max_sents_len)
        total_ldpa_counts += ldpa_counts

    # calculate LDPA
    valid_dist = np.where(total_ldpa_counts[:, 0] > 0)
    ldpa = total_ldpa_counts[valid_dist, 1] / total_ldpa_counts[valid_dist, 0]  
    wcpa = np.min(ldpa) 
    return wcpa, (valid_dist, ldpa)


def record_complexity(y_pred, target, sentence_id, vocab, f=None):
    """
    For test only.
    Used to record statistics on predicted words. The log can then be used to analyze
    long term sentence dependencies.
    Tests for sentence dependencies are found at repo: https://github.com/richardlyf/LM_syneval
    @param y_pred (batch_size * seq_len, vocab_size): Output of the model
    @param target (batch_size * seq_len, ): Correct indices for the prediction
    @sentence_id int: The index for the sentence in the batch, used to track which test sentence it is
    @vocab: Object that stores mapping from words to word indices
    @f file object: Results will be written to this opened file.
    @return None
    """
    # Compute entropy
    probs = torch.exp(y_pred)
    # (batch_size * seq_len)
    entropy = -1 * torch.sum(probs * y_pred, dim=1)

    # Compute Shannon surprise
    surprises = -1 * y_pred

    for target_pos, target_index in enumerate(target):
        # Stop at the end token
        if target_index == vocab.end_id:
            break
        target_index = target_index.item()
        word = vocab.id2word[target_index]
        surprise = surprises[target_pos][target_index]
        stat = str(word) + ' ' + str(sentence_id) + ' ' + str(target_pos) + ' ' + str(len(word)) + ' ' + str(surprise.item()) + ' ' + str(entropy[target_pos].item())
        f.write(stat)
        f.write("\n")


def get_distances(y, close_idx=[6, 7], open_idx=[4, 5], pad_idx=0):
    """
    Calculate distance between pairs of open and close parentheses.

    @param y (list[int]): sentence of parentheses without the first character
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


def get_LDPA_counts(y, y_pred, batch_size, max_dist, close_idx=[6, 7], open_idx=[4, 5], thresh=0.8, pad_idx=0):
    """
    The closing distance is the distance between a close paren and its corresponding open paren
    For each closing distance, count how many parens in the batch close at that distance.
    For each closing distance, count how many close parens are correctly predicted with a probability 
    mass > thresh between all close paren predictions.
    The two counts can be used to compute the long-distance prediction accuracy.

    @param y Tensor(batch_size * seq_len, ): A flattened batch of target sentence
    @param y_pred Tensor(batch_size * seq_len, vocab_size): Prediction distribution for each word in 
    target sentence
    @param batch_size
    @param max_dist (int): The maximum sentence length, which is also the maximum distance between open 
    and closed parentheses

    @return ldpa_counts (max_dist, 2): Array holding [num close parens, num prediction above thresh]
    for each closing distance
    """

    # split y into batches
    y = y.cpu().detach().numpy()
    targets = np.array_split(y, batch_size)

    # create an array that holds [num close parens, num predictions above prob threshold] at each distance
    # max_dist is inclusive
    ldpa_counts = np.zeros((max_dist, 2), dtype=np.int64)
    all_closing_dists = []

    for i, target in enumerate(targets):
        # calculate distances
        closing_dists = get_distances(target, close_idx, open_idx, pad_idx)
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
    frac = (pred_prob / close_prob).cpu().detach().numpy()

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