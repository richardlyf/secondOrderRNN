import torch
from tqdm import tqdm
import numpy as np


def generate_sentences(model, vocab, batch_size, max_sentence_length=10, is_penn=False, device=None):
    """
    Returns a single generated sentence using the language model.
    @param model: The language model with weights pre-loaded
    @param vocab: Vocab object containing mapping from word to indices
    @param batch_size: Number of sentences to generate at a time
    @param max_sentence_length: Max length of a generated sentence if <end> is not generated
    @param is_penn: Boolean of whether the model is for PTB dataset
    @param device: gpu or cpu
    @return sents List[string]: A list of sentences
    """
    # Generate start tokens
    sents = torch.tensor([vocab.start_id] * batch_size).view(batch_size, 1).to(device)
    init_state = model.init_lstm_state(device)
    for seq_idx in tqdm(range(max_sentence_length)):
        y_pred, ret_state = model(sents, init_state)
        init_state = ret_state if is_penn else init_state
        next_words = greedy_select(y_pred)
        sents = torch.stack((sents, next_words), dim=1).to(device)
    sents = tensor_to_sentences(sents, vocab)
    return sents


def tensor_to_sentences(sents_tensor, vocab):
    """
    Takes in a tensor of predicted word indices and converts it into a list of sentences
    @param sents Tensor(batch_size, max_sentence_length): Predicted word indices
    @param vocab: Vocab object containing mapping from word to indices
    @return sents List[string]: A list of sentences
    """
    shape = sents_tensor.shape
    # (batch_size * max_sentence_length, )
    flat_sents = sents_tensor.view(-1).cpu().detach().tolist()
    sents_np = np.asarray(vocab.indices2words(flat_sents))
    # (batch_size, max_sentence_length)
    sents_np = sents_np.reshape(shape)
    sents = []
    for sent in sents_np:
        # Join all words into string but remove start token
        sent = " ".join(sent[1:])
        end_idx = sent.find(vocab.id2word[vocab.end_id])
        if end_idx != -1:
            sent = sent[:end_idx]
        sents.append(sent.strip())
    return sents


def greedy_select(y_pred, batch_size):
    """
    Greedily select indices with the highest probability.
    Only consider predictions for the latest time sequence.
    @param y_pred (batch_size * seq_len, vocab_size): Probability distribution of all next words
    in the batch
    @param batch_size: Number of sentences to generate at a time
    @return next_words (batch_size, 1): Indicies of the next selected words
    """
    # (batch_size, seq_len, vocab_size)
    y_pred = y_pred.view(batch_size, -1, y_pred.size(1))
    # (batch_size, vocab_size)
    pred_latest = y_pred[:, -1]
    # Convert log probability to probability
    prob_latest = torch.exp(pred_latest)
    next_words = torch.argmax(prob_latest, dim=1).view(-1, 1)

    return next_words


####### TESTS #######
def tests():
    print("-"*80)
    # Test greedy_select()
    target = np.array([0, 1, 2]).reshape(3, 1)
    y_prob = np.array([[0.8, 0.1, 0.1], [0.3, 0.5, 0.2], [0.3, 0.1, 0.6]])
    y_pred = np.log(y_prob)
    y_pred = torch.tensor(y_pred)
    output = greedy_select(y_pred, batch_size=3).numpy()
    assert(np.allclose(target, output)), "Tests for greedy_select failed!\n Expected {} \n Got {} \n".format(target, output)

    # Test tensor_to_sentences()
    class testVocab():
        def __init__(self):
            self.id2word = {0: "end", 1: "word1", 2: "word2", 3: "start"}
            self.end_id = 0
        def indices2words(self, word_ids):
            return [self.id2word[w_id] for w_id in word_ids]
    vocab = testVocab()
    sents_tensor = torch.tensor([[3, 1, 2, 1, 0], [3, 2, 0, 1, 0]])
    target = ["word1 word2 word1", "word2"]
    output = tensor_to_sentences(sents_tensor, vocab)
    assert(np.array_equal(target, output)), "Tests for tensor_to_sentences failed!\n Expected {} \n Got {} \n".format(target, output)

    print("All Sanity Checks Passed!")
    print ("-"*80)


if __name__ == '__main__':
    tests()