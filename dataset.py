import os
import json
import numpy as np
from torch.utils.data import Dataset
from collections import Counter
from itertools import chain


def get_processed_dataset_path(dataset_path):
    """
    Return two dataset paths
    The first is .npy and the second is .json
    npy file stores all the preprocessed data
    json file stores all the vocab and its indices
    """
    npy_path = dataset_path[:-4] + ".npy"
    json_path = dataset_path[:-4] + ".json"
    return npy_path, json_path


def tokenize_parens(string):
    """
    Tokenizer function for synthetic parenthesis dataset
    """
    return string.replace("END", "").split()

def tokenize_ptb(string):
    """
    Tokenizer function for synthetic parenthesis dataset
    """
    return string.split()

def pad_sents(sents, pad_token):
    """ 
    Pad list of sentences according to the longest sentence in the batch.
    The paddings should be at the end of each sentence.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentencesnce are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []
    longest_length = max(len(sentence) for sentence in sents)
    sents_padded = [sentence + [pad_token] * (longest_length - len(sentence)) for sentence in sents]
    return sents_padded


def preprocess_parens_dataset(dataset_path):
    """
    Preprocesses a data file to generate a vocab list and a npy file that stores 
    (input, target) pairs.
    This will be called to generate dataset for train, val, and test.
    The .json and .npy file names have the same root name as the dataset_path.
    """
    npy_path, json_path = get_processed_dataset_path(dataset_path)

    # Create the corpus by splitting the file by word
    corpus = []
    with open(dataset_path, 'r') as f:
        for line in f:
            line = tokenize_parens(line)
            corpus.append(line)

    # Create the vocab and its mapping to indices
    vocab = Vocab(corpus=corpus)
    vocab.save(json_path)
    # Transform sentences to corresponding indices
    sentences = vocab.words2indices(corpus)
    # add start and end tokens
    sentences = [[vocab['<start>']] + s + [vocab['<end>']] for s in sentences]
    # add padding
    padded_sentences = pad_sents(sentences, vocab['<pad>'])

    # Create the dataset of (input, target) pairs
    dataset = []
    for sentence in padded_sentences:
            input_ = sentence[:-1]
            target_ = sentence[1:]
            sample = [input_, target_]
            dataset.append(sample)

    # Save to npy file for future use
    dataset = np.asarray(dataset)
    np.save(npy_path, dataset, allow_pickle=True)


class ParensDataset(Dataset):
    def __init__(self, dataset_path):
        processed_dataset_path, json_path = get_processed_dataset_path(dataset_path)
        # Create the preprocessed dataset if it doesn't already exist
        if not os.path.exists(processed_dataset_path):
            preprocess_dataset(dataset_path, tokenize_parens)
        # Load the dataset from npy file
        self.dataset = np.load(processed_dataset_path, allow_pickle=True)
        # Load the vocab
        self.vocab = Vocab(file_path=json_path)


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        input_, target_ = self.dataset[idx]
        return input_, target_


    def get_vocab(self):
        return self.vocab


class Vocab(object):
    """ Vocabulary object that contains mapping from vocab words to indicies"""
    def __init__(self, file_path=None, corpus=None):
        """
        Init Vocab Instance from json file or corpus.
        If corpus is passed in, create a new vocab object using words in corpus and save as json file
        If existing json file is passed in, load vocab object using json file
        @param file_path: file path to json
        @param corpus (list[lsit[str]]): Read in from text file and then tokenized
        """
        if not file_path and not corpus:
            raise IllegalArgumentException("Need to pass in file_path or corpus")
        if file_path:
            entry = json.load(open(file_path, 'r'))
            self.word2id = entry["word2id"]
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0   # Pad Token
            self.word2id['<unk>'] = 1   # Unknown Token
            self.word2id['<start>'] = 2 # Start Token
            self.word2id['<end>'] = 3   # End Token
            self.id2word = {v: k for k, v in self.word2id.items()}
            word_freq = Counter(chain(*corpus))
            unique_words = [w for w, v in sorted(word_freq.items())]
            for word in unique_words:
                self.add(word)

        self.pad_id = self.word2id['<pad>']
        self.unk_id = self.word2id['<unk>']
        self.start_token = self.word2id['<start>']
        self.end_token = self.word2id['<end>']
        self.id2word = {v: k for k, v in self.word2id.items()}
        

    def __getitem__(self, word):
        """ Retrieve word's index. Return the index for the unk
        token if the word is out of vocabulary.
        @param word (str): word to look up.
        @returns index (int): index of word 
        """
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        """ Check if word is captured by VocabEntry.
        @param word (str): word to look up
        @returns contains (bool): whether word is contained    
        """
        return word in self.word2id

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the VocabEntry.
        """
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        """ Compute number of words in VocabEntry.
        @returns len (int): number of words in VocabEntry
        """
        return len(self.word2id)

    def __repr__(self):
        """ Representation of VocabEntry to be used
        when printing the object.
        """
        return 'Vocabulary[size=%d]' % len(self)

    def save(self, file_path):
        """
        Save Vocab to file as JSON dump.
        @param file_path (str): file path to vocab file
        """
        json.dump(dict(word2id=self.word2id), open(file_path, 'w'), indent=2)

    def id2word(self, wid):
        """ Return mapping of index to word.
        @param wid (int): word index
        @returns word (str): word corresponding to index
        """
        return self.id2word[wid]

    def add(self, word):
        """ Add word to VocabEntry, if it is previously unseen.
        @param word (str): word to add to VocabEntry
        @return index (int): index that the word has been assigned
        """
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        """ Convert list of words or list of sentences of words
        into list or list of list of indices.
        @param sents (list[str] or list[list[str]]): sentence(s) in words
        @return word_ids (list[int] or list[list[int]]): sentence(s) in indices
        """
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        """ Convert list of indices into words.
        @param word_ids (list[int]): list of word ids
        @return sents (list[str]): list of words
        """
        return [self.id2word[w_id] for w_id in word_ids]