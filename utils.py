
################    #####################
def string_to_batch(string, device):
    """
    """
    relevant_split = string.split() # last two words, ignore ___
    ids = [word_to_id(word) for word in relevant_split]
    return Variable(torch.LongTensor(ids, device = device))
    

def word_to_id(word, TEXT):
    return TEXT.vocab.stoi[word]


def batch_to_input(batch, device):
    ngrams = collect_batch_ngrams(batch)
    x = Variable(torch.LongTensor([ngram[:-1] for ngram in ngrams], device = device))
    y = Variable(torch.LongTensor([ngram[-1] for ngram in ngrams], device = device))
    return x, y


def collect_batch_ngrams(batch, n = 3):
    data = batch.text.view(-1).data.tolist()
    return [tuple(data[idx:idx + n]) for idx in range(0, len(data) - n + 1)]


def predict_sentence(string, model, TEXT):
    string = string[:-4]
    model.batch_size = 1
    hidden = model.init_hidden()
    x = string_to_batch(string, TEXT)
    logits, _ = model.forward(x, hidden, train = False)
    argsort_ids = np.argsort(logits[-1].data.tolist())
    out_ids = argsort_ids[-20:][::-1]
    out_words = ' '.join([TEXT.vocab.itos[out_id] for out_id in out_ids])
    return out_words


def write_kaggle(model, TEXT, input_file = 'input.txt'):        
    inputs = open(input_file, 'r').read().splitlines()
    outputs = [predict_sentence(sentence, model, TEXT) for sentence in inputs]
    with open('lstm_output.txt', 'w') as f:
        f.write('id,word')
        for idx, line in enumerate(outputs):
            f.write('\n')
            f.write(str(idx) + ',')
            f.write(line) 