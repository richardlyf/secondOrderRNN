import argparse
import numpy as np
import torchtext, random, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from model import LSTMLanguageModel

global use_cuda
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

global TEXT
TEXT = torchtext.data.Field()
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path="./data/penn/", 
    train="train.txt", 
    validation="valid.txt", 
    test="valid.txt", 
    text_field=TEXT)

TEXT.build_vocab(train, max_size=1000) if False else TEXT.build_vocab(train)
TEXT.vocab.load_vectors('glove.840B.300d')
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, val, test), 
    batch_size=10, 
    device=device, 
    bptt_len=32, 
    repeat=False)


def argParser():
    """
    This function creates a parser object which parses all the flags from the command line
    We can access the parsed command line values using the args object returned by this function
    Usage:
        First field is the flag name.
        dest=NAME is the name to reference when using the parameter (args.NAME)
        default is the default value of the parameter
    Example:
        > python run.py --gpu 0
        args.gpu <-- 0
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', dest='gpu', default='0', help="The gpu number if there's more than one gpu")
    parser.add_argument('--log', dest='log', default='log/', help="directory to save logs")
    parser.add_argument("--epochs", dest="epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=10, help="Size of the minibatch")

    args = parser.parse_args()
    return args


def train():
    """
    """
    pass


def test():
    """
    """
    pass


class Trainer:
    def __init__(self, TEXT, train_iter, val_iter):
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.TEXT = TEXT

    def string_to_batch(self, string, TEXT = self.TEXT):
        relevant_split = string.split() # last two words, ignore ___
        ids = [self.word_to_id(word) for word in relevant_split]
        return Variable(torch.LongTensor(ids, device = torch.device))
        
    def word_to_id(self, word, TEXT = self.TEXT):
        return TEXT.vocab.stoi[word]
    
    def batch_to_input(self, batch):
        ngrams = self.collect_batch_ngrams(batch)
        x = Variable(torch.LongTensor([ngram[:-1] for ngram in ngrams], device = torch.device))
        y = Variable(torch.LongTensor([ngram[-1] for ngram in ngrams], device = torch.device))
        return x, y
    
    def collect_batch_ngrams(self, batch, n = 3):
        data = batch.text.view(-1).data.tolist()
        return [tuple(data[idx:idx + n]) for idx in range(0, len(data) - n + 1)]
    
    def train_model(self, model, num_epochs):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(params = parameters, lr=1e-3)
        criterion = nn.NLLLoss()
        
        for epoch in tqdm(range(num_epochs)):

            epoch_loss = []
            hidden = model.init_hidden()
            model.train()

            for batch in tqdm(train_iter):
                x, y = batch.text, batch.target.view(-1)
                if use_cuda: x, y = x.cuda(), y.cuda()

                optimizer.zero_grad()

                y_pred, hidden = model.forward(x, hidden, train = True)

                loss = criterion(y_pred, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.lstm.parameters(), 1)

                optimizer.step()

                epoch_loss.append(loss.item())
                
            model.eval()
            train_ppl = np.exp(np.mean(epoch_loss))
            val_ppl = self.validate(model)

            print('Epoch {0} | Loss: {1} | Train PPL: {2} | Val PPL: {3}'.format(epoch+1, np.mean(epoch_loss), train_ppl,  val_ppl))
    
        print('Model trained.')
        self.write_kaggle(model)
        print('Output saved.')
        
    def validate(self, model):
        criterion = nn.NLLLoss()
        hidden = model.init_hidden()
        aggregate_loss = []
        for batch in val_iter:
            y_p, _ = model.forward(batch.text, hidden, train = False)
            y_t = batch.target.view(-1)
            
            loss = criterion(y_p, y_t)
            aggregate_loss.append(loss.data[0])        
        val_ppl = np.exp(np.mean(aggregate_loss))
        return val_ppl
    
    def predict_sentence(self, string, model, TEXT):
        string = string[:-4]
        model.batch_size = 1
        hidden = model.init_hidden()
        x = self.string_to_batch(string, TEXT)
        logits, _ = model.forward(x, hidden, train = False)
        argsort_ids = np.argsort(logits[-1].data.tolist())
        out_ids = argsort_ids[-20:][::-1]
        out_words = ' '.join([TEXT.vocab.itos[out_id] for out_id in out_ids])
        return out_words
    
    def write_kaggle(self, model, TEXT = self.TEXT, input_file = 'input.txt'):        
        inputs = open(input_file, 'r').read().splitlines()
        outputs = [self.predict_sentence(sentence, model, TEXT) for sentence in inputs]
        with open('lstm_output.txt', 'w') as f:
            f.write('id,word')
            for idx, line in enumerate(outputs):
                f.write('\n')
                f.write(str(idx) + ',')
                f.write(line) 


def main():
    """
    """
    model = LSTMLanguageModel(hidden_dim = 1024, TEXT = TEXT)
    model = model.to(device)

    trainer = Trainer(train_iter = train_iter, val_iter = val_iter)
    trainer.train_model(model = model, num_epochs = 10)


if __name__ == "__main__":
    main()