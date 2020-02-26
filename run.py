import argparse
import numpy as np
import torchtext, random, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from model import LSTMLanguageModel


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
    parser.add_argument("--train_path", dest="train_path", help="Training data file")
    parser.add_argument("--valid_path", dest="valid_path", help="Validation data file")
    parser.add_argument("--bptt_len", dest="bptt_len", default=32, type=int, help="Length of sequences for backpropagation through time")
    parser.add_argument("--hidden-size", dest="hidden_size", type=int, default=256, help="dimension fo hidden layer")
    parser.add_argument("--dropout", dest="dropout", type=float, default=0.3, help="dropout rate")

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
    def __init__(self, args, device):

        self.device=  device
        self.TEXT = self.build_dataset(args)

        self.model = LSTMLanguageModel(
            TEXT = self.TEXT,
            hidden_dim = args.hidden_size, 
            batch_size = args.batch_size, 
            dropout_rate=args.dropout)

        self.model = self.model.to(device)

    def build_dataset(self, args):
        TEXT = torchtext.data.Field()
        train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
            path=".", 
            train=args.train_path, 
            validation=args.valid_path, 
            test=args.valid_path, 
            text_field=TEXT)

        TEXT.build_vocab(train, max_size=1000) if False else TEXT.build_vocab(train)
        TEXT.vocab.load_vectors('glove.840B.300d')
        train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
            (train, val, test), 
            batch_size=args.batch_size, 
            device=self.device, 
            bptt_len=args.bptt_len, 
            repeat=False)

        self.train_iter = train_iter
        self.val_iter = val_iter
        return TEXT

    def string_to_batch(self, string):
        relevant_split = string.split() # last two words, ignore ___
        ids = [self.word_to_id(word) for word in relevant_split]
        return Variable(torch.LongTensor(ids, device = self.device))
        
    def word_to_id(self, word):
        return self.TEXT.vocab.stoi[word]
    
    def batch_to_input(self, batch):
        ngrams = self.collect_batch_ngrams(batch)
        x = Variable(torch.LongTensor([ngram[:-1] for ngram in ngrams], device = self.device))
        y = Variable(torch.LongTensor([ngram[-1] for ngram in ngrams], device = self.device))
        return x, y
    
    def collect_batch_ngrams(self, batch, n = 3):
        data = batch.text.view(-1).data.tolist()
        return [tuple(data[idx:idx + n]) for idx in range(0, len(data) - n + 1)]
    
    def train_model(self, num_epochs):
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(params = parameters, lr=1e-3)
        criterion = nn.NLLLoss()
        
        for epoch in tqdm(range(num_epochs)):

            epoch_loss = []
            hidden = self.model.init_hidden()
            self.model.train()

            for batch in tqdm(self.train_iter):
                x, y = batch.text, batch.target.view(-1)

                optimizer.zero_grad()

                y_pred, hidden = self.model.forward(x, hidden, train = True)

                loss = criterion(y_pred, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.lstm.parameters(), 1)

                optimizer.step()

                epoch_loss.append(loss.item())
                
            self.model.eval()
            train_ppl = np.exp(np.mean(epoch_loss))
            val_ppl = self.validate(self.model)

            print('Epoch {0} | Loss: {1} | Train PPL: {2} | Val PPL: {3}'.format(epoch+1, np.mean(epoch_loss), train_ppl,  val_ppl))
    
        print('Model trained.')
        self.write_kaggle(self.model)
        print('Output saved.')
        
    def validate(self):
        criterion = nn.NLLLoss()
        hidden = self.model.init_hidden()
        aggregate_loss = []
        for batch in val_iter:
            y_p, _ = self.model.forward(batch.text, hidden, train = False)
            y_t = batch.target.view(-1)
            
            loss = criterion(y_p, y_t)
            aggregate_loss.append(loss.data[0])        
        val_ppl = np.exp(np.mean(aggregate_loss))
        return val_ppl
    
    def predict_sentence(self, string):
        string = string[:-4]
        self.model.batch_size = 1
        hidden = self.model.init_hidden()
        x = self.string_to_batch(string, self.TEXT)
        logits, _ = self.model.forward(x, hidden, train = False)
        argsort_ids = np.argsort(logits[-1].data.tolist())
        out_ids = argsort_ids[-20:][::-1]
        out_words = ' '.join([TEXT.vocab.itos[out_id] for out_id in out_ids])
        return out_words
    
    def write_kaggle(self, input_file = 'input.txt'):        
        inputs = open(input_file, 'r').read().splitlines()
        outputs = [self.predict_sentence(sentence, self.model, self.TEXT) for sentence in inputs]
        with open('lstm_output.txt', 'w') as f:
            f.write('id,word')
            for idx, line in enumerate(outputs):
                f.write('\n')
                f.write(str(idx) + ',')
                f.write(line) 


def main():
    """
    """
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    args = argParser()
    trainer = Trainer(args, device)
    trainer.train_model(num_epochs = args.epochs)

if __name__ == "__main__":
    main()