import argparse
import numpy as np
import torchtext, random, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from logger import Logger
from utils import *
from model import ModelChooser


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
    parser.add_argument('--log', dest='log', default='', help="unique log directory name under log/. If the name is empty, do not store logs")
    parser.add_argument('--model', dest='model', default='baseline_lstm', help="name of model to use")
    parser.add_argument("--epochs", dest="epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=10, help="Size of the minibatch")
    parser.add_argument("--train-path", dest="train_path", help="Training data file")
    parser.add_argument("--valid-path", dest="valid_path", help="Validation data file")
    parser.add_argument("--bptt-len", dest="bptt_len", default=32, type=int, help="Length of sequences for backpropagation through time")
    parser.add_argument("--hidden-size", dest="hidden_dim", type=int, default=256, help="dimension fo hidden layer")
    parser.add_argument("--dropout", dest="dropout_rate", type=float, default=0.3, help="dropout rate")

    args = parser.parse_args()
    return args


def build_dataset(args):
    TEXT = torchtext.data.Field()
    train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
        path=".", 
        train=args.train_path, 
        validation=args.valid_path, 
        test=args.valid_path, 
        text_field=TEXT)

    TEXT.build_vocab(train, max_size=1000) if False else TEXT.build_vocab(train)
    TEXT.vocab.load_vectors('glove.840B.300d')
    return TEXT, (train, val, test)


def train(model, dataset, TEXT, args, device, num_epochs):

    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
        dataset, 
        batch_size=args.batch_size, 
        device=device, 
        bptt_len=args.bptt_len, 
        repeat=False)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params = parameters, lr=3e-4)
    criterion = nn.NLLLoss()
    
    for epoch in tqdm(range(num_epochs)):

        epoch_loss = []
        hidden = model.init_hidden()
        model.train()

        for batch in tqdm(train_iter):
            x, y = batch.text, batch.target.view(-1)

            optimizer.zero_grad()

            y_pred, hidden = model.forward(x, hidden, train=True)

            loss = criterion(y_pred, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.lstm.parameters(), 1)

            optimizer.step()

            epoch_loss.append(loss.item())
            
        model.eval()
        train_ppl = np.exp(np.mean(epoch_loss))
        val_ppl = validate(model, val_iter)

        print('Epoch {0} | Loss: {1} | Train PPL: {2} | Val PPL: {3}'.format(epoch+1, np.mean(epoch_loss), train_ppl,  val_ppl))

    print('Model trained.')

        
def validate(model, val_iter):
    criterion = nn.NLLLoss()
    hidden = model.init_hidden()
    aggregate_loss = []
    for batch in val_iter:
        y_p, _ = model.forward(batch.text, hidden, train = False)
        y_t = batch.target.view(-1)
        
        loss = criterion(y_p, y_t)
        aggregate_loss.append(loss.item())        
    val_ppl = np.exp(np.mean(aggregate_loss))
    return val_ppl
    

def main():
    # setup
    args = argParser()
    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else "cpu")
    unique_logdir = create_unique_logdir(args.log)
    logger = Logger(unique_logdir) if args.log != '' else None

    # build TEXT object
    TEXT, dataset = build_dataset(args)

    # build model
    kwargs = vars(args) # Turns args into a dictionary
    kwargs["TEXT"] = TEXT
    model = ModelChooser(args.model, **kwargs)
    model = model.to(device)
    
    # train model
    train(model, dataset, TEXT, args, device, num_epochs=args.epochs)


if __name__ == "__main__":
    main()