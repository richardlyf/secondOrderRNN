import argparse
import numpy as np
import os
import torchtext, random, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from logger import Logger
from utils import *
from model import ModelChooser
from dataset import *


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

    parser.add_argument('--gpu', dest='gpu', type=int, default=0, help="The gpu number if there's more than one gpu")
    parser.add_argument('--log', dest='log', default='', help="Unique log directory name under log/. If the name is empty, do not store logs")
    parser.add_argument('--log_every', dest='log_every', type=int, default=100, help="Number of itertions between logging to tensorboard within an epoch")
    parser.add_argument('--lr', dest='lr', type=float, default=3e-4, help="Learning rate for training")
    parser.add_argument('--model', dest='model', default='baseline_lstm', help="Name of model to use")
    parser.add_argument('--embedding_dim', dest='embedding_dim', type=int, default=12, help="Size of the word embedding")
    parser.add_argument('--is_parens', dest='is_parens', type=int, default=1, help="Train on the parenthesis dataset when 1, 0 on TreeBank dataset")
    parser.add_argument("--epochs", dest="epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=10, help="Size of the minibatch")
    parser.add_argument("--train-path", dest="train_path", help="Training data file")
    parser.add_argument("--valid-path", dest="valid_path", help="Validation data file")
    parser.add_argument("--bptt-len", dest="bptt_len", default=32, type=int, help="Length of sequences for backpropagation through time")
    parser.add_argument("--hidden-size", dest="hidden_dim", type=int, default=256, help="Dimension fo hidden layer")
    parser.add_argument("--dropout", dest="dropout_rate", type=float, default=0.3, help="Dropout rate")

    args = parser.parse_args()
    return args


def train(model, dataset, TEXT, args, device, num_epochs, logger=None):
    batch_size = args.batch_size
    backprop_len = args.bptt_len
    log_every = args.log_every
    lr = args.lr
    save_to_log = logger is not None

    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
        dataset, 
        batch_size=batch_size, 
        device=device, 
        bptt_len=backprop_len, 
        repeat=False)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=lr)
    criterion = nn.NLLLoss()

    log_step = 0
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = []
        hidden = model.init_hidden()
        model.train()

        for batch_iter, batch in enumerate(tqdm(train_iter)):
            x, y = batch.text, batch.target.view(-1)
            optimizer.zero_grad()
            y_pred, hidden = model.forward(x, hidden, train=True)

            loss = criterion(y_pred, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.lstm.parameters(), 1)
            optimizer.step()
            epoch_loss.append(loss.item())

            # Add to logger on tensorboard within an epoch
            if save_to_log and batch_iter % log_every == 0:
                average_train_loss = np.mean(epoch_loss)
                average_train_ppl = np.exp(average_train_loss)
                logger.scalar_summary("average_training_loss", average_train_loss, log_step)
                logger.scalar_summary("average_training_ppl", average_train_ppl, log_step)
                log_step += 1
            
        model.eval()
        epoch_average_loss = np.mean(epoch_loss)
        epoch_train_ppl = np.exp(epoch_average_loss)
        epoch_val_ppl = validate(model, val_iter)

        # Add to logger on tensorboard at the end of an epoch
        if save_to_log:
            logger.scalar_summary("epoch_training_loss", epoch_average_loss, epoch)
            logger.scalar_summary("epoch_train_ppl", epoch_train_ppl, epoch)
            logger.scalar_summary("epoch_val_ppl", epoch_val_ppl, epoch)

        print('Epoch {0} | Loss: {1} | Train PPL: {2} | Val PPL: {3}' \
            .format(epoch + 1, epoch_average_loss, epoch_train_ppl, epoch_val_ppl))

    print('Model trained.')

        
def validate(model, val_iter):
    criterion = nn.NLLLoss()
    hidden = model.init_hidden()
    aggregate_loss = []
    for batch in val_iter:
        y_p, _ = model.forward(batch.text, hidden, train=False)
        y_t = batch.target.view(-1)
        
        loss = criterion(y_p, y_t)
        aggregate_loss.append(loss.item())        
    val_ppl = np.exp(np.mean(aggregate_loss))
    return val_ppl
    

def main():
    # setup
    print("Setting up...")
    args = argParser()
    args.is_parens = True if args.is_parens == 1 else False
    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else "cpu")
    unique_logdir = create_unique_logdir(args.log)
    logger = Logger(unique_logdir) if args.log != '' else None
    print("All training logs will be saved to: ", unique_logdir)
    print("Will log to tensorboard: ", logger is not None)

    # build TEXT object
    print("Creating TEXT...")
    TEXT, dataset = build_dataset(args, is_parens=args.is_parens)
    print("Done!")

    # build model
    kwargs = vars(args) # Turns args into a dictionary
    kwargs["TEXT"] = TEXT
    model = ModelChooser(args.model, **kwargs)
    model = model.to(device)
    
    # train model
    print("Starting training...")
    train(model, dataset, TEXT, args, device, num_epochs=args.epochs, logger=logger)


if __name__ == "__main__":
    main()