import argparse
import numpy as np
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.logger import Logger
from utils.utils import *
from utils.eval import *
from utils.word_generation import generate_sentences
from model.model import ModelChooser
from model.dataset import *

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

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
    # General model specification
    parser.add_argument("--mode", dest="mode", default='train', help="Mode is one of 'train', 'test', 'generate'")
    parser.add_argument("--model", dest="model", default="baseline_lstm", help="Name of model to use")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=10, help="Size of the minibatch, or num lines to generate")
    parser.add_argument("--embedding-size", dest='embed_size', type=int, default=12, help="Size of the word embedding")
    parser.add_argument("--hidden-size", dest="hidden_size", type=int, default=256, help="Dimension of hidden layer")
    parser.add_argument("--num-layers", dest='num_layers', type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--epochs", dest="epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--dropout", dest="dropout_rate", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--lr", dest="lr", type=float, default=3e-4, help="Learning rate for training")
    parser.add_argument("--gpu", dest="gpu", type=str, default='0', help="The gpu number if there's more than one gpu")
    # Dataset and log paths
    parser.add_argument("--log", dest="log", default='', help="Unique log directory name under log/. If the name is empty, do not store logs")
    parser.add_argument("--log-every", dest="log_every", type=int, default=5, help="Number of epochs between logging to tensorboard")
    parser.add_argument("--train-path", dest="train_path", help="Training data file")
    parser.add_argument("--valid-path", dest="valid_path", help="Validation data file")
    parser.add_argument("--test-path", dest="test_path", help="Testing data file")
    parser.add_argument("--checkpoint", dest="checkpoint", type=str, default="", help="Path to the .pth checkpoint file. Used to continue training from checkpoint")
    # Arguments for PTB dataset
    parser.add_argument("--is-stream", dest="is_stream", type=int, default=1, help="1: Treats data as a stream 0: Treats each line as a separate sentence")
    parser.add_argument("--bptt", dest="bptt", type=int, default=70, help="Length of backpropogation through time")
    # Arguments for attention model
    parser.add_argument("--second-order-size", dest="second_order_size", type=int, default=2, help="Number of cells for the attention model.")
    parser.add_argument("--lr-decay", dest="lr_decay", type=float, default=0.5, help="Factor by which the learning rate decays")
    parser.add_argument("--patience", dest="patience", type=int, default=3, help="Learning rate decay scheduler patience, number of epochs")
    # Arguments for generating texts
    # Number of lines to generate is set by the batch size
    parser.add_argument("--output-file", dest="output_file", type=str, default="generated.txt", help="File name to save the generated texts.")
    parser.add_argument("--sentence-length", dest="max_sentence_length", type=int, default=10, help="Max length of generated sentence")
    parser.add_argument("--generation-method", dest="generation_method", type=str, default="greedy", help="Select next words using greedy, random, or beam")
    # Arguments for LM syntax eval
    parser.add_argument("--stats-output-file", dest="stats_output_file", type=str, default="None", help="File name to save the statistics on generated sentence syntax.")

    args = parser.parse_args()
    return args


def train(model, vocab, train_dataset, val_dataset, args, device, logger=None):
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    log_every = args.log_every
    lr = args.lr
    lr_factor = args.lr_decay
    patience = args.patience
    is_stream = args.is_stream
    num_epochs = args.epochs
    save_to_log = logger is not None
    logdir = logger.get_logdir() if logger is not None else None

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=lr_factor, patience=patience)
    criterion = nn.NLLLoss(ignore_index=vocab.pad_id)

    # Load checkpoint if specified
    if args.checkpoint != "":
        model = load_checkpoint(args.checkpoint, model, device, optimizer)

    min_val_loss = None
    early_stopping_counter = 0
    # Limit step to wait for 2x lr decay patience.
    # After lr decay if the model still did not improve, stop it
    early_stopping_limit = 2 * patience
    
    for epoch in range(num_epochs):
        epoch_loss = []
        model.train()

        # Initialized as zeros, after first call to forward()
        # will be tuple(Tensor, Tensor), each Tensor (batch_size, hidden_size)
        init_state = model.init_lstm_state(device)
        for batch_iter, batch in enumerate(tqdm(train_dataset)):
            x, y = batch
            x = x.to(device)
            y = y.view(-1).to(device)

            optimizer.zero_grad()
            # When training on PTB, we want to preserve internal states between
            # batches in the epoch
            y_pred, ret_state = model(x, init_state)
            init_state = ret_state if is_stream else init_state
            # Criterion takes in y: (batch_size*seq_len) correct labels and 
            # y_pred: (batch_size*seq_len, vocab_size) softmax prob of vocabs
            loss = criterion(y_pred, y)
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.lstm.parameters(), 1)
            optimizer.step()
            epoch_loss.append(loss.item())
        
        # End of epoch, run validations
        model.eval()
        with torch.no_grad():
            epoch_average_loss = np.mean(epoch_loss)
            epoch_train_ppl = np.exp(epoch_average_loss)
            epoch_val_ppl, epoch_val_loss, epoch_val_wcpa, _ = \
                validate(model, criterion, val_dataset, is_stream, device)
            scheduler.step(epoch_val_loss)

            # Check for early stopping
            if min_val_loss is None or epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_limit:
                print("Early stopping after waiting {} epochs".format(early_stopping_limit))
                break

            # Add to logger on tensorboard at the end of an epoch
            if save_to_log:
                logger.scalar_summary("epoch_training_loss", epoch_average_loss, epoch)
                logger.scalar_summary("epoch_train_ppl", epoch_train_ppl, epoch)
                logger.scalar_summary("epoch_val_loss", epoch_val_loss, epoch)
                logger.scalar_summary("epoch_val_ppl", epoch_val_ppl, epoch)
                logger.scalar_summary("epoch_val_wcpa", epoch_val_wcpa, epoch)
                # Save epoch checkpoint
                if epoch % log_every == 0:
                    save_checkpoint(logdir, model, optimizer, epoch, epoch_average_loss, lr)
                # Save best validation checkpoint
                if epoch_val_loss == min_val_loss:
                    save_checkpoint(logdir, model, optimizer, epoch, epoch_average_loss, lr, "val_ppl")

            print('Epoch {} | Train Loss: {} | Val Loss: {} | Train PPL: {} | Val PPL: {} | Val WCPA: {}' \
                .format(epoch + 1, epoch_average_loss, epoch_val_loss, epoch_train_ppl, epoch_val_ppl, epoch_val_wcpa))

    print('Model trained.')


def test(checkpoint, model, vocab, test_dataset, args, device):
    batch_size = args.batch_size

    stats_output_file = None
    if args.stats_output_file != "None":
        dirname, filename = os.path.split(checkpoint)
        stats_output_file = os.path.abspath(os.path.join(dirname, "..", args.stats_output_file))
    # load model from checkpoint
    model = load_checkpoint(checkpoint, model, device)
    model.eval()

    # initialize criterion 
    criterion = nn.NLLLoss(ignore_index=vocab.pad_id)
    with torch.no_grad():
        test_ppl, test_loss, test_wcpa, graph_data = validate(
            model, criterion, test_dataset, args.is_stream, device, vocab, stats_output_file)

    # plot ldpa by distance
    if not args.is_stream and args.stats_output_file == "None":
        dirname, filename = os.path.split(checkpoint)
        save_path = os.path.abspath(os.path.join(dirname, 
            "test_ldpa_{}.png".format(os.path.splitext(filename)[0])))
        plot_ldpa(graph_data, save_path=save_path)

    print('Test Loss: {} | Test PPL: {} | Test WCPA: {}' \
        .format(test_loss, test_ppl, test_wcpa))


def generate_texts(checkpoint, model, vocab, args, device):
    dirname, filename = os.path.split(checkpoint)
    save_file = os.path.abspath(os.path.join(dirname, "..", args.output_file))
    model = load_checkpoint(checkpoint, model, device)
    model.eval()
    with torch.no_grad():
        sents = generate_sentences(model, vocab, args.batch_size, args.max_sentence_length, \
            args.is_stream, args.generation_method, device)
    print("Saving generated text to file...")
    with open(save_file, "w") as f:
        for line in sents:
            f.write(line + "\n")
    print("File saved to: ", save_file)
        

def main():
    # setup
    print("Setting up...")
    args = argParser()
    args.is_stream = True if args.is_stream == 1 else False
    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else "cpu")
    unique_logdir = create_unique_logdir(args.log, args.lr)
    logger = Logger(unique_logdir) if args.log != '' else None
    print("Using device: ", device)
    print("All training logs will be saved to: ", unique_logdir)
    print("Will log to tensorboard: ", logger is not None)

    # build dataset object
    print("Creating Dataset...")
    train_dataset = CustomDataset(args.train_path, args.batch_size, args.bptt, is_stream=args.is_stream)
    train_json_path = train_dataset.get_json_path()
    val_dataset = CustomDataset(args.valid_path, args.batch_size, args.bptt, is_stream=args.is_stream, json_path_override=train_json_path)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    kwargs = vars(args) # Turns args into a dictionary
    params = kwargs.copy()
    vocab = train_dataset.get_vocab()
    kwargs["vocab"] = vocab
    kwargs["temp_decay_interval"] = len(train_dataloader)

    if args.mode == 'test':
        test_dataset = CustomDataset(args.test_path, args.batch_size, args.bptt, is_stream=args.is_stream, json_path_override=train_json_path)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print("Done!")

    # build model
    model = ModelChooser(args.model, **kwargs)
    model = model.to(device)

    if args.mode == 'train':
        # train model
        print("Starting training...")
        # Save all params used to train
        json.dump(params, open(os.path.join(unique_logdir, "params.json"), 'w'), indent=2)
        train(model, vocab, train_dataloader, val_dataloader, args, device, logger=logger)
    elif args.mode == 'test':
        print("Starting testing...")
        test(args.checkpoint, model, vocab, test_dataloader, args, device)
    elif args.mode == 'generate':
        print("Starting text generation...")
        generate_texts(args.checkpoint, model, vocab, args, device)


if __name__ == "__main__":
    main()
