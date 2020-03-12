import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import torch
import scipy.linalg

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

    parser.add_argument("--checkpoint-path", dest="checkpoint_path", type=str, default="", help="Path to the folder containing checkpoints")
    parser.add_argument("--gpu", dest="gpu", type=str, default='0', help="The gpu number if there's more than one gpu")
    args = parser.parse_args()
    return args

def sort_checkpoints(checkpoint_path):
    """
    Get list of training checkpoints sorted by epoch
    @param checkpoint_path (str): path to checkpoints folder
    """
    checkpoints = os.listdir(checkpoint_path) 
    checkpoints.remove('best_val_ppl.pth')
    cp_epochs = [(cp, int(re.search("epoch[0-9]+", cp)[0][5:])) for cp in checkpoints]
    sorted_checkpoints = [cp[0] for cp in sorted(cp_epochs, key=lambda x: x[1])]
    return sorted_checkpoints

def plot_norms(norms, title, save_path):
    epochs, n_cells = norms.shape
    x = range(epochs)
    fig, ax = plt.subplots(dpi=150)
    ax.set_ylabel('Norm')
    ax.set_xlabel('Epochs')
    ax.set_title(title, fontsize=12)
    for i in range(n_cells):
        ax.plot(x, norms[:,i], c = 'blue')
    plt.savefig(save_path)

def plot_angles(angles, title, save_path):
    n_pairs, n_angles = angles.shape
    x = range(n_angles)
    fig, ax = plt.subplots(dpi=150)
    ax.set_ylabel('Degrees')
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, 90)
    for i in range(n_pairs):
        ax.scatter(x, angles[i], c = 'blue')
    plt.savefig(save_path)


def compute_norms(checkpoint_path, weight_type, norm, device):
    """
    Compute the norms of the pairwise differences (W1 - W2) of LSTM cell
    weight matrices

    @param checkpoint_path (str): path to checkpoint folder
    @param weight_type (str): weight_ih, weight_hh, bias_ih, bias_hh
    @param norm: order for np.linalg.norm, one of 1, 2, np.inf, 'fro'
    @return norms (np.ndarray): matrix of size (epochs, # pairwise comparisons)
    """
    # get sorted checkpoints
    sorted_checkpoints = sort_checkpoints(checkpoint_path)
    # initialize storage
    norms = []
    # loop over checkpoints
    for idx, cp in enumerate(sorted_checkpoints):
        model = torch.load(os.path.join(checkpoint_path, cp), map_location=device)
        state_dict = model['model_state_dict']

        # extract names of LSTM weight matrices
        cell_weights = [key for key in state_dict.keys() if weight_type in key]

        # calculate norms of difference
        cell_norms = [np.linalg.norm(state_dict[w1] - state_dict[w2], norm) 
            for i, w1 in enumerate(cell_weights)
            for j, w2 in enumerate(cell_weights)
            if i < j]

        norms.append(cell_norms)

    norms = np.array(norms)

    # plot path of norms over epochs of training
    norm_names = {1: "L1", 2: "L2", np.inf: "Infinity", 'fro': "Frobenius"}
    weight_names = {
        'weight_ih': "Input to Hidden Weight", 
        'weight_hh': "Hidden to Hidden Weight", 
        'bias_ih': "Input to Hidden Bias", 
        'bias_hh': "Hidden to Hidden Bias"
    }
    title = "Path of the {} Norm of LSTM Cell {}".format(
        norm_names.get(norm), weight_names.get(weight_type))
    save_path = os.path.join(checkpoint_path, "..", "{}_{}.png".format(
        norm_names.get(norm), weight_type))

    plot_norms(norms, title, save_path)
    return norms

def compute_PABS(checkpoint_path, weight_type, device):
    """
    Compute the Principal Angles Between Subspaces (PABS) for pairwise 
    combinations of LSTM Cell weight matrices, and plot them

    @param checkpoint_path (str): path to checkpoint folder
    @param weight_type (str): weight_ih, weight_hh
    @return 
    """
    # only compute PABS for the last checkpoint
    last_checkpoint = sort_checkpoints(checkpoint_path)[-1]

    # load the checkpoint and extract the state dictionary 
    model = torch.load(os.path.join(checkpoint_path, last_checkpoint), map_location=device)
    state_dict = model['model_state_dict']

    # extract names of LSTM weight matrices that match weight_type
    cell_weights = [key for key in state_dict.keys() if weight_type in key]

    # Compute the subspace angles between the column spaces of weight matrices 
    # W1 and W2, in descending order
    pabs_radians = [scipy.linalg.subspace_angles(state_dict[w1],state_dict[w2]) 
        for i, w1 in enumerate(cell_weights)
        for j, w2 in enumerate(cell_weights)
        if i < j]
    
    # convert radians to degrees
    pabs_degrees = np.rad2deg(np.array(pabs_radians))

    # prepare title and path for plotting
    weight_names = {
        'weight_ih': "Input to Hidden Weight", 
        'weight_hh': "Hidden to Hidden Weight", 
        'bias_ih': "Input to Hidden Bias", 
        'bias_hh': "Hidden to Hidden Bias"
    }
    title = "Principal Angles Between Subspaces (PABS) \nLSTM Cell {}".format(
        weight_names.get(weight_type))
    save_path = os.path.join(checkpoint_path, "..", "PABS_{}.png".format(weight_type))
    plot_angles(pabs_degrees, title, save_path)
    return pabs_degrees


def main():
    print("Setting up...")
    args = argParser()
    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else "cpu")

    print("Computing matrix norms...")
    compute_norms(args.checkpoint_path, weight_type="weight_ih", norm='fro', device=device)
    compute_norms(args.checkpoint_path, weight_type="weight_hh", norm='fro', device=device)

    print("Computing matrix angles...")
    compute_PABS(args.checkpoint_path, weight_type="weight_ih", device=device)
    compute_PABS(args.checkpoint_path, weight_type="weight_hh", device=device)
    print("Done!")
    
if __name__ == "__main__":
    main()
