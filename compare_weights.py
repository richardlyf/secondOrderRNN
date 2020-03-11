import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import torch

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

def plot_norms(norms, title, save_path):
    epochs, n_cells = norms.shape
    x = range(epochs)
    fig, ax = plt.subplots(dpi=150)
    ax.set_ylabel('Norm')
    ax.set_xlabel('Epochs')
    ax.set_title(title, fontsize=12)
    for i in range(n_cells):
        ax.plot(x, norms[:,i])
    plt.savefig(save_path)

def distances(checkpoint_path, weight_type, norm, device):
    """
    @param weight_type (str): weight_ih, weight_hh, bias_ih, bias_hh
    @norm: order for np.linalg.norm, one of 1, 2, np.inf, 'fro'
    """
    # get list of checkpoints sorted by epoch
    checkpoints = os.listdir(checkpoint_path) 
    checkpoints.remove('best_val_ppl.pth')
    cp_epochs = [(cp, int(re.search("epoch[0-9]+", cp)[0][5:])) for cp in checkpoints]
    sorted_checkpoints = [cp[0] for cp in sorted(cp_epochs, key=lambda x: x[1])]

    # initialize storage
    norms = []
    # loop over checkpoints
    for idx, cp in enumerate(sorted_checkpoints):
        model = torch.load(os.path.join(checkpoint_path, cp), map_location=device)
        state_dict = model['model_state_dict']

        # extract names of LSTM weight matrices
        cell_weights = [key for key in state_dict.keys() if weight_type in key]

        # calculate norms
        cell_norms = [np.linalg.norm(state_dict[W], norm) for W in cell_weights]

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
    save_path = os.path.join(checkpoint_path, "..", "normpath_{}_{}.png".format(
        norm_names.get(norm), weight_type))

    plot_norms(norms, title, save_path)
    return norms


def main():
    print("Setting up...")
    args = argParser()
    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else "cpu")

    print("Computing matrix norms...")
    distances(args.checkpoint_path, weight_type="weight_ih", norm=1, device=device)
    distances(args.checkpoint_path, weight_type="weight_hh", norm=2, device=device)
    distances(args.checkpoint_path, weight_type="weight_hh", norm=np.inf, device=device)
    distances(args.checkpoint_path, weight_type="weight_hh", norm='fro', device=device)
    print("Done!")
    
if __name__ == "__main__":
    main()
