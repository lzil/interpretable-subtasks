import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import matplotlib.cm as cm

import random
import pickle
import argparse
import pdb
import json

from helpers import sigmoid
from utils import load_rb, get_config, update_config
from testers import load_model_path, test_model

from skills import SkillTask

import fig_format

# for plotting some instances of a trained model on a specified dataset

parser = argparse.ArgumentParser()
parser.add_argument('model', help='path to a model file, to be loaded into pytorch')
# parser.add_argument('-Z', type=int, help='output dimension')
parser.add_argument('-d', '--dataset', nargs='+', help='path to a dataset of trials')
# parser.add_argument('--noise', default=0, help='noise to add to trained weights')
# parser.add_argument('-r', '--res_noise', default=None, type=float)
# parser.add_argument('-m', '--m_noise', default=None, type=float)
# parser.add_argument('--x_noise', default=None, type=float)
# parser.add_argument('-x', '--reservoir_x_init', default=None, type=str)
parser.add_argument('-a', '--test_all', action='store_true')
parser.add_argument('-n', '--no_plot', action='store_true')
parser.add_argument('-c', '--config', default=None, help='path to config file if custom')

parser.add_argument('--no_rnn_fb', action='store_true')
args = parser.parse_args()


if args.config is None:
    config = get_config(args.model, ctype='model')
else:
    config = json.load(open(args.config, 'r'))
config = update_config(config, args, use_none=False)
dsets = config.dataset

# pdb.set_trace()
net = load_model_path(args.model, config=config)
# assuming config is in the same folder as the model

if args.no_rnn_fb:
    net.args.rnn_fb = False

if args.test_all:
    loss = np.mean(test_model(net, config)['losses'])
    print(f'avg summed loss (all): {loss}')

if not args.no_plot:
    data = test_model(net, config, n_tests=12)
    # pdb.set_trace()
    print(f'avg loss: {np.mean(data["losses"])}')

    run_id = '/'.join(args.model.split('/')[-3:-1])

    # pdb.set_trace()

    # data = data[0]
    # pdb.set_trace()

    fig, axes = plt.subplots(3,4,sharex=False, sharey=False, figsize=(12,8))
    for i, ax in enumerate(fig.axes):
        ix = data['ixs'][i]
        trial = data['trials'][i]
        x = data['x'][i]
        y = data['y'][i]
        z = data['outs'][i]
        loss = data['losses'][i]

        xr = np.arange(x.shape[-1])

        ax.axvline(x=0, color='dimgray', alpha = 1)
        ax.axhline(y=0, color='dimgray', alpha = 1)
        ax.grid(True, which='major', lw=1, color='lightgray', alpha=0.4)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        fig_format.hide_frame(ax)

        config.S = net.args.S
        config.Z = net.args.Z
        colors = cm.cool(np.linspace(0, 1, config.S))
        for j in range(0, config.S):
            ax.plot(xr, x[j], color=colors[j], lw=1, ls='-', alpha=.5, label='x')
        for j in range(0, config.Z):
            ax.plot(xr, y[j], color=colors[j], lw=1, ls='--', alpha=.5, label='y')
            ax.plot(xr, z[j], color=colors[j], lw=2, ls='-', alpha=1, label='z')

        ax.tick_params(axis='both', color='white', labelsize=8)
        ax.set_title(f'trial {ix}: {trial.s_ids}, loss {np.round(float(loss), 2)}', size=8)


    fig.text(0.5, 0.04, 'timestep', ha='center', va='center', size=12)
    fig.text(0.03, 0.5, 'value', ha='center', va='center', rotation='vertical', size=14)

    # handles, labels = axes.get_legend_handles_labels()
    fig.suptitle(f'Final performance: {run_id}', size=14)
    # fig.legend(handles, labels, loc='lower right')

    plt.tight_layout(rect=(.04, .06, 1, 0.95))

    plt.show()


