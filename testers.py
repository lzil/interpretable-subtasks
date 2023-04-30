import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import fig_format

import random
import os
import pdb
import json
import sys

from network import TwoStageRNN
from utils import Bunch, load_rb, get_config

from helpers import get_mse_loss, get_v_loss, create_loaders


def load_model_path(path, config=None):
    if config is None:
        config = get_config(path)
    if type(config) is dict:
        config = Bunch(**config)
    config.model_path = path

    net = TwoStageRNN(config)

    net.eval()
    return net

# given a model and a dataset, see how well the model does on it
# works with plot_trained.py
def test_model(net, config, n_tests=128):
    test_set, test_loader = create_loaders(config.dataset, config, split_test=False, test_size=n_tests)
    trials = next(iter(test_loader))

    mse_loss_fn = get_mse_loss(config)
    v_loss_fn = get_v_loss(config)

    # pdb.set_trace()

    with torch.no_grad():
        idxs = [t.n for t in trials['trialobj']]
        x = trials['x']
        y = trials['y']

        # saving each individual loss per sample, per timestep
        losses = np.zeros(len(x))
        outs = []
        vs = []

        for j in range(x.shape[2]):
            # run the step
            net_in = x[:,:,j]
            net_out, etc = net(net_in, extras=True)
            # net_out = net(t=net_g, c=net_c)
            # net_out = net(net_in)
            outs.append(net_out)
            vs.append(etc['v2'])

        # pdb.set_trace()
        vs = torch.stack(vs, dim=2)
        outs = torch.stack(outs, dim=2)

        # pdb.set_trace()
        
        for k in range(len(x)):
            losses[k] += mse_loss_fn(outs[k], y[k], i=trials['trialobj'][k], t_ix=0, single=True).item()

        # pdb.set_trace()

        data = {
            'ixs': idxs,
            'trials': trials['trialobj'],
            'x': x,
            'y': y,
            'outs': outs,
            'losses': losses,
            'v': vs
        }

    return data

# returns hidden states as [N, T, H]
# note: this returns hidden states as the last dimension, not timesteps!
def get_states(net, x):
    states = []
    with torch.no_grad():
        net.reset()
        for j in range(x.shape[2]):
            net_in = x[:,:,j]
            net_out, extras = net(net_in, extras=True)
            states.append(extras['x'])

    A = torch.stack(states, dim=1)
    return A


if __name__ == '__main__':
    pass