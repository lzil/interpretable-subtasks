import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
import pickle
import pdb
import random
import copy
import sys

from utils import Bunch, load_rb, update_config
from helpers import get_activation

# for easy rng manipulation
class TorchSeed:
    def __init__(self, seed):
        self.seed = seed
    def __enter__(self):
        self.rng_pt = torch.get_rng_state()
        torch.manual_seed(self.seed)
    def __exit__(self, type, value, traceback):
        torch.set_rng_state(self.rng_pt)


DEFAULT_ARGS = {
    'N': 100,    # recurrent units
    'S': 10,     # skill units
    'Z': 10,     # output units
    'D': 5,

    'rnn_burn_steps': 100,
    'rnn_noise': 0,

    'ff_bias': False,
    'rnn_bias': False,
    'out_act': 'none',
    'rnn_fb': True,

    'model_path': None,
    'rnn_path': None,
    'network_seed': None,
    'rnn_seed': None,
    'rnn_x_seed': None
}

class TwoStageRNN(nn.Module):
    def __init__(self, args=DEFAULT_ARGS):
        super().__init__()
        self.args = update_config(DEFAULT_ARGS, args)

        self.tau_x = 5
        self.rnn_activation = torch.tanh

        self._init_vars()
        self.reset()

    def _init_vars(self):
        # if self.args.network_seed is None:
        #     self.args.network_seed = np.random.randint(1e6)

        self.stage1_params = {
            'N': 100,
            'S': 22,
            'Z': 5,
            'tau': 10
        }
        self.stage1 = RNN(self.stage1_params)
        self.stage2_params = {
            'N': 200,
            'S': 5,
            'Z': 3,
            'tau': 5
        }
        self.stage2 = RNN(self.stage2_params)

        if self.args.model_path is not None:
            state_dict = torch.load(self.args.model_path)
            self.load_state_dict(state_dict)

    def forward(self, inp=None, extras=False):

        if extras:
            v, etc1 = self.stage1(inp, extras=True)
            z, etc2 = self.stage2(v, extras=True)
            etc = {'u': inp.detach(), 'v': v.detach(), 'x1': etc1['x'], 'x2': etc2['x']}
            return z, etc
        else:
            v = self.stage1(inp, extras=False)
            z = self.stage2(v, extras=False)
            return z

    def reset(self, *args, **kwargs):
        self.stage1.reset(*args, **kwargs)
        self.stage2.reset(*args, **kwargs)


DEFAULT_ARGS_RNN = {
    'N': 100,
    'S': 1,
    'Z': 1,
    'rnn_bias': False,
    'rnn_noise': 0,
    'rnn_burn_steps': 100,
    'out_act': 'none',
    'rnn_path': None,
    'rnn_seed': None,
    'rnn_x_seed': None
}

class RNN(nn.Module):
    def __init__(self, args=DEFAULT_ARGS_RNN):
        super().__init__()
        self.args = update_config(DEFAULT_ARGS_RNN, args)

        self.tau_x = self.args.tau
        self.rnn_activation = torch.tanh
        self.out_act = get_activation(self.args.out_act)

        self._init_vars()
        self.reset()

    def _init_vars(self):
        # seeds for each component of the network
        if self.args.rnn_seed is None:
            self.args.rnn_seed = np.random.randint(1e6)
        if self.args.rnn_x_seed is None:
            self.args.rnn_x_seed = np.random.randint(1e6)

        # layers use the same seed because no resevoirs here
        with TorchSeed(self.args.rnn_seed):
            self.W_ri = nn.Linear(self.args.S, self.args.N, bias=False)
            self.W_ro = nn.Linear(self.args.N, self.args.Z, bias=False)
            self.J = nn.Linear(self.args.N, self.args.N, bias=self.args.rnn_bias)
            torch.nn.init.normal_(self.J.weight.data, std=1.5/np.sqrt(self.args.N))

        # if paths exist, load them. have to do this after layers are defined
        # if self.args.model_path is not None:
        #     state_dict = torch.load(self.args.model_path)
        #     if state_dict['W_ri.weight'].shape[1] != self.W_ri.weight.shape[1]:
        #         # we are loading from W_ri with differently sized inputs
        #         old_S = state_dict['W_ri.weight'].shape[1]
        #         new_W_ri = torch.zeros_like(self.W_ri.weight)
        #         new_W_ri[:,:old_S] = state_dict['W_ri.weight']
        #         state_dict['W_ri.weight'] = new_W_ri
        #     self.load_state_dict(state_dict)
        if self.args.rnn_path is not None:
            self.load_state_dict(torch.load(self.args.rnn_path))


    def forward(self, s=None, extras=False):
        # pass through the forward part
        # u is input
        u = torch.zeros(self.args.N)
        if s is not None:
            u = u + self.W_ri(s)

        # rnn forward
        a = self.rnn_activation(self.J(self.x) + u)
        # adding any inherent reservoir noise
        if self.args.rnn_noise > 0:
            a = a + torch.normal(torch.zeros_like(a), self.args.rnn_noise)
        delta_x = (-self.x + a) / self.tau_x
        self.x = self.x + delta_x

        z_pre = self.W_ro(self.x)
        self.z = self.out_act(z_pre)

        if extras:
            return self.z, {'u': u, 'x': self.x.detach(), 'z_pre': z_pre}
        else:
            return self.z

    def burn_in(self, steps):
        for i in range(steps):
            a = torch.tanh(self.J(self.x))
            delta_x = (-self.x + a) / self.tau_x
            self.x = self.x + delta_x
        self.x.detach_()

    def reset(self, rnn_state=None, burn_in=True, device=None):
        self.z = torch.zeros((1, self.args.Z))
        '''
        guide to rnn x states:
        self.x shouuld be size [self.args.batch_size, self.args.N]
        - None: load from self.args.rnn_x_seed, which is then loaded later
        - [np.ndarray]: convert array to tensor and load
        - [torch.Tensor]: load tensor
        - 'zero': load zero Tensor
        - 'random': load random values
        - [int]: load with that seed
        - else: ERROR
        '''
        if rnn_state is None:
            # load specified hidden state from seed
            rnn_state = self.args.rnn_x_seed

        if type(rnn_state) is np.ndarray:
            # load an actual particular hidden state
            # if there's an error here then highly possible that rnn_state has wrong form
            self.x = torch.as_tensor(rnn_state).float()
        elif type(rnn_state) is torch.Tensor:
            self.x = rnn_state
        elif rnn_state == 'zero':
            # reset to 0
            self.x = torch.zeros((1, self.args.N))
        elif rnn_state == 'random':
            # reset to totally random value without using reservoir seed
            self.x = torch.normal(0, 1, (1, self.args.N))
        elif type(rnn_state) is int:
            # if any seed set, set the net to that seed and burn in
            with TorchSeed(rnn_state):
                self.x = torch.normal(0, 1, (1, self.args.N))
        else:
            print('not any of these types, something went wrong')
            pdb.set_trace()

        if device is not None:
            self.x = self.x.to(device)

        if burn_in:
            self.burn_in(self.args.rnn_burn_steps)

