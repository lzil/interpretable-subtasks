
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import pdb

import random
from collections import OrderedDict

from utils import load_rb, get_config

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def get_optimizer(args, train_params):
    op = None
    if args.optimizer == 'adam':
        op = optim.Adam(train_params, lr=args.lr, weight_decay=args.l2_reg)
    elif args.optimizer == 'sgd':
        op = optim.SGD(train_params, lr=args.lr, weight_decay=args.l2_reg)
    elif args.optimizer == 'rmsprop':
        op = optim.RMSprop(train_params, lr=args.lr, weight_decay=args.l2_reg)
    return op

def get_scheduler(args, op):
    if args.s_rate is not None:
        return optim.lr_scheduler.MultiStepLR(op, milestones=[1,2,3], gamma=args.s_rate)
    return None


# simple dataset for skill learning
class TrialDataset(Dataset):
    def __init__(self, trials, args):
        self.args = args
        self.data = trials

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[ii] for ii in range(len(self))[idx]]

        tasktrial = self.data[idx]
        trial = {}
        trial['x'] = tasktrial.get_st_data(self.args.n_tasks, x_noise=self.args.x_noise)
        trial['y']  = tasktrial.get_target_data()
        trial['trialobj'] = tasktrial

        return trial

    def get_task(self, idx):
        return np.argmax(self.max_idxs > idx)


# turns data samples into stuff that can be run through network
def collater(samples):
    # data = list(zip(*samples))
    data = {key: [i[key] for i in samples] for key in samples[0]}
    # # pad xs and ys to be the length of the max-length example
    # # pdb.set_trace()
    max_len = np.max([x.shape[-1] for x in data['x']])
    xs_pad = [np.pad(x, ([0,0],[0,max_len-x.shape[-1]])) for x in data['x']]
    data['x'] = torch.as_tensor(np.stack(xs_pad), dtype=torch.float)
    ys_pad = [np.pad(y, ([0,0],[0,max_len-y.shape[-1]])) for y in data['y']]
    data['y'] = torch.as_tensor(np.stack(ys_pad), dtype=torch.float)

    # data['x'] = torch.as_sensor(x)
    return data

# creates datasets and dataloaders
def create_loaders(dataset, args, split_test=True, test_size=1):
    data_config = load_rb(dataset)
    dset = data_config['data']
    config = data_config['config']
    args.n_tasks = config['n_tasks']
    if split_test:
        # create both training and test sets
        cutoff = round(.9 * len(dset))
        train_set = TrialDataset(dset[:cutoff], args)
        test_set = TrialDataset(dset[cutoff:], args)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collater, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=test_size, shuffle=True, collate_fn=collater, drop_last=False)
        return (train_set, train_loader), (test_set, test_loader)
    else:
        # only one training set created with all the data, but it's the test set
        all_set = TrialDataset(dset, args)
        all_loader = DataLoader(all_set, batch_size=test_size, shuffle=True, collate_fn=collater, drop_last=True)
        return (all_set, all_loader)

def get_mse_loss(args):
    fn = nn.MSELoss(reduction='sum')
    # do this in a roundabout way due to truncated bptt
    def tbptt_mse(o, t, i, t_ix=None, single=False):
        # last dimension is number of timesteps
        # divide by batch size to avoid doing so logging and in test
        loss = 0.
        if single:
            o = o.unsqueeze(0)
            t = t.unsqueeze(0)
            i = [i]
        for j in range(len(t)):
            length = i[j].trial_len
            if t_ix + t.shape[-1] <= length:
                loss += fn(o[j], t[j])# / length
            elif t_ix < length:
                t_adj = t[j,:,:length-t_ix]
                o_adj = o[j,:,:length-t_ix]
                loss += fn(o_adj, t_adj)# / length # order matters for bce
        return args.lambda_mse * loss

    return tbptt_mse

def get_v_loss(args):
    def v_loss_fn(vs):
        loss = torch.sum(torch.abs(vs))
        return args.lambda_vl1 * loss
    return v_loss_fn


# def get_criteria(args):

#     criteria = []
#     for l in args.loss:
#         if l == 'mse':
#             fn = nn.MSELoss(reduction='sum')
#             # do this in a roundabout way due to truncated bptt
#             def tbptt_mse(o, t, i, t_ix=None, single=False):
#                 # last dimension is number of timesteps
#                 # divide by batch size to avoid doing so logging and in test
#                 loss = 0.
#                 if single:
#                     o = o.unsqueeze(0)
#                     t = t.unsqueeze(0)
#                     infos = [infos]
#                 for j in range(len(t)):
#                     length = infos[j].trial_len
#                     if t_ix + t.shape[-1] <= length:
#                         loss += fn(o[j], t[j])# / length
#                     elif t_ix < length:
#                         t_adj = t[j,:,:length-t_ix]
#                         o_adj = o[j,:,:length-t_ix]
#                         loss += fn(o_adj, t_adj)# / length # order matters for bce
#                 return args.lambda_mse * loss
#             criteria.append(tbptt_fn)
#         elif l == 'l1':
#             fn = 
#         else:
#             raise NotImplementedError
    
#     return criteria

def get_activation(name):
    if name == 'exp':
        fn = torch.exp
    elif name == 'relu':
        fn = nn.ReLU()
    elif name == 'sigmoid':
        fn = nn.Sigmoid()
    elif name == 'tanh':
        fn = nn.Tanh()
    elif name == 'none':
        fn = lambda x: x
    return fn

def get_dim(a):
    if hasattr(a, '__iter__'):
        return len(a)
    else:
        return 1

    # return l2 * total_loss
