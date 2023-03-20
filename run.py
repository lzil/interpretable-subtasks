import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.optimize import minimize

import os
import argparse
import pdb
import sys
import pickle
import logging
import random
import csv
import math
import json
import copy
import pandas as pd


from utils import log_training, load_rb, get_config, update_config, load_config
from helpers import get_optimizer, get_scheduler, get_criteria, create_loaders

from skills import TaskTrial

from trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-N', type=int, default=300, help='number of neurons in reservoir')

    parser.add_argument('--train_parts', type=str, nargs='+', default=[''])
    parser.add_argument('-c', '--config', type=str, default=None, help='use args from config file')
    
    # make sure model_config path is specified if you use any paths! it ensures correct dimensions, bias, etc.
    parser.add_argument('--model_config_path', type=str, default=None, help='config path corresponding to model load path')
    parser.add_argument('--model_path', type=str, default=None, help='start training from certain model. superseded by below')
    parser.add_argument('--rnn_path', type=str, default=None, help='start training from certain reservoir representation')
    
    # network arguments
    parser.add_argument('--x_noise', type=float, default=0, help='up/downnoise in the input')
    parser.add_argument('--rnn_noise', type=float, default=0, help='noise in RNN operations')
    parser.add_argument('--out_act', type=str, default='none', help='output activation at the very end of the network')
    parser.add_argument('--rnn_bias', action='store_true', help='bias term as part of recurrent connections, with J')
    parser.add_argument('--no_ff_bias', action='store_false', dest='ff_bias', help='bias in readout weights')
    parser.add_argument('--no_fb', action='store_false', dest='rnn_fb', help='feedback from network output to input')

    # dataset arguments
    parser.add_argument('-d', '--dataset', type=str, default='datasets/debug_l0.pkl', help='dataset to use')
    parser.add_argument('--same_test', action='store_true', help='use entire dataset for both training and testing')
    parser.add_argument('--full_len', action='store_true', help='use full length of command in training')
    
    # training arguments
    parser.add_argument('--optimizer', choices=['adam', 'sgd', 'rmsprop'], default='adam')
    parser.add_argument('--k', type=int, default=0, help='k for t-bptt. use 0 for full bptt')

    # adam parameters
    parser.add_argument('--batch_size', type=int, default=1, help='size of minibatch used')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate. adam only')
    parser.add_argument('--n_epochs', type=int, default=40, help='number of epochs to train for. adam only')
    parser.add_argument('--conv_type', type=str, choices=['patience', 'grad'], default='patience', help='how to determine convergence. adam only')
    parser.add_argument('--patience', type=int, default=4000, help='stop training if loss doesn\'t decrease. adam only')
    parser.add_argument('--l2_reg', type=float, default=0, help='amount of l2 regularization')
    # parser.add_argument('--s_rate', default=None, type=float, help='scheduler rate. dont use for no scheduler')
    parser.add_argument('--loss', type=str, nargs='+', default=['mse'], choices=['mse', 'bce', 'l1'])

    # adam lambdas
    parser.add_argument('--l1', type=float, default=1, help='weight of normal loss')
    parser.add_argument('--lambda2', type=float, default=1, help='weight of l1 loss for layer v')

    # seeds
    parser.add_argument('--seed', type=int, help='general purpose seed')
    parser.add_argument('--network_seed', type=int, help='seed for network initialization')
    parser.add_argument('--rnn_seed', type=int, help='seed for rnn')
    parser.add_argument('--rnn_x_seed', type=int, default=0, help='seed for rnn init hidden states')
    parser.add_argument('--rnn_burn_steps', type=int, default=200, help='number of steps for rnn to burn in')

    parser.add_argument('-x', '--rnn_x_init', type=str, default=None, help='other seed options for rnn')

    # control logging
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--log_checkpoint_models', action='store_true')
    parser.add_argument('--log_checkpoint_samples', action='store_true')

    parser.add_argument('-n', '--name', type=str, default='debug')
    parser.add_argument('--slurm_param_path', type=str, default=None)
    parser.add_argument('--slurm_id', type=int, default=None)
    # parser.add_argument('--use_cuda', action='store_true')

    args = parser.parse_args()

    # print(parser._actions)
    # print([arg.startswith(option) for arg in sys.argv[1:] for a in parser._actions for option in a.option_strings ])
    # print(list(filter(lambda arg: any([arg.startswith(option) for a in parser._actions for option in a.option_strings]), sys.argv[1:])))

    # pdb.set_trace()
    return args

def adjust_args(args):

    # don't use logging.info before we initialize the logger!! or else stuff is gonna fail

    # dealing with slurm. do this first!! before anything else
    # needs to be before seed setting, so we can set it
    if args.slurm_id is not None:
        from parameters import apply_parameters
        args = apply_parameters(args.slurm_param_path, args)

    # loading from a config file
    # try not to specify any new arguments because they will be overridden
    # might later try to do fancy argparse magic to be able to specify new arguments
    if args.config is not None:
        config = load_config(args.config)
        args = update_config(args, config)
    dset_config = get_config(args.dataset, ctype='dset')

    if args.loss == 'bce':
        args.out_act = 'sigmoid'

    # setting seeds
    if args.rnn_seed is None:
        args.rnn_seed = np.random.randint(1e6)
    if args.seed is None:
        args.seed = np.random.randint(1e6)
    if args.network_seed is None:
        args.network_seed = np.random.randint(1e6)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # initializing logging
    # do this last, because we will be logging previous parameters into the config file
    if not args.no_log:
        if args.slurm_id is not None:
            log = log_training(log_dir='logs', config=args, log_name = os.path.join(args.name.split('_')[0], args.name.split('_')[1]), checkpoints=args.log_checkpoint_models)
        else:
            log = log_training(log_dir='logs', config=args, log_name = args.name, checkpoints=args.log_checkpoint_models)

        logging.basicConfig(format='%(message)s', filename=log.run_log, level=logging.DEBUG)
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        logging.getLogger('').addHandler(console)
        args.log = log
    else:
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)
        logging.info('NOT LOGGING THIS RUN.')

    # logging, when loading models from paths
    if args.model_path is not None:
        logging.info(f'Using model path {args.model_path}')

    return args


if __name__ == '__main__':
    args = parse_args()
    args = adjust_args(args)

    trainer = Trainer(args)
    logging.info(f'Initialized trainer. Using device {trainer.device}, optimizer {args.optimizer}.')

    if args.optimizer == 'lbfgs':
        best_loss, n_iters = trainer.optimize_lbfgs()
    elif args.optimizer in ['sgd', 'rmsprop', 'adam']:
        best_loss, n_iters = trainer.train()

    # if args.slurm_id is not None:
    #     # if running many jobs, then we gonna put the results into a csv
    #     csv_path = os.path.join('logs', args.name.split('_')[0] + '.csv')
    #     csv_exists = os.path.exists(csv_path)
    #     with open(csv_path, 'a') as f:
    #         writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #         labels_csv = ['slurm_id', 'N', 'D1', 'D2', 'seed', 'rseed', 'fp', 'fb', 'mnoise', 'rnoise', 'dset', 'niter', 'tparts', 'loss']
    #         vals_csv = [
    #             args.slurm_id, args.N, args.D1, args.D2, args.seed,
    #             args.res_seed, args.fixed_pts, args.fixed_beta, args.m_noise, args.res_noise,
    #             args.dataset, n_iters, '-'.join(args.train_parts), best_loss
    #         ]
    #         if args.optimizer != 'lbfgs':
    #             labels_csv.extend(['lr', 'epochs'])
    #             vals_csv.extend([args.lr, args.n_epochs])

    #         if not csv_exists:
    #             writer.writerow(labels_csv)
    #         writer.writerow(vals_csv)

    logging.shutdown()

