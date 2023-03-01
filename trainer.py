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

# from network import BasicNetwork, Reservoir
from network import TwoStageRNN

from utils import log_training, load_rb, get_config, update_config
from helpers import get_optimizer, get_scheduler, get_criteria, create_loaders, collater

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')

        trains, tests = create_loaders(self.args.dataset, self.args, split_test=True, test_size=50)

        self.train_set, self.train_loader = trains
        self.test_set, self.test_loader = tests
        logging.info(f'Using dataset {self.args.dataset}')

        self.net = TwoStageRNN(self.args)

        # getting number of elements of every parameter
        self.n_params = {}
        self.train_params = []
        self.not_train_params = []
        logging.info('Training the following parameters:')
        for k,v in self.net.named_parameters():
            # k is name, v is weight
            found = False
            # filtering just for the parts that will be trained
            for part in self.args.train_parts:
                if part in k:
                    logging.info(f'  {k}')
                    self.n_params[k] = (v.shape, v.numel())
                    self.train_params.append(v)
                    found = True
                    break
            if not found:
                self.not_train_params.append(k)
        logging.info('Not training:')
        for k in self.not_train_params:
            logging.info(f'  {k}')

        self.criteria = get_criteria(self.args)
        self.optimizer = get_optimizer(self.args, self.train_params)
        # self.scheduler = get_scheduler(self.args, self.optimizer)

        # pdb.set_trace()
        
        self.log_interval = self.args.log_interval
        if not self.args.no_log:
            self.log = self.args.log
            self.run_id = self.args.log.run_id
            self.vis_samples = []
            self.csv_path = open(os.path.join(self.log.run_dir, f'losses_{self.run_id}.csv'), 'a')
            self.writer = csv.writer(self.csv_path, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            self.writer.writerow(['ix', 'train_loss', 'test_loss'])
            self.plot_checkpoint_path = os.path.join(self.log.run_dir, f'checkpoints_{self.run_id}.pkl')
            self.save_model_path = os.path.join(self.log.run_dir, f'model_{self.run_id}.pth')

    def log_model(self, ix=0, name=None):
        # if we want to save a particular name, just do it and leave
        if name is not None:
            model_path = os.path.join(self.log.run_dir, name)
            if os.path.exists(model_path):
                os.remove(model_path)
            torch.save(self.net.state_dict(), model_path)
            return
        # saving all checkpoints takes too much space so we just save one model at a time, unless we explicitly specify it
        if self.args.log_checkpoint_models:
            self.save_model_path = os.path.join(self.log.checkpoint_dir, f'model_{ix}.pth')
        elif os.path.exists(self.save_model_path):
            os.remove(self.save_model_path)
        torch.save(self.net.state_dict(), self.save_model_path)

    def log_checkpoint(self, ix, x, y, z, train_loss, test_loss):
        self.writer.writerow([ix, train_loss, test_loss])
        self.csv_path.flush()

        self.log_model(ix)

        # we can save individual samples at each checkpoint, that's not too bad space-wise
        if self.args.log_checkpoint_samples:
            self.vis_samples.append([ix, x, y, z, train_loss, test_loss])
            if os.path.exists(self.plot_checkpoint_path):
                os.remove(self.plot_checkpoint_path)
            with open(self.plot_checkpoint_path, 'wb') as f:
                pickle.dump(self.vis_samples, f)

    # runs an iteration where we want to match a certain trajectory
    def run_trial(self, x, y, trial, training=True, extras=False):
        self.net.reset(self.args.rnn_x_init, device=self.device)
        trial_loss = 0.
        k_loss = 0.
        outs = []
        us = []
        # setting up k for t-BPTT
        if training and self.args.k != 0:
            k = self.args.k
        else:
            # k to full n means normal BPTT
            k = x.shape[2]
        for j in range(x.shape[2]):
            net_s = x[:,:,j]
            net_out, etc = self.net(inp=net_s, extras=True)
            outs.append(net_out)
            us.append(etc['u'])
            # t-BPTT with parameter k
            if (j+1) % k == 0:
                # the first timestep with which to do BPTT
                k_outs = torch.stack(outs[-k:], dim=2)
                k_targets = y[:,:,j+1-k:j+1]
                for c in self.criteria:
                    k_loss += c(k_outs, k_targets, i=trial, t_ix=j+1-k)
                trial_loss += k_loss.detach().item()
                if training:
                    k_loss.backward()
                k_loss = 0.
                # TODO fix this
                self.net.stage1.x = self.net.stage1.x.detach()
                self.net.stage2.x = self.net.stage2.x.detach()

        trial_loss /= x.shape[0]

        if extras:
            net_us = torch.stack(us, dim=2)
            net_outs = torch.stack(outs, dim=2)
            etc = {
                'outs': net_outs,
                'us': net_us,
            }
            return trial_loss, etc
        return trial_loss

    def train_iteration(self, x, y, trial, ix_callback=None):
        self.optimizer.zero_grad()
        trial_loss, etc = self.run_trial(x, y, trial, extras=True)

        if ix_callback is not None:
            ix_callback(trial_loss, etc)
        self.optimizer.step()

        etc = {
            'ins': x,
            'goals': y,
            'us': etc['us'].detach(),
            'outs': etc['outs'].detach()
        }
        return trial_loss, etc

    def test(self):
        with torch.no_grad():
            # x, y, trials = next(iter(self.test_loader))
            trial = next(iter(self.test_loader))
            x = trial['x']
            y = trial['y']
            x, y = x.to(self.device), y.to(self.device)
            loss, etc = self.run_trial(x, y, trial['task'], training=False, extras=True)

            # pdb.set_trace()
        etc = {
            'ins': x,
            'goals': y,
            'us': etc['us'].detach(),
            'outs': etc['outs'].detach()
        }

        return loss, etc

    def train(self, ix_callback=None):
        ix = 0
        # for convergence testing
        running_min_error = float('inf')
        running_no_min = 0

        running_loss = 0.0
        ending = False

        for e in range(self.args.n_epochs):
            for epoch_idx, trial in enumerate(self.train_loader):
                ix += 1

                x = trial['x']
                y = trial['y']

                x, y = x.to(self.device), y.to(self.device)
                iter_loss, etc = self.train_iteration(x, y, trial['task'], ix_callback=ix_callback)

                if iter_loss == -1:
                    logging.info(f'iteration {ix}: is nan. ending')
                    ending = True
                    break

                running_loss += iter_loss

                if ix % self.log_interval == 0:
                    z = etc['outs'].cpu().numpy().squeeze()
                    train_loss = running_loss / self.log_interval
                    test_loss, test_etc = self.test()
                    log_arr = [
                        f'*{ix}',
                        f'train {train_loss:.3f}',
                        f'test {test_loss:.3f}'
                    ]

                    log_str = '\t| '.join(log_arr)
                    logging.info(log_str)

                    if not self.args.no_log:
                        self.log_checkpoint(ix, etc['ins'].cpu().numpy(), etc['goals'].cpu().numpy(), z, train_loss, test_loss)
                    running_loss = 0.0

                    # convergence based on no avg loss decrease after patience samples
                    if test_loss < running_min_error:
                        running_no_min = 0
                        running_min_error = test_loss
                        if not self.args.no_log:
                            self.log_model(name='model_best.pth')
                    else:
                        running_no_min += self.log_interval
                    if running_no_min > self.args.patience:
                        logging.info(f'iteration {ix}: no min for {self.args.patience} samples. ending')
                        ending = True
                if ending:
                    break
            logging.info(f'Finished dataset epoch {e+1}')
            # if self.scheduler is not None:
            #     self.scheduler.step()
            if ending:
                break

        if not self.args.no_log and self.args.log_checkpoint_samples:
            # for later visualization of outputs over timesteps
            with open(self.plot_checkpoint_path, 'wb') as f:
                pickle.dump(self.vis_samples, f)

            self.csv_path.close()

        logging.info(f'END | iterations: {(ix // self.log_interval) * self.log_interval} | best loss: {running_min_error}')
        return running_min_error, ix


