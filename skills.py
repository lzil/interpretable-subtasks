import numpy as np
from scipy.stats import norm
# from sklearn.gaussian_process import GaussianProcessRegressor as gpr
# from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import pickle
import os
import sys
import json
import pdb
import random
import pandas as pd
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as colors

import fig_format

import argparse

# from motifs import gen_fn_motifs
from utils import update_config, load_config, load_rb, Bunch

eps = 1e-6

mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['lines.linewidth'] = .5

cols = ['coral', 'cornflowerblue', 'magenta', 'orchid']
cspaces = [cm.spring, cm.summer, cm.autumn, cm.winter]

DEFAULT_S_MAP = dict([(i, i) for i in range(10)])

# dset_id is the name of the dataset (as saved)
# n is the index of the trial in the dataset
class TaskTrial:
    def __init__(self, task_id, skill_arr, sensory_data, target_data, dset_id=None, n=None):
        # self.trial_len = trial_len
        self.dset_id = dset_id
        self.n = n

        self.task_id = task_id
        self.s_ids = skill_arr[0]
        self.s_starts = skill_arr[1]
        self.s_ends = skill_arr[2]
        self.sensory_data = sensory_data
        self.target_data = target_data

        self.sensory_dim = sensory_data.shape[0]
        self.trial_len = sensory_data.shape[1]

        # assert len(s_ids) == len(s_starts)
        # assert len(s_starts) == len(s_ends)


    def get_x(self, task_count, x_noise=0):
        x = np.zeros((self.sensory_dim+task_count, self.trial_len))
        x[self.sensory_dim+self.task_id, self.s_starts[0]:self.s_ends[-1]] = 1
        x[:self.sensory_dim] = self.sensory_data
        # noisy up/down corruption
        if x_noise != 0:
            x = corrupt_x(x, x_noise)
        return x

    def get_y(self):
        return self.target_data


# add noise to x
def corrupt_x(x, x_noise):
    x += np.random.normal(scale=x_noise, size=x.shape)
    return x

# sample a length for a skill
def draw_skill_length(s_id):
    if s_id == 0:
        # copy sensory input to output
        trial_len = 10 + np.random.exponential(10)
    elif s_id == 1:
        # story sensory input
        trial_len = 10 + np.random.exponential(10)
    elif s_id == 2:
        # output stored memory
        trial_len = 10 + np.random.exponential(10)
    elif s_id == 3:
        # flip memory
        trial_len = 10
    elif s_id == 4:
        # press binary button
        trial_len = 10 + np.random.exponential(10)
    return trial_len

# create a task with spaced intervals and skill lengths
def create_task(s_ids, args):
    s_starts = []
    s_ends = []
    s_start = 0
    s_end = 0

    # we can have up to n_skills in this sequence, but might not hit that limit
    for i in range(len(s_ids)):
        s_id = s_ids[i]
        # if not the first skill, add some interval in between
        if i != 0:
            # s_start = s_end + int(np.random.exponential(args.scale))
            s_start = s_end + int(np.random.exponential(args.scale))
        s_end = s_start + int(draw_skill_length(s_id))
        s_starts.append(s_start)
        s_ends.append(s_end)

        # check if ending of this skill is too late
        # if so, return None to demonstrate task creation failed
        if s_end >= args.trial_len:
            return None

    s_ids = [int(x) for x in s_ids]
    s_starts = [int(x) for x in s_starts]
    s_ends = [int(x) for x in s_ends]

    skill_arr = (s_ids, s_starts, s_ends)
    return skill_arr

# adds sensory and target data to a task
def create_tasktrial(task_id, skill_arr, trial_len):
    s_ids, s_starts, s_ends = skill_arr

    task_len = s_ends[-1]
    s_start_new = np.random.randint(0, trial_len - task_len)
    s_starts = [x + s_start_new for x in s_starts]
    s_ends = [x + s_start_new for x in s_ends]

    new_skill_arr = [s_ids, s_starts, s_ends]

    sensory_data = np.zeros((2, trial_len))
    target_data = np.zeros((3, trial_len))

    cur_mem = [0, 0]
    for i in range(len(s_ids)):
        s_id = s_ids[i]
        s_start = s_starts[i]
        s_end = s_ends[i]

        angle = np.random.uniform(2*np.pi)
        sin = np.sin(angle)
        cos = np.cos(angle)
        if s_id == 0 or s_id == 1:
            sensory_data[0, s_start:s_end] = sin
            sensory_data[1, s_start:s_end] = cos

            if s_id == 0:
                target_data[0, s_start:s_end] = sin
                target_data[1, s_start:s_end] = cos

            if s_id == 1:
                # remember the input
                cur_mem = [sin, cos]
        
        if s_id == 2:
            target_data[0, s_start:s_end] = cur_mem[0]
            target_data[1, s_start:s_end] = cur_mem[1]

        if s_id == 3:
            cur_mem = [-cur_mem[0], -cur_mem[1]]

        if s_id == 4:
            target_data[2, s_start:s_end] = 1

    trial = TaskTrial(task_id, new_skill_arr, sensory_data, target_data)
    return trial


# load task json and assert that it's valid
def load_tasks(path):
    tasks = json.load(open(path, 'r'))
    # tasks is just a list of s_ids
    return tasks

def create_dataset(args):
    if args.seed is None:
        args.seed = int(str(time.time())[-5:])
    np.random.seed(args.seed)

    trial_len = args.trial_len

    if args.tasks is not None:
        s_ids_list = load_tasks(args.tasks)

    task_arr = []
    for i in range(len(s_ids_list)):
        # create the task
        task = None
        while task is None:
            task = create_task(s_ids_list[i], args)
        task_arr.append(task)

    trials = []
    for j in range(args.n_trials):
        # fetch a task id
        task_id = np.random.randint(0, len(s_ids_list))
        # create a trial based on the task
        trial = create_tasktrial(task_id, task_arr[task_id], args.trial_len)
        trials.append(trial)

    return trials, args

# turn task_args argument into usable argument variables
# lots of defaults are written down here
def get_task_args(args):
    tarr = args.task_args
    targs = Bunch()

    if args.mode == 'create':
        targs.trial_len = get_tval(tarr, 'len', 150, int)
        targs.scale = get_tval(tarr, 'scale', 10, int)
        targs.n_trials = get_tval(tarr, 'n', 4000, int)

    return targs

# get particular value(s) given name and casting type
def get_tval(targs, name, default, dtype, n_vals=1):
    if name in targs:
        # set parameter(s) if set in command line
        idx = targs.index(name)
        if n_vals == 1: # one value to set
            val = dtype(targs[idx + 1])
        else: # multiple values to set
            vals = []
            for i in range(1, n_vals+1):
                vals.append(dtype(targs[idx + i]))
    else:
        # if parameter is not set in command line, set it to default
        val = default
    return val


def save_dataset(dset, name, config=None):
    fname = os.path.join('datasets', name + '.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(dset, f)
    gname = os.path.join('datasets', 'configs', name + '.json')
    if config is not None:
        with open(gname, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['create', 'load'])
    ap.add_argument('-c', '--config', default=None, help='create from a config file')

    ap.add_argument('-n', '--name', default='debug', help='name for creation')
    ap.add_argument('-t', '--tasks', help='path to json list of tasks')
    ap.add_argument('-d', '--dataset', help='dataset path for loading')

    # task-specific arguments
    ap.add_argument('-a', '--task_args', nargs='*', default=[], help='terms to specify parameters of trial type')
    ap.add_argument('-s', '--seed', default=None, type=int, help='seed for task generation. default is based on time')
    

    args = ap.parse_args()
    if args.config is not None:
        # if using config file, load args from config, ignore everything else
        config_args = load_config(args.config)
        del config_args.name
        del config_args.config
        args = update_config(args, config_args)
    else:
        # add task-specific arguments. shouldn't need to do this if loading from config file
        task_args = get_task_args(args)
        args = update_config(args, task_args)

    args.argv = ' '.join(sys.argv)

    if args.mode == 'create':
        # create and save a dataset
        dset, config = create_dataset(args)
        save_dataset(dset, args.name, config=config)
    elif args.mode == 'load':
        
        # visualize a dataset
        dset = load_rb(args.dataset)
        xr = np.arange(dset[0].trial_len)

        samples = random.sample(dset, 12)
        fig, ax = plt.subplots(3,4,sharex=True, sharey=True, figsize=(14, 8))
        for i, ax in enumerate(fig.axes):

            fig_format.hide_frame(ax)

            trial = samples[i]
            trial_x = trial.get_x(20)
            trial_y = trial.get_y()

            trial_arr = np.concatenate([trial_x, trial_y])
            ax.imshow(trial_arr, aspect='auto', cmap='Blues', interpolation='none', vmin=-1, vmax=1)

            # ax.imshow(trial_y, aspect='auto', cmap='Blues', alpha=1, interpolation='none')
            # ax.imshow(trial_x, aspect='auto', cmap='Oranges', alpha=.5, interpolation='none')
            ax.set_title(f'#{trial.n}: {trial.s_ids}', fontsize=15)
        handles, labels = ax.get_legend_handles_labels()
        #fig.legend(handles, labels, loc='lower center')
        plt.show()
