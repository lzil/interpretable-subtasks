import numpy as np
from scipy.stats import norm
# from sklearn.gaussian_process import GaussianProcessRegressor as gpr
# from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import pickle
import os
import copy
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

'''
tasks
0: copy sensory input to output. duration >=10
1: store sensory input; store last one during ON. duration >=10
2: output stored memory for duration of ON. duration >=10
3: flip memory. duration=10
4: press binary button. duration=10
'''

# dset_id is the name of the dataset (as saved)
# n is the index of the trial in the dataset

# stores task data
class TaskTrial:
    def __init__(self, task_id, task_dict, n_tasks, sensory_data, target_data, dset_id=None, n=None):
        # self.trial_len = trial_len
        self.dset_id = dset_id
        self.n = n

        self.s_ids = task_dict['ids']
        self.s_starts = task_dict['starts']
        self.s_ends = task_dict['ends']
        self.sensory_data = sensory_data
        self.target_data = target_data

        self.task_id = task_id
        self.n_tasks_init = n_tasks

        self.sensory_dim = sensory_data.shape[0]
        self.trial_len = sensory_data.shape[1]

        # assert len(s_ids) == len(s_starts)
        # assert len(s_starts) == len(s_ends)


    def get_task(self, task_count=None, x_noise=0):
        if task_count is None:
            task_count = self.n_tasks_init
        x = np.zeros((self.sensory_dim+task_count, self.trial_len))
        x[:self.sensory_dim] = self.sensory_data
        x[self.sensory_dim+self.task_id, self.s_starts[0]:self.s_ends[-1]] = 1
        # noisy up/down corruption
        if x_noise != 0:
            x = corrupt_x(x, x_noise)
        return x

    def get_sensory_data(self):
        x = copy.deepcopy(self.sensory_data)
        return x

    def get_subtasks(self, subtask_count):
        x = np.zeros((subtask_count, self.trial_len))
        for i in range(self.s_ids):
            x[self.s_ids[i], self.s_starts[i]:self.s_ends[i]] = 1
        return x

    def get_target(self):
        return self.target_data


# add noise to x
def corrupt_x(x, x_noise):
    x += np.random.normal(scale=x_noise, size=x.shape)
    return x


### FOR CREATING TASKSETS

# sample a length for a subtask. used when creating a task
def draw_subtask_length(s_id):
    if s_id == 0:
        # copy sensory input to output
        st_len = min(10 + np.random.exponential(5), 30)
    elif s_id == 1:
        # story sensory input
        st_len = min(10 + np.random.exponential(5), 30)
    elif s_id == 2:
        # output stored memory
        st_len = min(10 + np.random.exponential(5), 30)
    elif s_id == 3:
        # flip memory
        st_len = 10
    elif s_id == 4:
        # toggle binary button
        st_len = 10
    return st_len

# create a task with spaced intervals and subtask lengths
def create_task(s_ids, args):
    s_starts = []
    s_ends = []
    s_start = 0

    # cur s_end
    s_end = 0

    for i in range(len(s_ids)):
        # if not the first subtask, add some interval in between
        if i != 0:
            s_start = s_end + int(np.random.exponential(args.scale))
        # add subtask length
        s_end = s_start + int(draw_subtask_length(s_ids[i]))
        s_starts.append(s_start)
        s_ends.append(s_end)

        # check if ending of this subtask is too late
        # if so, return None to demonstrate task creation failed
        if s_end >= args.trial_len:
            return None

    s_ids = [int(x) for x in s_ids]
    s_starts = [int(x) for x in s_starts]
    s_ends = [int(x) for x in s_ends]

    task_dict = {
        'ids': s_ids,
        'starts': s_starts,
        'ends': s_ends
    }
    return task_dict

# create a set of tasks based on a list of subtask sequences
def create_taskset(args):
    # load from custom subtask_sequences
    s_ids_list = json.load(open(args.subtasks, 'r'))
    taskset = {'tasks': {}, 'config': args}
    task_id = 0
    for i in range(len(s_ids_list)):
        # create the task
        task_dict = None
        while task_dict is None:
            task_dict = create_task(s_ids_list[i], args)
        taskset['tasks'][task_id] = task_dict
        task_id += 1
    return taskset


### FOR CREATING DATASETS

# helper function that will add some random data to sensory and target arrays from the basic task list
def add_basic_subtasks(sensory_arr, target_arr, s_ids, s_starts, s_ends):
    cur_mem = [0, 0]
    for i in range(len(s_ids)):
        s_id = s_ids[i]
        s_start = s_starts[i]
        s_end = s_ends[i]

        angle = np.random.uniform(2*np.pi)
        sin = np.sin(angle)
        cos = np.cos(angle)

        # input is relevant
        if s_id == 0 or s_id == 1:
            sensory_arr[0, s_start:s_end] = sin
            sensory_arr[1, s_start:s_end] = cos

            # output the input
            if s_id == 0:
                target_arr[0, s_start:s_end] = sin
                target_arr[1, s_start:s_end] = cos

            # remember the input
            if s_id == 1:
                cur_mem = [sin, cos]
        
        if s_id == 2:
            target_arr[0, s_start:s_end] = cur_mem[0]
            target_arr[1, s_start:s_end] = cur_mem[1]

        if s_id == 3:
            cur_mem = [-cur_mem[0], -cur_mem[1]]

        if s_id == 4:
            target_arr[2, s_start:s_end] = 1

    return sensory_arr, target_arr

# create a set of trials based on a list of tasks
def create_dataset(args):
    # if args.seed is None:
    #     args.seed = int(str(time.time())[-5:])
    # np.random.seed(args.seed)

    taskset = json.load(open(args.tasks, 'r'))['tasks']
    n_tasks = len(taskset)
    args.n_tasks = n_tasks

    trials = []
    for j in range(args.n_trials):
        # fetch a task id
        task_id = np.random.randint(0, n_tasks)
        task_dict = taskset[str(task_id)]

        s_ids = task_dict['ids']
        s_starts = task_dict['starts']
        s_ends = task_dict['ends']

        trial_len = args.trial_len
        dset_id = args.name

        # distributing subtask sequence within total time allotment
        task_len = s_ends[-1]
        s_start_new = np.random.randint(0, trial_len - task_len)
        s_starts = [x + s_start_new for x in s_starts]
        s_ends = [x + s_start_new for x in s_ends]

        # create sensory and output targets
        sensory_data = np.zeros((2, trial_len))
        target_data = np.zeros((3, trial_len))
        sensory_data, target_data = add_basic_subtasks(sensory_data, target_data, s_ids, s_starts, s_ends)

        # create a trial itself
        new_task_dict = {
            'ids': s_ids,
            'starts': s_starts,
            'ends': s_ends
        }
        trial = TaskTrial(task_id, new_task_dict, n_tasks, sensory_data, target_data, dset_id=args.name, n=j)
        trials.append(trial)

    return trials, args


### GENERAL TASKS.PY FUNCTIONS

# turn task_args argument into usable argument variables
# lots of defaults are written down here
def get_task_args(args):
    tarr = args.task_args
    targs = Bunch()

    if args.mode == 'tasks':
        targs.trial_len = get_tval(tarr, 'len', 150, int)
        targs.scale = get_tval(tarr, 'scale', 5, int)

    elif args.mode == 'trials':
        targs.trial_len = get_tval(tarr, 'len', 150, int)
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

def save_taskset(tset, name, config=None):
    fname = os.path.join('datasets', 'tasksets', name + '.json')
    json.dump(tset, open(fname, 'w'), indent=2)

def save_dataset(dset, name, config=None):
    fname = os.path.join('datasets', name + '.pkl')
    dset_dict = {
        'data': dset,
        'config': config.to_dict()
    }
    with open(fname, 'wb') as f:
        pickle.dump(dset_dict, f)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['tasks', 'trials', 'load'])
    ap.add_argument('-c', '--config', default=None, help='create from a config file')

    ap.add_argument('-n', '--name', default='debug', help='name for creation')
    ap.add_argument('--subtasks', help='path to json list of subtask sequences')
    ap.add_argument('-t', '--tasks', help='path to json list of tasks, built from subtask sequences')
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

    # pdb.set_trace()

    if args.mode == 'tasks':
        # create and save a taskset
        tset = create_taskset(args)
        save_taskset(tset, args.name, config=None)
    elif args.mode == 'trials':
        # create and save a dataset
        dset, config = create_dataset(args)
        save_dataset(dset, args.name, config=config)
    elif args.mode == 'load':
        
        # visualize a dataset
        dset = load_rb(args.dataset)['data']
        xr = np.arange(dset[0].trial_len)
        n_tasks = dset[0].n_tasks_init

        samples = random.sample(dset, 12)
        fig, ax = plt.subplots(3,4,sharex=True, sharey=True, figsize=(14, 6))
        for i, ax in enumerate(fig.axes):

            fig_format.hide_frame(ax)

            trial = samples[i]
            trial_x = trial.get_sensory_data()
            trial_task = np.zeros([1, trial.trial_len])
            trial_task[0,trial.s_starts[0]:trial.s_ends[-1]] = 1
            trial_y = trial.get_target()
            trial_empty = np.zeros([1, trial.trial_len])

            trial_arr = np.concatenate([trial_task, trial_x, trial_empty, trial_y])
            ax.imshow(trial_arr, aspect='auto', cmap='RdBu', interpolation='none', vmin=-1, vmax=1)

            # ax.imshow(trial_y, aspect='auto', cmap='Blues', alpha=1, interpolation='none')
            # ax.imshow(trial_x, aspect='auto', cmap='Oranges', alpha=.5, interpolation='none')
            ax.set_title(f'#{trial.n}: Task {trial.task_id}, {trial.s_ids}', fontsize=15)
            ax.set(yticklabels=[])
        handles, labels = ax.get_legend_handles_labels()
        #fig.legend(handles, labels, loc='lower center')
        plt.show()
