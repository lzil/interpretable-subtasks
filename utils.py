import os
import numpy as np

#import tensorflow as tf

import yaml
import logging
import time
import json
import csv
import pickle
import copy
import pdb
import re
import random
from pathlib import Path


# produce run id and create log directory
def log_training(log_dir='logs', log_name=None, config=None, checkpoints=False):
    # log_dir: the overarching path to all logs for this project
    # log_name: name of this particular log / set of settings
    # run_dir: directory created inside log_dir/log_name for this log
    # run_log: log file for this run

    run_id = str(int(time.time() * 100))[-5:] + f'{random.randrange(256):x}'

    log_name = log_name
    if log_name is None or len(log_name) == 0:
        log_name = run_id

    run_dir = Path(log_dir, log_name)
    Path.mkdir(run_dir, parents=True, exist_ok=True)
    run_log = run_dir / f'log_{run_id}.log'

    checkpoint_dir = None
    if checkpoints:
        checkpoint_dir = run_dir / f'checkpoints_{run_id}'
        Path.mkdir(checkpoint_dir, parents=True, exist_ok=True)

    if config is not None:
        config.run_id = run_id
        run_config = Path(run_dir, f'config_{run_id}.json')
        with open(run_config, 'w', encoding='utf-8') as f:
            json.dump(vars(config), f, indent=4)

    log = Bunch()
    log.run_dir = run_dir
    log.run_log = run_log
    log.run_id = run_id
    log.run_config = run_config
    log.checkpoint_dir = checkpoint_dir
            
    print('\n=== Logging ===', flush=True)
    print(f'ID: {run_id}', flush=True)
    print(f'Name: {log_name}', flush=True)
    print(f'Folder: {run_dir}', flush=True)
    print(f'File: {run_log}', flush=True)
    if checkpoints:
        print(f'Logging checkpoints to: {checkpoint_dir}', flush=True)
    if config:
        print(f'Config file saved to: {run_config}', flush=True)
    print('===============\n', flush=True)

    return log

# turn arbitrary file into config to be used
def load_config(path=None, to_bunch=True):
    load_yaml = lambda p: yaml.safe_load(open(p))
    load_json = lambda p: json.load(open(p, 'r'))

    config = {}
    if path:
        for load_fn in (load_yaml, load_json):
            try:
                config = load_fn(path)
                break
            except:
                pass
        else:
            print('can\'t load either yaml or json, returning empty')
    if to_bunch:
        try:
            return Bunch(config)
        except:
            print('Bunchify failed!')
    else:
        return config

# combine two configs, overwriting with the second
def update_config(config, new_config, use_none=True, to_bunch=True):
    dic = config if type(config) is dict else vars(config)
    new_dic = new_config if type(new_config) is dict else vars(new_config)
    dic = copy.deepcopy(dic)
    new_dic = copy.deepcopy(new_dic)
    for k in new_dic.keys():
        # use_none=True will always overwrite, use_none=False won't use Nones
        if use_none or new_dic[k] is not None:
            dic[k] = new_dic[k]
    if to_bunch:
        return Bunch(dic)
    return dic

# extends basic dict properties to update both __dict__ and dict.items()
def naturalize(*methods):
    def decorate(cls):
        def method_wrapper(method):
            def fix_attrs(self, *args, **kwargs):
                dict_fn = getattr(dict, method)
                super_fn = getattr(super(self.__class__, self), method)
                dict_fn(self.__dict__, *args, **kwargs)
                return super_fn(*args, **kwargs)
            return fix_attrs
        for method in methods:
            setattr(cls, method, method_wrapper(method))
        return cls
    return decorate

@naturalize("clear", "pop", "popitem", "setdefault", "update")
class Bunch(dict):
    def __init__(self, *args, **kwds):
        super().__init__()
        if len(args) > 0:
            for arg in args:
                if type(arg) is dict:
                    self.update(arg)
                else:
                    self.update(arg.__dict__)
        self.update(kwds)
        for k, v in self.items():
            self.__dict__[k] = v

    def __dir__(self):
        # doesn't report extra methods, e.g. to_dict
        return dir(dict)

    def __repr__(self):
        return 'Bunch(' + super().__repr__() + ')'

    def __getattr__(self, k):
        return self.__dict__[k]

    def __setattr__(self, k, v):
        super().__setitem__(k, v)
        self.__dict__[k] = v

    def __delattr__(self, k):
        super().__delitem__(k)
        del self.__dict__[k]

    def __getitem__(self, k):
        return self.__getattr__(k)

    def __setitem__(self, k, v):
        return self.__setattr__(k, v)

    def __delitem__(self, k):
        return self.__delattr__(k)

    def __copy__(self):
        cls = self.__class__
        result = cls(self)
        return result

    def __deepcopy__(self, memo):
        new_dict = {}
        memo[id(self)] = new_dict
        for k, v in self.__dict__.items():
            new_dict[k] = copy.deepcopy(v, memo)
        cls = self.__class__
        result = cls(new_dict)
        return result

    def to_dict(self):
        def unbunchify(x):
            if isinstance(x, dict):
                return dict((k, unbunchify(v)) for k,v in x.items())
            elif isinstance(x, (list, tuple)):
                return type(x)(unbunchify(v) for v in x)
            else:
                return x
        return unbunchify(self)

def load_rb(path):
    with open(path, 'rb') as f:
        qs = pickle.load(f)
    return qs

def lrange(l, p=0.1):
    return np.linspace(0, (l-1) * p, l)


# get config dictionary from the model path
def get_config(path, ctype='model', to_bunch=False):
    head, tail = os.path.split(path)
    if ctype == 'dset':
        fname = '.'.join(tail.split('.')[:-1]) + '.json'
        c_folder = os.path.join(head, 'configs')
        if os.path.isfile(os.path.join(c_folder, fname)):
            c_path = os.path.join(head, 'configs', fname)
        else:
            raise NotImplementedError

    elif ctype == 'model':
        if tail == 'model_best.pth' or 'test' in tail:
            for i in os.listdir(head):
                # using whatever first config file is found in log directory
                if i.startswith('config'):
                    c_path = os.path.join(head, i)
                    break
        else:
            folders = head.split('/')
            if folders[-1].startswith('checkpoints_'):
                # loading a checkpoint model, so have to go back a directory to find config
                run_id = folders[-1].split('_')[-1]
                c_path = os.path.join(*folders[:-1], 'config_'+run_id+'.json')
            else:
                # loading some trained model so just load config in same directory
                run_id = re.split('_|\.', tail)[1]
                c_path = os.path.join(head, 'config_'+run_id+'.json')
        if not os.path.isfile(c_path):
            raise NotImplementedError
    else:
        raise NotImplementedError
    config = load_config(c_path)
    if to_bunch:
        return Bunch(**config)
    else:
        return config

