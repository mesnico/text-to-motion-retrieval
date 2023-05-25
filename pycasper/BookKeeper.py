import pickle as pkl
import json
import os
import sys
from datetime import datetime
from tqdm import tqdm
import copy
import random
import numpy as np
from pathlib import Path
import argparse
import argunparse
from .argsUtils import get_args_update_dict
import warnings

from prettytable import PrettyTable

from torch.utils.tensorboard import SummaryWriter
import torch

try:
  from pycasper.name import Name
except: ## only for unittest
  from name import Name
  
import pdb

def accumulate_grads(model, grads_list):
  if grads_list:
    grads_list = [param.grad.data+old_grad.clone() for param, old_grad in zip(model.parameters(), grads_list)]
  else:
    grads_list += [param.grad.data for param in model.parameters()]
  return grads_list

def save_grads(val, file_path):
  pkl.dump(val, open(file_path, 'wb'))

def load_grads(file_path):
  return pkl.load(open(file_path))

class TensorboardWrapper():
  '''
  Wrapper to add values to tensorboard using a dictionary of values
  '''
  def __init__(self, log_dir):
    self.log_dir = log_dir
    self.writer = SummaryWriter(log_dir=self.log_dir, comment='NA')
    
  def __call__(self, write_dict):
      for key in write_dict:
        for value in write_dict[key]:
          getattr(self.writer, 'add_' + key)(*value)

class BookKeeper():
  '''BookKeeper
  if load_pretrained_model = True
  bookKeeper will not update args and will also call _new_exp

  TODO: add documentation
  TODO: add save_optimizer_args as well
  TODO: choice of score kind to decide early-stopping (currently dev is default)
  Required properties in args
  - load
  - seed
  - save_dir
  - num_epochs
  - cuda
  - save_model
  - greedy_save
  - stop_thresh
  - eps
  - early stopping
  '''
  def __init__(self, args, args_subset,
               args_ext='args.args',
               name_ext='name.name',
               weights_ext='weights.p',
               res_ext='res.json',
               log_ext='log.log',
               script_ext='script.sh',
               args_dict_update={},
               res={'train':[], 'dev':[], 'test':[]},
               tensorboard=None,
               load_pretrained_model=False):

    self.args = args
    self.args_subset = args_subset
    self.args_dict_update = args_dict_update
    
    self.args_ext = args_ext.split('.')
    self.name_ext = name_ext.split('.')
    self.weights_ext = weights_ext.split('.')
    self.res_ext = res_ext.split('.')
    self.log_ext = log_ext.split('.')
    self.script_ext = script_ext.split('.')

    ## params for saving/notSaving models
    self.stop_count = 0

    ## init empty results
    self.res = res
    if 'dev_key' in args:
      self.dev_key = args.dev_key
      self.dev_sign = args.dev_sign
    else:
      self.dev_key = 'dev'
      self.dev_sign = 1
    self.best_dev_score = np.inf * self.dev_sign
    
    self.load_pretrained_model = load_pretrained_model
    
    if self.args.load:
      if os.path.isfile(self.args.load):
        ## update the save_dir if the files have moved
        self.save_dir = Path(args.load).parent.as_posix()

        ## load Name
        self.name = self._load_name()

        ## load args
        self._load_args(args_dict_update)

      # if not self.load_pretrained_model:
      #   ## Serialize and save args
      #   self._save_args()

      ## load results
      self.res = self._load_res()

    else:
      ## run a new experiment
      self._new_exp()

    if self.load_pretrained_model:
      self._new_exp()

    ## Tensorboard 
    if tensorboard:
      self.tensorboard = TensorboardWrapper(log_dir=(Path(self.save_dir)/Path(self.name.name+'tb')).as_posix())
    else:
      self.tensorboard = None

    self._set_seed()

  def _set_seed(self):
    ## seed numpy and torch
    random.seed(self.args.seed)
    np.random.seed(self.args.seed)
    torch.manual_seed(self.args.seed)
    torch.cuda.manual_seed_all(self.args.seed)
    torch.cuda.manual_seed(self.args.seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    
  '''
  Stuff to do for a new experiment
  '''
  def _new_exp(self):
    ## update the experiment number
    self._update_exp()

    self.save_dir = self.args.save_dir
    self.name = Name(self.args, *self.args_subset)

    ## save name
    self._save_name()

    ## update args
    self.args.__dict__.update(self.args_dict_update)
    
    ## Serialize and save args
    self._save_args()

    ## save script
    #self._save_script() ## not functional yet. needs some work

    ## reinitialize results to empty
    self.res = {key:[] for key in self.res}
    
  def _update_exp(self):
    if self.args.exp is not None:
      exp = 0
      exp_file = '.experiments'
      if not os.path.exists(exp_file):
        with open(exp_file, 'w') as f:
          f.writelines([f'{exp}\n'])
      else:
        with open(exp_file, 'r') as f:
          lines = f.readlines()
          exp = int(lines[0].strip())
        exp += 1
        with open(exp_file, 'w') as f:
          f.writelines([f'{exp}\n'])
    else:
      exp = 0
    print(f'Experiment Number: {exp}')
    self.args.__dict__.update({'exp':exp})
    
  def _load_name(self):
    name_filepath = '_'.join(self.args.load.split('_')[:-1] + ['.'.join(self.name_ext)])
    return pkl.load(open(name_filepath, 'rb'))

  def _save_name(self):
    name_filepath = self.name(self.name_ext[0], self.name_ext[1], self.save_dir)
    pkl.dump(self.name, open(name_filepath, 'wb'))

  def _load_res(self):
    res_filepath = self.name(self.res_ext[0], self.res_ext[1], self.save_dir)
    if os.path.exists(res_filepath):
      print('Results Loaded')
      return json.load(open(res_filepath))
    else:
      warnings.warn('Counld not find results file')
      return self.res

  def _save_res(self):
    res_filepath = self.name(self.res_ext[0], self.res_ext[1], self.save_dir)
    json.dump(self.res, open(res_filepath,'w'))

  def update_res(self, res):
    for key in res:
      if key in self.res:
        self.res[key].append(res[key])
      else:
        self.res[key] = [res[key]]

  def update_tb(self, write_dict):
    if self.tensorboard:
      self.tensorboard(write_dict)
    else:
      warnings.warn('TensorboardWrapper not declared')

  def print_res(self, epoch, key_order=['train', 'dev', 'test'], metric_order=[], exp=0, lr=None, fmt='{:.4f}'):
    print_str = "epoch: {}, lr:{}"
    table = PrettyTable([''] + key_order)
    table_str = ['loss'] + [fmt.format(self.res[key][-1]) for key in key_order] ## loss
    table.add_row(table_str)
    for metric in metric_order:
      table_str = [metric] + [fmt.format(self.res['{}_{}'.format(key, metric)][-1]) for key in key_order]
      table.add_row(table_str)
    
    if isinstance(lr, list):
      lr = lr[0]
    tqdm.write(print_str.format(exp, epoch, lr))
    tqdm.write(table.__str__())

  def print_res_archive(self, epoch, key_order=['train', 'dev', 'test'], exp=0, lr=None, fmt='{:.8f}'):
    print_str = ', '.join(["exp: {}, epch: {}, lr:{}, "] + ["{}: {}".format(key,fmt) for key in key_order])
    result_list = [self.res[key][-1] for key in key_order]
    if isinstance(lr, list):
      lr = lr[0]
    tqdm.write(print_str.format(exp, epoch, lr, *result_list))

  def _load_args(self, args_dict_update):
    args_filepath = self.name(self.args_ext[0], self.args_ext[1], self.save_dir)
    if os.path.isfile(args_filepath):
      args_dict = json.load(open(args_filepath))
      ## update load path and cuda device to use
      args_dict.update({'load':self.args.load,
                        'cuda':self.args.cuda,
                        'save_dir':self.save_dir})
      ## any new argument to be updated
      args_dict.update(args_dict_update)

      self.args.__dict__.update(args_dict)

  def _save_args(self):
    args_filepath = self.name(self.args_ext[0], self.args_ext[1], self.save_dir)
    json.dump(self.args.__dict__, open(args_filepath, 'w'))

  def _save_script(self):
    '''
    Not functional 
    '''
    args_filepath = self.name(self.script_ext[0], self.script_ext[1], self.save_dir)
    unparser = argunparse.ArgumentUnparser()
    options = get_args_update_dict(self.args)#self.args.__dict__
    args = {}
    script = unparser.unparse_to_list(*args, **options)
    script = ['python', sys.argv[0]] + script
    script = ' '.join(script)
    with open(args_filepath, 'w') as fp:
      fp.writelines(script)

  def _load_model(self, model):
    weights_path = self.name(self.weights_ext[0], self.weights_ext[1], self.save_dir)
    if isinstance(model, torch.nn.DataParallel):
      model.module.load_state_dict(pkl.load(open(weights_path, 'rb')))
    else:
      t = pkl.load(open(weights_path, 'rb'))
      model.load_state_dict(t)

  @staticmethod
  def load_pretrained_model(model, path2model):
    model.load_state_dict(pkl.load(open(path2model, 'rb')))
    return model
    
  def _save_model(self, model_state_dict):
    weights_path = self.name(self.weights_ext[0], self.weights_ext[1], self.save_dir)
    f = open(weights_path, 'wb') 
    pkl.dump(model_state_dict, f)
    f.close()

  def _copy_best_model(self, model):
    if isinstance(model, torch.nn.DataParallel):
      self.best_model = copy.deepcopy(model.module.state_dict())
    else:
      self.best_model = copy.deepcopy(model.state_dict())
    
  def _start_log(self):
    with open(self.name(self.log_ext[0],self.log_ext[1], self.save_dir), 'w') as f:
      f.write("S: {}\n".format(str(datetime.now())))
    
  def _stop_log(self):
    with open(self.name(self.log_ext[0],self.log_ext[1], self.save_dir), 'r') as f:
      lines = f.readlines()
      if len(lines) > 1: ## this has already been sampled before
        lines = lines[0:1] + ["E: {}\n".format(str(datetime.now()))]
      else:
        lines.append("E: {}\n".format(str(datetime.now())))
    with open(self.name(self.log_ext[0],self.log_ext[1], self.save_dir), 'w') as f:
      f.writelines(lines)

  def stop_training(self, model, epoch, warmup=False):
    ## copy the best model
    if self.dev_sign * self.res[self.dev_key][-1] < self.dev_sign * self.best_dev_score and (not warmup):
      if self.args.greedy_save:
        save_flag = True
      else:
        save_flag=False
      self._copy_best_model(model)
      self.best_dev_score = self.res[self.dev_key][-1]
    else:
      save_flag = False

    ## debug mode with no saving
    if not self.args.save_model:
      save_flag = False

    if self.args.overfit:
      self._copy_best_model(model)    
      save_flag = True

    if save_flag:
      tqdm.write('Saving Model')
      self._save_model(self.best_model)

    ## early_stopping
    if self.args.early_stopping and len(self.res['train'])>=2 and not self.args.overfit:
      if (self.dev_sign*(self.res[self.dev_key][-2] - self.args.eps) < self.dev_sign * self.res[self.dev_key][-1]):
        self.stop_count += 1
      else:
        self.stop_count = 0

    if self.stop_count >= self.args.stop_thresh:
      print('Validation Loss is increasing')
      ## save the best model now
      if self.args.save_model:
        print('Saving Model by early stopping')
        self._save_model(self.best_model)
      return True

    ## end of training loop
    if epoch == self.args.num_epochs-1 and self.args.save_model:
      print('Saving model after exceeding number of epochs')
      self._save_model(self.best_model)

    return False
