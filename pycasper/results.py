import os
import pdb
import json
import pandas as pd
import numpy as np
from warnings import warn

"""
walkthroughResults

  path: the directory where all the result files are stored
  args_subset: (optional) list of args to be added to the table; None if you want to add everything
  res_subset: (optional) list of result values to be added to the table; None if you want to add everything
  val_key: (optional) name of the result type to be used for selecting the best model
         must be present in the `result file` and `res_subset`
"""
def walkthroughResults(path, args_subset=None, 
                       res_subset=['train', 'val', 'test'], 
                       val_key='val',
                       log='log.log'):
  if isinstance(path, str):
    paths = [path]
  else:
    paths = path
    
  ## discover args columns of a table
  for path in paths:
    for tup in os.walk(path):
      for fname in tup[2]:
        if fname.split('_')[-1] == 'args.args':
          try:
            all_args = json.load(open(os.path.join(tup[0], fname)))
          except:
            continue
          if args_subset is None:
            args_subset = all_args
          else:
            ## check if the args in the subset are available in the args
            for arg in args_subset:
              try:
                all_args[arg]
              except:
                warn('arg {} not in the args file of the model'.format(arg))
          ## assign [] to all args in args_subset
          best_df_dict = dict([(arg, []) for arg in args_subset])
          all_df_dict = dict([(arg, []) for arg in args_subset])

          break
      else:
        continue
      break

  ## discover result columns of the table
  for path in paths:
    for tup in os.walk(path):
      for fname in tup[2]:
        if fname.split('_')[-1] == 'res.json':
          all_res = json.load(open(os.path.join(tup[0], fname)))
          if res_subset is None:
            res_subset = all_res
          else:
            ## check if the res in the subset are available in the res.json file
            for res in res_subset:
              try:
                all_res[res]
              except:
                warn('res {} not in the res.json file of the model'.format(arg))
          if isinstance(val_key, str):
            assert np.array([r == val_key for r in res_subset]).any() or val_key is None, 'res_key not found in res_subset'
          ## assign [] to all res in res_subset
          best_df_dict.update(dict([(res, []) for res in res_subset]))
          all_df_dict.update(dict([(res, []) for res in res_subset]))
          break
      else:
        continue
      break
    
  ## add epoch to both the dictionaries
  best_df_dict.update({'epoch':[]})
  all_df_dict.update({'epoch':[]})

  ## add name to both dictionaries
  best_df_dict.update({'name':[]})
  all_df_dict.update({'name':[]})

  ## add status key to both dictionaries
  best_df_dict.update({'status':[]})
  all_df_dict.update({'status':[]})  

  for path in paths:
    for tup in os.walk(path):
      for fname in tup[2]:
        if fname.split('_')[-1] == 'res.json':
          ## load raw results
          res = json.load(open(os.path.join(tup[0],fname)))

          ## load args
          name = '_'.join(fname.split('.')[0].split('_')[:-1])
          args_path = '_'.join(fname.split('.')[0].split('_')[:-1] + ['args.args'])
          args = json.load(open(os.path.join(tup[0], args_path)))

          ## find the best result index
          if isinstance(val_key, str):
            min_index = np.argmin(res[val_key])
          elif isinstance(val_key, int):
            min_index = -val_key ## take k'th value from the end
          else:
            min_index =  -1 ## take the last value if val_key is not provided

          ## add args to df_dict
          for arg in args_subset:
            best_df_dict[arg].append(args.get(arg))
            all_df_dict[arg].append(args.get(arg))

          ## add loss values to df_dict
          for r in res_subset:
            if res.get(r):
              try:
                best_df_dict[r].append(res.get(r)[min_index])
              except:
                pdb.set_trace()
            else:
              best_df_dict[r].append(None)
            all_df_dict[r].append(res.get(r))

          ## add num_epochs to train to df_dict
          best_df_dict['epoch'].append(min_index+1)
          all_df_dict['epoch'].append(min_index+1)

          ## add name to dict
          best_df_dict['name'].append(name)
          all_df_dict['name'].append(name)

          ## add if the experiment is running or not
          log_file = '_'.join(fname.split('_')[:-1] + [log])
          log_file = os.path.join(tup[0],log_file)
          if not os.path.exists(log_file):
            status = 'DL'
          else:
            with open(log_file, 'r') as f:
              lines = f.readlines()
              if len(lines) == 1:
                status = 'started'
              elif len(lines) == 2:
                status = 'ended'
              else:
                status = 'rendered'
          best_df_dict['status'].append(status)
          all_df_dict['status'].append(status)
            

  ## Convert dictionary of results to a dataframe
  best_df = pd.DataFrame(best_df_dict)
  all_df = pd.DataFrame(all_df_dict)
  best_df = best_df[['name'] + list(args_subset) + ['epoch'] + list(res_subset) + ['status']]
  all_df = all_df[['name'] + list(args_subset) + ['epoch'] + list(res_subset) + ['status']]
  return best_df, all_df

def walkthroughMetrics(path, args_subset=None, 
                       res_subset=['train', 'val', 'test']):
  if isinstance(path, str):
    paths = [path]
  else:
    paths = path
    
  res_list = []
  for path in paths:
    for tup in os.walk(path):
      for fname in tup[2]:
        if fname.split('_')[-1] == 'cummMetrics.json':
          ## load res
          res = json.load(open(os.path.join(tup[0], fname)))
          fname_split = fname.split('_')
          fname_split[-1] = 'args.args'

          ## load args
          all_args = json.load(open(os.path.join(tup[0], '_'.join(fname_split))))
          if args_subset is None:
            args = all_args
            args_subset = args.keys()
          else:
            args = {}
            for arg in args_subset:
              args[arg] = all_args.get(arg)

          res.update(args)

          ## add name
          res['name'] = '_'.join(fname.split('.')[0].split('_')[:-1])
          res_list.append(res)

  df = pd.DataFrame(res_list, columns=['name'] + args_subset + res_subset)
  return df
                
def walkthroughModels(path):
  model_paths = []
  for tup in os.walk(path):
    for fname in tup[2]:
      if fname.split('_')[-1] == 'weights.p':
        model_paths.append(os.path.join(tup[0], fname))

  return model_paths
