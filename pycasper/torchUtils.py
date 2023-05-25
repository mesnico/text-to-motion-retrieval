import torch
import numpy as np

class LambdaScheduler():
  def __init__(self, params, kind='incremental', step_size=0.1, max_lambda=1, max_interval=5000):
    self.params = params
    self.kind = kind
    self.step_size = step_size
    self.max_lambda = max_lambda
    self.max_interval = max_interval
    self.interval = 0
    self.random_list = np.arange(0, max_lambda, step_size)
    
  def step(self):
    if self.interval > self.max_interval:
      self.interval = 0
      getattr(self, self.kind)()
    self.interval += 1
    return self.params
  
  def incremental(self):
    for i, param in enumerate(self.params):
      param += self.step_size
      self.params[i] = min(self.max_lambda, param)

  def random(self):
    idxs = list(torch.randint(0, len(self.random_list), (len(self.params),)).numpy())
    self.params = list(self.random_list[idxs])

def remove_slices(y, mask=[0], dim=-1):
  '''
  Remove slice using mask and return as 2 different `torch.Tensor`s 
    along `dim`
  Can be useful in preprocessing which involves removing some features
  '''
  if not mask:
    return y, None
  inv_mask = torch.LongTensor(sorted(set(range(y.shape[dim])) - set(mask))).to(y.device)
  mask = torch.LongTensor(sorted(mask)).to(y.device)
  return torch.index_select(y, index=inv_mask, dim=dim), torch.index_select(y, index=mask, dim=dim)

def add_slices(y, insert=None, mask=[0], dim=-1):
  '''
  Goes hand-in-hand with remove_slices. It can re-insert the removed slices by `remove_slices`
    along `dim`. 
  Can be useful after the training is complete and the data is needed in the original format
  '''
  if not mask:
    return y
  
  dim = dim % len(y.shape)
  inv_dim = sorted(set(range(len(y.shape))) - set({dim}))

  if insert is None: ## create a zero tensor if None
    insert_shape = list(y.shape)
    insert_shape[dim] = len(mask)
    insert = torch.zeros(insert_shape).to(y.device)

  assert len(insert.shape) == len(y.shape), 'num_dims of `y` and `insert` should be the same'
  assert insert.shape[dim] == len(mask), 'length of mask should be the same as insert.shape[dim]'
  assert (torch.Tensor(list(insert.shape))[inv_dim] == \
          torch.Tensor(list(y.shape))[inv_dim]).all(), \
          'shape of `y` and `insert` should be the same except at `dim`'

  new_shape = list(y.shape)
  new_shape[dim] += len(mask)

  assert max(mask) < new_shape[dim], \
    'max(mask)={} >= new_shape[dim]={}'.format(max(mask), new_shape[dim])

  inv_mask = sorted(set(range(new_shape[dim])) - set(mask))
  inv_slices = (slice(None), ) * dim + (inv_mask,)
  slices = (slice(None), ) * dim + (mask,)

  new_y = torch.zeros(new_shape).to(y.device).type(y.type())
  new_y[inv_slices] = y
  new_y[slices] = insert

  return new_y

class CustomTensor(torch.Tensor):
  def split_dims(self, split_size, dim=0):
    assert isinstance(split_size, int), 'split_size={} is not an int'.format(split_size)
    assert not self.shape[dim]%split_size, 'dim={} is not divisible by split_size={}'.format(dim, split_size)

    new_shape = list(self.shape)
    factor = int(new_shape[dim]/split_size)
    new_shape[dim] = split_size
    new_shape = torch.Size(new_shape[:dim] + [factor] + new_shape[dim:])
    return CustomTensor(self.clone().view(new_shape))

  def merge(self, dim=0, dim2=1):
    dim = dim + len(self.shape) if dim<0 else dim
    dim2 = dim2 + len(self.shape) if dim2<0 else dim2
    
    assert dim<dim2 or (), 'dim2={} must be bigger than dim={}'.format(dim2, dim)
    
    permutation = list(range(len(self.shape)))
    permutation = permutation[:dim + 1] + permutation[dim2:dim2+1] + permutation[dim+1:dim2] + permutation[dim2+1:]
    
    output = self.clone().permute(permutation)
    new_shape = list(output.shape)
    new_shape = torch.Size(new_shape[:dim] + [new_shape[dim]*new_shape[dim+1]] + new_shape[dim+2:])
     
    return CustomTensor(output.contiguous().view(new_shape))

