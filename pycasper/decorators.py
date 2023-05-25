import time

__all__ = ['calculate_time']

class BaseDecorator():
  '''
  Build a decorator using before and after execution functions
  
  Must-Implement-Methods::
    before_exec(func) - Stuff to do before function execution
    after_exec(func, before_values) - Stuff to do after function execution
    
  Methods::
    decorator_name = self.build_decorator(before_exec, after_exec)
  '''
  def __init__(self):
    pass
  
  def before_exec(self, func):
    raise NotImplementedError('`before_exec` must be implemented')
  
  def after_exec(self, func, before_values):
    raise NotImplementedError('`after_exec` must be implemented')
  
  def build_decorator(self):  
    def base_decorator(func):   
      def inner(*args, **kwargs):
        before_values = self.before_exec(func)
        returned_values = func(*args, **kwargs)
        self.after_exec(func, before_values)
        return returned_values
      return inner
    return base_decorator

class CalculateTime(BaseDecorator):
  def __init__(self):
    super(CalculateTime, self).__init__()
    
  def before_exec(self, func=None):
    begin = time.time()
    return begin
  
  def after_exec(self, func=None, before_values=None):
    end = time.time()
    print('Execution Time for {}: {:.2f} seconds'.format(func.__name__, end-before_values))

calculate_time = CalculateTime().build_decorator()