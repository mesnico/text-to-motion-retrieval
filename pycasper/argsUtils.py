import sys

def get_args_update_dict(args):
  args_update_dict = {}
  for string in sys.argv:
    string = ''.join(string.split('-'))
    if string in args:
      args_update_dict.update({string:args.__dict__[string]})
  return args_update_dict

