from pathlib import Path

def replace_Nth_parent(filename, by, N=1):
  if isinstance(filename, str):
    filename = Path(filename)
    str_flag = True
  elif isinstance(filename, Path):
    str_flag = False
  else:
    raise 'filename must be either string or Path'
  
  parents = []
  for n in range(N):
    if n == 0:
      parents.append(filename.name)
    else:      
      parents.append(filename.parent.name)
      filename = filename.parent
  filename = filename.parent.parent / by

  for p in reversed(parents):
    filename = filename/p

  if str_flag:
    filename = filename.as_posix()
  return filename    
