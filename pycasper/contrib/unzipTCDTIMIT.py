'''
Unzip/Untar all files in a given directory and all its subdirectories.
Could be useful in unpacking a huge dataset
'''
import os
from pathlib import Path
import zipfile

import argparse
from tqdm import tqdm

def unzip(path, delete):
  zip_ref = zipfile.ZipFile(path.as_posix(), 'r')
  zip_ref.extractall(path=path.parent.as_posix())
  if delete:
    os.remove(path)
  zip_ref.close()

def untar(path, delete):
  tar_ref = tarfile.open(path.as_posix())
  tar_ref.extractall(path=path.parent.as_posix())
  if delete:
    os.remove(path)
  tar_ref.close()
  
def walkThroughNunzip(submissionPath, delete):
  count = 0
  pbar = tqdm()
  for tup in os.walk(submissionPath):
    for file in tup[2]:
      ext = file.split('.')[-1]
      if  ext == 'zip':
        unzip(Path(tup[0])/Path(file), delete)
        print(Path(tup[0])/Path(file))
        count+=1
        pbar.update(1)
      elif ext == 'tar' or ext == 'gz':
        untar(Path(tup[0])/Path(file), delete)
        print(Path(tup[0])/Path(file))
        count+=1
        pbar.update(1)
  pbar.close()
  print('Zip files extracted: {}'.format(count))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-path', type=str, default='./',
                      help='add path to a dataset containing zip/tar/gz files to be unzipped')
  parser.add_argument('-delete', type=int, default=0,
                      help='delete zip file after extracting')

  args = parser.parse_args()

  walkThroughNunzip(args.path, args.delete)
