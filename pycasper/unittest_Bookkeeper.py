import unittest
import argparse
import copy
from BookKeeper import BookKeeper
import torch
import pdb

dummy_model = torch.nn.Linear(10, 20)

parser = argparse.ArgumentParser()

## BookKeeper Args
parser.add_argument('-exp', type=int, default=None,
                    help='experiment number')
parser.add_argument('-debug', type=int, default=0,
                    help='debug mode')
parser.add_argument('-save_dir', type=str, default='save/model',
                    help='directory to store checkpointed models')
parser.add_argument('-cpk', type=str, default='m',
                    help='checkpointed model name')
parser.add_argument('-dev_key', type=str, default='dev',
                    help='Dev Key. Metric used to decide early stopping')
parser.add_argument('-dev_sign', type=int, default=1,
                    help='if lesser loss is better choose 1, else choose -1')
parser.add_argument('-tb', type=int, default=0,
                    help='Tensorboard Flag')
parser.add_argument('-seed', type=int, default=11212,
                    help='manual seed')
parser.add_argument('-load', type=str, default=None,
                    help='Load weights from this file')
parser.add_argument('-cuda', type=int, default=0,
                    help='choice of gpu device, -1 for cpu')
parser.add_argument('-overfit', type=int, default=0,
                    help='disables early stopping and saves models even if the dev loss increases. useful for performing an overfitting check')
parser.add_argument('-num_epochs', type=int, default=50,
                    help='number of epochs for training')
parser.add_argument('-early_stopping', type=int, default=1,
                    help='Use 1 for early stopping')
parser.add_argument('-greedy_save', type=int, default=1,
                    help='save weights after each epoch if 1')
parser.add_argument('-save_model', type=int, default=1,
                    help='flag to save model at every step')
parser.add_argument('-stop_thresh', type=int, default=3,
                    help='number of consequetive validation loss increses before stopping')
parser.add_argument('-eps', type=float, default=0,
                    help='if the decrease in validation is less than eps, it counts for one step in stop_thresh ')

args, unknown = parser.parse_known_args()


class TestBook(unittest.TestCase):
  def update_book(self, args_update_dict={}):
    self.args = copy.deepcopy(args)
    self.args.__dict__.update(args_update_dict)
    self.book = BookKeeper(self.args, args_subset=['cpk'])

  def test_stopping_criterion_dev_key(self):
    self.update_book(args_update_dict={'dev_key':'dev', 'dev_sign':1})
    res = [{'train':0, 'dev':10, 'test':0},
           {'train':0, 'dev':9, 'test':0},
           {'train':0, 'dev':11, 'test':0}]

    self.book.update_res(res[0])
    self.book.stop_training(dummy_model, 0)
    self.assertEqual(self.book.best_dev_score, 10)

    self.book.update_res(res[1])
    self.book.stop_training(dummy_model, 0)
    self.assertEqual(self.book.best_dev_score, 9)

    self.book.update_res(res[2])
    self.book.stop_training(dummy_model, 0)
    self.assertEqual(self.book.best_dev_score, 9)

    self.update_book(args_update_dict={'dev_key':'dev', 'dev_sign':-1})
    res = [{'train':0, 'dev':9, 'test':0},
           {'train':0, 'dev':10, 'test':0},
           {'train':0, 'dev':8, 'test':0}]

    self.book.update_res(res[0])
    self.book.stop_training(dummy_model, 0)
    self.assertEqual(self.book.best_dev_score, 9)

    self.book.update_res(res[1])
    self.book.stop_training(dummy_model, 0)
    self.assertEqual(self.book.best_dev_score, 10)

    self.book.update_res(res[2])
    self.book.stop_training(dummy_model, 0)
    self.assertEqual(self.book.best_dev_score, 10)

if __name__ == '__main__':
  unittest.main()
