import math
import matplotlib.pyplot as plt
import os
import torch
import time
import torch.optim as optim
import torch.nn as nn
from net import quater_emonet2 as quater_emonet

from torchlight.torchlight.io import IO

torch.manual_seed(1234)

rec_loss = losses.quat_angle_loss

def find_all_substr(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


def get_best_epoch_and_loss(path_to_model_files):
    all_models = os.listdir(path_to_model_files)
    if len(all_models) < 2:
        return 0, np.inf
    loss_list = -1. * np.ones(len(all_models))
    acc_list = -1. * np.ones(len(all_models))
    for i, model in enumerate(all_models):
        loss_acc_val = str.split(model, '_')
        if len(loss_acc_val) > 1:
            loss_list[i] = float(loss_acc_val[3])
            acc_list[i] = float(loss_acc_val[5])
    if len(loss_list) < 3:
        best_model = all_models[np.argwhere(loss_list == min([n for n in loss_list if n > 0]))[0, 0]]
    else:
        loss_idx = np.argpartition(loss_list, 2)
        best_model = all_models[loss_idx[1]]
    all_underscores = list(find_all_substr(best_model, '_'))
    # return model name, best loss, best acc
    return best_model, int(best_model[all_underscores[0] + 1:all_underscores[1]]),\
           float(best_model[all_underscores[2] + 1:all_underscores[3]]),\
           float(best_model[all_underscores[4] + 1:all_underscores[5]])



class Processor(object):
    """
        Processor for gait generation
    """

    def __init__(self, args, dataset, data_loader, T, V, C, D, A, S,
                 joints_to_model, joint_offsets,
                 num_labels, prefix_length, target_length,
                 min_train_epochs=20, generate_while_train=False,
                 save_path=None, device='cuda:0'):

        self.args = args
        self.dataset = dataset
        self.mocap = MocapDataset(V, C, joints_to_model=joints_to_model, joint_offsets_all=joint_offsets)
        # self.mocap_old = MocapOld(V, C, joints_to_model, joint_parents_all, joint_parents)
        self.device = device
        self.data_loader = data_loader
        self.num_labels = num_labels
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)
        self.io = IO(
            self.args.work_dir,
            save_log=self.args.save_log,
            print_log=self.args.print_log)

        # model
        self.T = T
        self.V = V
        self.C = C
        self.D = D
        self.A = A
        self.S = S
        self.O = 1
        self.Z = 1
        self.RS = 1
        self.o_scale = 10.
        self.prefix_length = prefix_length
        self.target_length = target_length
        self.model = quater_emonet.QuaterEmoNet(V, D, S, A, self.O, self.Z, self.RS, num_labels[0])
        self.model.cuda(device)
        self.quat_h = None
        self.o_rs_loss_func = nn.L1Loss()
        self.affs_loss_func = nn.L1Loss()
        self.spline_loss_func = nn.L1Loss()
        self.best_loss = math.inf
        self.best_mean_ap = 0.
        self.loss_updated = False
        self.mean_ap_updated = False
        self.step_epochs = [math.ceil(float(self.args.num_epoch * x)) for x in self.args.step]
        self.best_loss_epoch = None
        self.best_acc_epoch = None
        self.min_train_epochs = min_train_epochs

        # generate
        self.generate_while_train = generate_while_train
        self.save_path = save_path

        # optimizer
        if self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.base_lr,
                momentum=0.9,
                nesterov=self.args.nesterov,
                weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.base_lr)
                # weight_decay=self.args.weight_decay)
        else:
            raise ValueError()
        self.lr = self.args.base_lr
        self.tf = self.args.base_tr

    def process_data(self, data, poses, quat, trans, affs):
        data = data.float().to(self.device)
        poses = poses.float().to(self.device)
        quat = quat.float().to(self.device)
        trans = trans.float().to(self.device)
        affs = affs.float().to(self.device)
        return data, poses, quat, trans, affs

    def load_best_model(self, ):
        if self.best_loss_epoch is None:
            model_name, self.best_loss_epoch, self.best_loss, self.best_mean_ap =\
                get_best_epoch_and_loss(self.args.work_dir)
            # load model
            # if self.best_loss_epoch > 0:
        loaded_vars = torch.load(os.path.join(self.args.work_dir, model_name))
        self.model.load_state_dict(loaded_vars['model_dict'])
        self.quat_h = loaded_vars['quat_h']

    def adjust_lr(self):
        self.lr = self.lr * self.args.lr_decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def adjust_tf(self):
        if self.meta_info['epoch'] > 20:
            self.tf = self.tf * self.args.tf_decay

    def show_epoch_info(self):

        print_epochs = [self.best_loss_epoch if self.best_loss_epoch is not None else 0,
                        self.best_acc_epoch if self.best_acc_epoch is not None else 0,
                        self.best_acc_epoch if self.best_acc_epoch is not None else 0]
        best_metrics = [self.best_loss, 0, self.best_mean_ap]
        i = 0
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}. Best so far: {} (epoch: {:d}).'.
                              format(k, v, best_metrics[i], print_epochs[i]))
            i += 1
        if self.args.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):

        if self.meta_info['iter'] % self.args.log_interval == 0:
            info = '\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.args.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def yield_batch(self, batch_size, dataset):
        batch_pos = np.zeros((batch_size, self.T, self.V, self.C), dtype='float32')
        batch_quat = np.zeros((batch_size, self.T, self.V * self.D), dtype='float32')
        batch_orient = np.zeros((batch_size, self.T, self.O), dtype='float32')
        batch_z_mean = np.zeros((batch_size, self.Z), dtype='float32')
        batch_z_dev = np.zeros((batch_size, self.T, self.Z), dtype='float32')
        batch_root_speed = np.zeros((batch_size, self.T, self.RS), dtype='float32')
        batch_affs = np.zeros((batch_size, self.T, self.A), dtype='float32')
        batch_spline = np.zeros((batch_size, self.T, self.S), dtype='float32')
        batch_labels = np.zeros((batch_size, 1, self.num_labels[0]), dtype='float32')
        pseudo_passes = (len(dataset) + batch_size - 1) // batch_size

        probs = []
        for k in dataset.keys():
            if 'spline' not in dataset[k]:
                raise KeyError('No splines found. Perhaps you forgot to compute them?')
            probs.append(dataset[k]['spline'].size())
        probs = np.array(probs) / np.sum(probs)

        for p in range(pseudo_passes):
            rand_keys = np.random.choice(len(dataset), size=batch_size, replace=True, p=probs)
            for i, k in enumerate(rand_keys):
                pos = dataset[str(k)]['positions'][:self.T]
                quat = dataset[str(k)]['rotations'][:self.T]
                # orient = dataset[str(k)]['orientations'][:self.T] * self.o_scale
                orient = np.zeros((self.T, 1))
                affs = dataset[str(k)]['affective_features'][:self.T]
                spline, phase = Spline.extract_spline_features(dataset[str(k)]['spline'])
                spline = spline[:self.T]
                phase = phase[:self.T]
                z = dataset[str(k)]['trans_and_controls'][:, 1][:self.T]
                z_mean = np.mean(z[:self.prefix_length])
                z_dev = z - z_mean
                root_speed = dataset[str(k)]['trans_and_controls'][:, -1][:self.T]
                labels = dataset[str(k)]['labels'][:self.num_labels[0]]

                batch_pos[i] = pos
                batch_quat[i] = quat.reshape(self.T, -1)
                batch_orient[i] = orient.reshape(self.T, -1)
                batch_z_mean[i] = z_mean.reshape(-1, 1)
                batch_z_dev[i] = z_dev.reshape(self.T, -1)
                batch_root_speed[i] = root_speed.reshape(self.T, 1)
                batch_affs[i] = affs
                batch_spline[i] = spline
                batch_labels[i] = np.expand_dims(labels, axis=0)
            yield batch_pos, batch_quat, batch_orient, batch_z_mean, batch_z_dev,\
                  batch_root_speed, batch_affs, batch_spline, batch_labels

    def return_batch(self, batch_size, dataset, randomized=True):
        if len(batch_size) > 1:
            rand_keys = np.copy(batch_size)
            batch_size = len(batch_size)
        else:
            batch_size = batch_size[0]
            probs = []
            for k in dataset.keys():
                if 'spline' not in dataset[k]:
                    raise KeyError('No splines found. Perhaps you forgot to compute them?')
                probs.append(dataset[k]['spline'].size())
            probs = np.array(probs) / np.sum(probs)
            if randomized:
                rand_keys = np.random.choice(len(dataset), size=batch_size, replace=False, p=probs)
            else:
                rand_keys = np.arange(batch_size)

        batch_pos = np.zeros((batch_size, self.T, self.V, self.C), dtype='float32')
        batch_quat = np.zeros((batch_size, self.T, (self.V - 1) * self.D), dtype='float32')
        batch_orient = np.zeros((batch_size, self.T, self.O), dtype='float32')
        batch_z_mean = np.zeros((batch_size, self.Z), dtype='float32')
        batch_z_dev = np.zeros((batch_size, self.T, self.Z), dtype='float32')
        batch_root_speed = np.zeros((batch_size, self.T, self.RS), dtype='float32')
        batch_affs = np.zeros((batch_size, self.T, self.A), dtype='float32')
        batch_spline = np.zeros((batch_size, self.T, self.S), dtype='float32')
        batch_labels = np.zeros((batch_size, 1, self.num_labels[0]), dtype='float32')

        for i, k in enumerate(rand_keys):
            pos = dataset[str(k)]['positions'][:self.T]
            quat = dataset[str(k)]['rotations'][:self.T]
            # orient = dataset[str(k)]['orientations'][:self.T] * self.o_scale
            orient = 0
            affs = dataset[str(k)]['affective_features'][:self.T]
            spline, phase = Spline.extract_spline_features(dataset[str(k)]['spline'])
            spline = spline[:self.T]
            phase = phase[:self.T]
            z = dataset[str(k)]['trans_and_controls'][:, 1][:self.T]
            z_mean = np.mean(z[:self.prefix_length])
            z_dev = z - z_mean
            root_speed = dataset[str(k)]['trans_and_controls'][:, -1][:self.T]
            labels = dataset[str(k)]['labels'][:self.num_labels[0]]

            batch_pos[i] = pos
            batch_quat[i] = quat.reshape(self.T, -1)
            batch_orient[i] = orient.reshape(self.T, -1)
            batch_z_mean[i] = z_mean.reshape(-1, 1)
            batch_z_dev[i] = z_dev.reshape(self.T, -1)
            batch_root_speed[i] = root_speed.reshape(self.T, 1)
            batch_affs[i] = affs
            batch_spline[i] = spline
            batch_labels[i] = np.expand_dims(labels, axis=0)

        return batch_pos, batch_quat, batch_orient, batch_z_mean, batch_z_dev,\
               batch_root_speed, batch_affs, batch_spline, batch_labels

    def per_train(self):

        self.model.train()
        train_loader = self.data_loader['train']
        batch_loss = 0.
        N = 0.

        for pos, quat, orient, z_mean, z_dev,\
                root_speed, affs, spline, labels in self.yield_batch(self.args.batch_size, train_loader):


            # forward
            self.optimizer.zero_grad()
            output = self.model(input_data)
            iter_loss = self.loss_function(input_data, ouput)   # define loss function
            iter_loss.backward() # computes gradient
            # nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)
            self.optimizer.step() #computes back propagation

            # Compute statistics
            batch_loss += iter_loss.item()
            N += quat.shape[0]

            # statistics
            self.iter_info['loss'] = iter_loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            self.iter_info['tf'] = '{:.6f}'.format(self.tf)
            self.show_iter_info()
            self.meta_info['iter'] += 1

        batch_loss = batch_loss / N
        self.epoch_info['mean_loss'] = batch_loss
        self.show_epoch_info()
        self.io.print_timer()
        self.adjust_lr()

    def per_eval(self):

        self.model.eval()
        eval_loader = self.data_loader['eval']
        eval_loss = 0.
        N = 0.

        for pos, quat, orient, z_mean, z_dev,\
                root_speed, affs, spline, labels in self.yield_batch(self.args.batch_size, eval_loader):
            with torch.no_grad():
                output = self.model(input_data)
                iter_loss = self.loss_function(input_data, output)
                eval_loss += iter_loss
                N += quat.shape[0]

        eval_loss /= N
        self.epoch_info['mean_loss'] = eval_loss
        if self.epoch_info['mean_loss'] < self.best_loss and self.meta_info['epoch'] > self.min_train_epochs:
            self.best_loss = self.epoch_info['mean_loss']
            self.best_loss_epoch = self.meta_info['epoch']
            self.loss_updated = True
        else:
            self.loss_updated = False
        self.show_epoch_info()

    def train(self):

        if self.args.load_last_best:
            self.load_best_model()
            self.args.start_epoch = self.best_loss_epoch
        for epoch in range(self.args.start_epoch, self.args.num_epoch):
            self.meta_info['epoch'] = epoch

            # training
            self.io.print_log('Training epoch: {}'.format(epoch))
            self.per_train()
            self.io.print_log('Done.')

            # evaluation
            if (epoch % self.args.eval_interval == 0) or (
                    epoch + 1 == self.args.num_epoch):
                self.io.print_log('Eval epoch: {}'.format(epoch))
                self.per_eval()
                self.io.print_log('Done.')

            # save model and weights
            if self.loss_updated:
                torch.save({'model_dict': self.model.state_dict(),
                            'quat_h': self.quat_h},
                           os.path.join(self.args.work_dir, 'epoch_{}_loss_{:.4f}_acc_{:.2f}_model.pth.tar'.
                                        format(epoch, self.best_loss, self.best_mean_ap * 100.)))
