import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

import hydra

from data_loaders.humanml.scripts.motion_process import recover_rot, recover_rot_pos

class MatchingModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # define devices
        if torch.cuda.is_available():
            device_1 = "cuda:0"
            device_2 = "cuda:0" if torch.cuda.device_count() == 1 else "cuda:1"
        else:
            device_1 = "cpu"
            device_2 = "cpu"

        # motion encoder
        self.pose_enc = hydra.utils.instantiate(config.motion_model.module, data_rep=config.data_rep, dataset=config.data.train.name)
        pose_out_dim = self.pose_enc.get_output_dim()
        self.pose_proj = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(config.final_dropout),
            nn.Linear(pose_out_dim, config.common_space_dim)
        )
        self.pose_enc.to(device_2)
        self.pose_proj.to(device_2)

        # sentence encoder
        self.sentence_enc = hydra.utils.instantiate(config.text_model.module)
        text_out_dim = self.sentence_enc.get_output_dim()
        self.sentence_proj = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(config.final_dropout),
            nn.Linear(text_out_dim, config.common_space_dim))

        self.sentence_enc.to(device_1)
        self.sentence_proj.to(device_1)

        # matching loss
        self.matching_loss = hydra.utils.instantiate(config.optim.loss, pose_out_dim=pose_out_dim, text_out_dim=text_out_dim)
        self.matching_loss.to(device_1)

        self.data_rep = config.data_rep
        self.device_1 = device_1
        self.device_2 = device_2

    def transform_representation(self, motion):
        if self.data_rep == 'cont_6d':
            motion = recover_rot(motion)
        elif self.data_rep == 'cont_6d_plus_rifke':
            motion = recover_rot_pos(motion)
        return motion

    def compute_embeddings(self, motion, motion_len, text, return_all=False):
        # transform motion into correct representation
        motion = motion.to(self.device_2)
        motion = self.transform_representation(motion)  # B x seqlen x num_joints x dims

        # process motion
        motion_emb_bkb = self.pose_enc(motion, motion_len)
        motion_emb = self.pose_proj(motion_emb_bkb)

        # process sentence
        text_emb_bkb = self.sentence_enc(text)
        text_emb = self.sentence_proj(text_emb_bkb)

        # normalize
        motion_emb = F.normalize(motion_emb, p=2, dim=1)
        text_emb = F.normalize(text_emb, p=2, dim=1)

        # move motion embs to device_1 so that both motion and text are on the same device
        motion_emb = motion_emb.to(self.device_1)
        motion_emb_bkb = motion_emb_bkb.to(self.device_1)

        if return_all:
            return {'motion_emb': motion_emb,
                    'motion_emb_bkb': motion_emb_bkb,
                    'text_emb': text_emb,
                    'text_emb_bkb': text_emb_bkb,
                    'texts': text}
        else:
            return motion_emb, text_emb

    def compute_loss(self, data, epoch):
        loss, monitors = self.matching_loss(data, epoch=epoch)
        return loss, monitors

    def forward(self, motion, motion_len, text, epoch=0):
        # forward the embeddings
        comp_data = self.compute_embeddings(motion, motion_len, text, return_all=True)

        # compute loss
        loss, monitors = self.compute_loss(comp_data, epoch)
        return loss, monitors
