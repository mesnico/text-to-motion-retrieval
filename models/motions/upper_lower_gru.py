import torch
import torch.nn as nn

from .skeleton import skeleton_parts_ids

class UpperLowerGRU(nn.Module):
    def __init__(self, h1, h2, h3, num_layers=1, data_rep='cont_6d', dataset='kit'):
        super(UpperLowerGRU, self).__init__()

        self.skel_parts_ids = skeleton_parts_ids[dataset]

        f = 6 if data_rep == 'cont_6d' else 6+3
        self.layer1_rarm_enc = nn.Linear(f * len(self.skel_parts_ids['right_arm']), h1)
        self.layer1_larm_enc = nn.Linear(f * len(self.skel_parts_ids['left_arm']), h1)
        self.layer1_rleg_enc = nn.Linear(f * len(self.skel_parts_ids['right_leg']), h1)
        self.layer1_lleg_enc = nn.Linear(f * len(self.skel_parts_ids['left_leg']), h1)
        self.layer1_torso_enc = nn.Linear(f * len(self.skel_parts_ids['mid_body']), h1)
        self.layer2_rarm_enc = nn.Linear(2*h1, h2)
        self.layer2_larm_enc = nn.Linear(2*h1, h2)
        self.layer2_rleg_enc = nn.Linear(2*h1, h2)
        self.layer2_lleg_enc = nn.Linear(2*h1, h2)
        self.batchnorm_up = nn.BatchNorm1d(2*h2)
        self.batchnorm_lo = nn.BatchNorm1d(2*h2)
        self.layer3_arm = nn.GRU(
            2*h2, h3, num_layers=num_layers, batch_first=True)
        self.layer3_leg = nn.GRU(
            2*h2, h3, num_layers=num_layers, batch_first=True)
        self.h3 = h3

    def get_output_dim(self):
        return self.h3 * 2

    def forward(self, P_in, lengths):
        # poseinput is of shape [b,t,num_joints,dim]

        # P_in, h = self.rnn(pose_input)
        b, t = P_in.shape[:2]
        right_arm = P_in[..., self.skel_parts_ids['right_arm'], :].view(b, t, -1)
        left_arm = P_in[..., self.skel_parts_ids['left_arm'], :].view(b, t, -1)
        right_leg = P_in[..., self.skel_parts_ids['right_leg'], :].view(b, t, -1)
        left_leg = P_in[..., self.skel_parts_ids['left_leg'], :].view(b, t, -1)
        mid_body = P_in[..., self.skel_parts_ids['mid_body'], :].view(b, t, -1)

        right_arm_layer1 = self.layer1_rarm_enc(right_arm)
        left_arm_layer1 = self.layer1_larm_enc(left_arm)
        mid_body_layer1 = self.layer1_torso_enc(mid_body)

        right_arm_layer2 = self.layer2_rarm_enc(
            torch.cat((right_arm_layer1, mid_body_layer1), dim=-1))
        left_arm_layer2 = self.layer2_larm_enc(
            torch.cat((left_arm_layer1, mid_body_layer1), dim=-1))

        upperbody = torch.cat((right_arm_layer2, left_arm_layer2), dim=-1)
        upperbody_bn = self.batchnorm_up(
            upperbody.permute(0, 2, 1)).permute(0, 2, 1)

        # handle padded sequence with correct lengths
        upperbody_seq = torch.nn.utils.rnn.pack_padded_sequence(upperbody_bn.view(
            upperbody.shape[0], upperbody.shape[1], -1), lengths, batch_first=True, enforce_sorted=False)
        _, h = self.layer3_arm(upperbody_seq)
        z_p_upper = h.squeeze(0)

        right_leg_layer1 = self.layer1_rleg_enc(right_leg)
        left_leg_layer1 = self.layer1_lleg_enc(left_leg)

        right_leg_layer2 = self.layer2_rleg_enc(
            torch.cat((right_leg_layer1, mid_body_layer1), dim=-1))
        left_leg_layer2 = self.layer2_lleg_enc(
            torch.cat((left_leg_layer1, mid_body_layer1), dim=-1))

        lower_body = torch.cat((right_leg_layer2, left_leg_layer2), dim=-1)
        lower_body_bn = self.batchnorm_lo(
            lower_body.permute(0, 2, 1)).permute(0, 2, 1)

        # handle padded sequence with correct lengths
        lowerbody_seq = torch.nn.utils.rnn.pack_padded_sequence(lower_body_bn.view(
            lower_body.shape[0], lower_body.shape[1], -1), lengths, batch_first=True, enforce_sorted=False)
        _, h = self.layer3_leg(lowerbody_seq)
        z_p_lower = h.squeeze(0)

        motion_emb = torch.cat((z_p_upper, z_p_lower), dim=-1)
        return motion_emb