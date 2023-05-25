from unicodedata import bidirectional
import torch
import torch.nn as nn

class BiGRU(nn.Module):
    def __init__(self, h1, h2, num_layers=1, data_rep='cont_6d', dataset='kit'):
        super(BiGRU, self).__init__()
        num_joints = 21 if dataset == 'kit' else 22
        num_feats = 6 if data_rep=='cont_6d' else 9
        input_dim = num_feats * num_joints

        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2)
        )
        self.gru = nn.GRU(
            h2, h2, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.h2 = h2

    def get_output_dim(self):
        return self.h2 * 2

    def forward(self, x, lengths):
        # reshape inputs
        bs, seq_len = x.shape[:2]
        x = x.view(bs, seq_len, -1)

        # encode inputs 
        x = self.input_encoder(x)

        # handle padded sequence with correct lengths
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # run GRU and concatenate the features from the final layers (because bidirectional)
        _ , hidden = self.gru(x)
        hidden = hidden.view(2, bs, self.h2)
        out = torch.cat([hidden[0], hidden[1]], dim=1)

        return out