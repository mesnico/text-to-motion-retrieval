import copy as cp
import torch
import torch.nn as nn

from .gcnutils import dggcn, dgmstcn, unit_tcn, Graph

EPS = 1e-4


class DGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, **kwargs):
        super().__init__()
        # prepare kwargs for gcn and tcn
        common_args = ['act', 'norm', 'g1x1']
        for arg in common_args:
            if arg in kwargs:
                value = kwargs.pop(arg)
                kwargs['tcn_' + arg] = value
                kwargs['gcn_' + arg] = value

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[1:4] != 'cn_'}
        assert len(kwargs) == 0

        self.gcn = dggcn(in_channels, out_channels, A, **gcn_kwargs)
        self.tcn = dgmstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)

        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)


class DGSTGCN(nn.Module):
    def __init__(self,
                 graph_cfg,
                 dataset,
                 data_rep,
                 base_channels=64,
                 num_frames=100,
                 ch_ratio=2,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 data_bn_type='VC',
                 num_person=1,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        in_channels = 6 if data_rep == 'cont_6d' else 6+3

        self.graph = Graph(layout=dataset, **graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs
        self.num_frames = num_frames

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)
        lw_kwargs[0].pop('g1x1', None)
        lw_kwargs[0].pop('gcn_g1x1', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages
        modules = []
        if self.in_channels != self.base_channels:
            modules = [DGBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        down_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(DGBlock(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))
            down_times += (i in down_stages)

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained
        self.pool = nn.AdaptiveAvgPool2d(1)

    # def init_weights(self):
    #     if isinstance(self.pretrained, str):
    #         self.pretrained = cache_checkpoint(self.pretrained)
    #         load_checkpoint(self, self.pretrained, strict=False)

    def get_output_dim(self):
        return 256

    def forward(self, x, lengths):
        # adapt the data format to a format dgstgcn can understand
        # x = x.unsqueeze(1)

        N, T, V, C = x.size()

        # temporal uniform subsample
        x = x.permute(0, 3, 2, 1).contiguous()  # N x C x V x T
        x_new = []
        for i, L in enumerate(lengths):
            t = x[i:i+1, ..., :L]
            t = torch.nn.functional.interpolate(t, [V, self.num_frames])
            x_new.append(t)
        x = torch.cat(x_new, dim=0)
        T = self.num_frames

        # normalization
        if self.data_bn_type == 'MVC':
            raise NotImplementedError
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N, V * C, T))
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous().view(N, C, T, V)

        # forward core
        for i in range(self.num_stages):
            x = self.gcn[i](x)

        # x = x.reshape((N, M) + x.shape[1:])
        x = self.pool(x)
        x = x.squeeze()

        return x
