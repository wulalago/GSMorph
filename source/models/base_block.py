import os
import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU()

    def forward(self, x):

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)

        return out


class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.pool(x)
        return out


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()
        self.sample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        out = self.sample(x)
        return out


class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
        self.device_param = nn.Parameter(torch.empty(0), requires_grad=False)

    def forward(self, *args, **kwargs):
        return

    def gradient_update(self, fwd):
        return

    def get_gradient(self):
        grads = []
        for p in self.parameters():
            if p.requires_grad:
                grads.append(p.grad.clone().flatten())
        return grads

    def set_gradient(self, gradients):
        start = 0
        for k, p in enumerate(self.parameters()):
            if p.requires_grad:
                dims = p.shape
                end = start + dims.numel()
                p.grad.data = gradients[start:end].reshape(dims)
                start = end

    def get_layer_names(self):
        names = []
        for name, _ in self.named_parameters():
            names.append(name)

        return names

    @property
    def model_device(self):
        return self.device_param.device

    def load_weight(self, weight_path):
        if not os.path.exists(weight_path):
            raise ValueError('Path Not Exist!')
        pretrained_dict = torch.load(weight_path, map_location=self.device_param.device)
        model_dict = self.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        if len(pretrained_dict) == len(model_dict):
            print('No dropped weights')
        else:
            print('Weights dropped!!')

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0)
                    else:
                        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0)
                    else:
                        nn.init.constant_(param, 1)


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size):
        super().__init__()

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow, mode='bilinear'):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=mode)
