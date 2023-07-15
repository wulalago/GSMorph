import torch
import torch.nn as nn


class Grad(nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1'):
        super(Grad, self).__init__()
        self.penalty = penalty

    def forward(self, flow):
        dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
        # dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            # dz = dz * dz

        # d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        d = torch.mean(dx) + torch.mean(dy)
        # grad = d / 3.0
        grad = d / 2.0
        return grad
