import torch
import torch.nn as nn

from models.base_block import SpatialTransformer, BaseModule
from models.UNet import UNet


class VoxelMorph(BaseModule):
    def __init__(self,
                 backbone,
                 feat_num,
                 img_size,
                 integrate_cfg,
                 ):
        super(VoxelMorph, self).__init__()
        self.backbone = backbone

        self.feat_num = feat_num

        self.img_size = img_size

        if self.backbone == 'UNet':
            self.feat_extractor = UNet(self.feat_num)
        else:
            raise NotImplementedError

        self.flow = nn.Conv2d(self.feat_num[0], 2, kernel_size=3, padding=1)

        self.init_weight()
        # init flow layer with small weights and bias
        nn.init.normal_(self.flow.weight, mean=0, std=1e-5)
        nn.init.constant_(self.flow.bias, 0)

        self.stn = SpatialTransformer((self.img_size, self.img_size))

        self.integrate_cfg = integrate_cfg

        if self.integrate_cfg['UseIntegrate']:
            self.vec_int = VecInt(self.img_size, integrate_cfg['TimeStep'])

    def forward(self, moving, fixed):

        fwd = {'Moving': moving, 'Fixed': fixed}

        x = torch.cat([moving, fixed], dim=1)

        feature = self.feat_extractor(x)

        if self.integrate_cfg['UseIntegrate']:
            velocity = self.flow(feature)
            flow_filed = self.vec_int(velocity)
            fwd['Velocity'] = velocity
        else:
            flow_filed = self.flow(feature)

        fwd['Flow'] = flow_filed

        moved = self.stn(moving, flow_filed)

        fwd['Moved'] = moved
        return fwd


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, img_size, n_steps):
        super().__init__()

        assert n_steps >= 0, 'n_steps should be >= 0, found: %d' % n_steps
        self.n_steps = n_steps
        self.scale = 1.0 / (2 ** self.n_steps)
        self.transformer = SpatialTransformer((img_size, img_size))

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.n_steps):
            vec = vec + self.transformer(vec, vec)
        return vec
