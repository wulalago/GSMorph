import torch
import torch.nn as nn

from models.base_block import ConvBlock, DownSample, UpSample


class UNet(nn.Module):
    def __init__(
            self,
            feat_num,
    ):
        super(UNet, self).__init__()
        self.init_conv = nn.Conv2d(2, feat_num[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.down_conv1 = ConvBlock(feat_num[0], feat_num[0])
        self.down1 = DownSample()

        self.down_conv2 = ConvBlock(feat_num[0], feat_num[1])
        self.down2 = DownSample()

        self.down_conv3 = ConvBlock(feat_num[1], feat_num[2])
        self.down3 = DownSample()

        self.down_conv4 = ConvBlock(feat_num[2], feat_num[3])
        self.down4 = DownSample()

        self.down_conv5 = ConvBlock(feat_num[3], feat_num[4])
        self.down5 = DownSample()

        self.up4 = UpSample()
        self.up_conv4 = ConvBlock(feat_num[4] + feat_num[3], feat_num[3])

        self.up3 = UpSample()
        self.up_conv3 = ConvBlock(feat_num[3] + feat_num[2], feat_num[2])

        self.up2 = UpSample()
        self.up_conv2 = ConvBlock(feat_num[2] + feat_num[1], feat_num[1])

        self.up1 = UpSample()
        self.up_conv1 = ConvBlock(feat_num[1] + feat_num[0], feat_num[0])

    def forward(self, x):
        x = self.init_conv(x)

        e1 = self.down_conv1(x)
        e2 = self.down1(e1)

        e2 = self.down_conv2(e2)
        e3 = self.down2(e2)

        e3 = self.down_conv3(e3)
        e4 = self.down3(e3)

        e4 = self.down_conv4(e4)
        mid = self.down4(e4)

        mid = self.down_conv5(mid)

        d4 = self.up4(mid)
        d4 = torch.cat((e4, d4), dim=1)
        d3 = self.up_conv4(d4)

        d3 = self.up3(d3)
        d3 = torch.cat((e3, d3), dim=1)
        d2 = self.up_conv3(d3)

        d2 = self.up2(d2)
        d2 = torch.cat((e2, d2), dim=1)
        d1 = self.up_conv2(d2)

        d1 = self.up1(d1)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.up_conv1(d1)

        return d1
