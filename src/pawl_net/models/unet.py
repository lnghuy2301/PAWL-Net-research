"""Standard U-Net used by PAWL-Net; original checkpoint names are preserved."""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class StandardUNet(nn.Module):
    def __init__(self, in_channels=10, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up_conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up_conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        if x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError("U-Net spatial dimensions must be divisible by 16")
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.down4(self.pool3(x3))

        x = self.bottleneck(self.pool4(x4))

        x = self.up4(x)
        x = torch.cat([x, x4], dim=1)
        x = self.up_conv4(x)

        x = self.up3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv3(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv1(x)

        return torch.sigmoid(self.out_conv(x))
