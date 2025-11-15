import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2):
        super(ConvBlock, self).__init__()
        layers = []
        for _ in range(num_convs):
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, num_convs)
        self.pool = nn.MaxPool2d(2, return_indices=True)

    def forward(self, x):
        x = self.conv_block(x)
        x, indices = self.pool(x)
        return x, indices


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2):
        super(DecoderBlock, self).__init__()
        self.unpool = nn.MaxUnpool2d(2)
        self.conv_block = ConvBlock(in_channels, out_channels, num_convs)

    def forward(self, x, indices):
        x = self.unpool(x, indices)
        x = self.conv_block(x)
        return x


class SegNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegNet, self).__init__()

        # Энкодер
        self.enc1 = EncoderBlock(in_channels, 64, 2)
        self.enc2 = EncoderBlock(64, 128, 2)
        self.enc3 = EncoderBlock(128, 256, 3)
        self.enc4 = EncoderBlock(256, 512, 3)
        self.enc5 = EncoderBlock(512, 512, 3)

        # Декодер
        self.dec5 = DecoderBlock(512, 512, 3)
        self.dec4 = DecoderBlock(512, 256, 3)
        self.dec3 = DecoderBlock(256, 128, 3)
        self.dec2 = DecoderBlock(128, 64, 2)
        self.dec1 = nn.Sequential(
            nn.MaxUnpool2d(2),
            ConvBlock(64, 64, 2),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )

    def forward(self, x):
        # Энкодер
        x1, idx1 = self.enc1(x)
        x2, idx2 = self.enc2(x1)
        x3, idx3 = self.enc3(x2)
        x4, idx4 = self.enc4(x3)
        x5, idx5 = self.enc5(x4)

        # Декодер
        x = self.dec5(x5, idx5)
        x = self.dec4(x, idx4)
        x = self.dec3(x, idx3)
        x = self.dec2(x, idx2)
        x = self.dec1[0](x, idx1)  # unpool
        x = self.dec1[1](x)  # conv_block
        x = self.dec1[2](x)  # final conv

        return x
