import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    """
    A double 3x3 same convolution followed by a Batch Normalization
    layer and a ReLU Activation function each.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    """
    U-Net Network Architecture as described in: https://arxiv.org/abs/1505.04597
    But making use of same convolutions and Batch Normalization.
    """
    def __init__(self, in_channels, n_classes, features=[64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contracting path (down)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Expanding path (up)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], n_classes, kernel_size=1)


    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Step size of 2 because ups contains a up-sampling step follow by a double conv
        for idx in range(0, len(self.ups), 2):
            # Do the up-sampling step
            x = self.ups[idx](x)
            feature_map = skip_connections[idx//2]

            if x.shape != feature_map.shape:
                x = TF.resize(x, size=feature_map.shape[2:])

            # Concatenate encoder and decoder feature maps from corresponding levels
            concat = torch.cat((feature_map, x), dim=1)
            # Do the double convolutions
            x = self.ups[idx+1](concat)

        return self.final_conv(x)