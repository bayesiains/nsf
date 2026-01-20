import torch

from torch import nn
from torch.nn import functional as F


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation=1):
        super(GatedConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, 2 * out_channels, kernel_size,
                              stride, padding, dilation)

    def forward(self, inputs):
        temps = self.conv(inputs)
        outputs = F.glu(temps, dim=1)
        return outputs


class GatedConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, output_padding=0, dilation=1):
        super(GatedConvTranspose2d, self).__init__()

        self.conv_transpose = nn.ConvTranspose2d(in_channels, 2 * out_channels,
                                                 kernel_size, stride, padding,
                                                 output_padding, dilation=dilation)

    def forward(self, inputs):
        temps = self.conv_transpose(inputs)
        outputs = F.glu(temps, dim=1)
        return outputs


class SylvesterFlowConvEncoderNet(nn.Module):
    def __init__(self, context_features, last_kernel_shape=(7, 7)):
        super().__init__()
        self.context_features = context_features
        self.last_kernel_shape = last_kernel_shape

        self.gated_conv_layers = nn.ModuleList([
            GatedConv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                padding=2,
                stride=1
            ),
            GatedConv2d(  # 2
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                padding=2,
                stride=2
            ),
            GatedConv2d(  # 3
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2,
                stride=1
            ),
            GatedConv2d(  # 4
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                padding=2,
                stride=2
            ),
            GatedConv2d(  # 5
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                padding=2,
                stride=1
            ),
            GatedConv2d(  # 6
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                padding=2,
                stride=1
            ),
            GatedConv2d(  # 7
                in_channels=64,
                out_channels=256,
                kernel_size=self.last_kernel_shape,
                padding=0,
                stride=1
            )
        ])

        self.context_layer = nn.Linear(
            in_features=256,
            out_features=self.context_features
        )

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        temps = inputs
        del inputs
        for gated_conv in self.gated_conv_layers:
            temps = gated_conv(temps)
        outputs = self.context_layer(temps.reshape(batch_size, -1))
        del temps
        return outputs


class SylvesterFlowConvDecoderNet(nn.Module):
    def __init__(self, latent_features, last_kernel_shape=(7, 7)):
        super().__init__()
        self.latent_features = latent_features
        self.last_kernel_shape = last_kernel_shape

        self.gated_conv_transpose_layers = nn.ModuleList([
            GatedConvTranspose2d(
                in_channels=self.latent_features,
                out_channels=64,
                kernel_size=self.last_kernel_shape,
                padding=0,
                stride=1
            ),
            GatedConvTranspose2d(  # 2
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                padding=2,
                stride=1
            ),
            GatedConvTranspose2d(  # 3
                in_channels=64,
                out_channels=32,
                kernel_size=5,
                padding=2,
                stride=2,
                output_padding=1
            ),
            GatedConvTranspose2d(  # 4
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                padding=2,
                stride=1
            ),
            GatedConvTranspose2d(  # 5
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                padding=2,
                stride=2,
                output_padding=1
            ),
            GatedConv2d(  # 6
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                padding=2,
                stride=1
            ),
            GatedConv2d(  # 7
                in_channels=32,
                out_channels=1,
                kernel_size=1,
                padding=0,
                stride=1
            )
        ])

    def forward(self, inputs):
        temps = inputs[..., None, None]
        del inputs
        for gated_conv_transpose in self.gated_conv_transpose_layers:
            temps = gated_conv_transpose(temps)
        outputs = temps
        del temps
        return outputs


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, resample=None, activation=F.relu,
                 dropout_probability=0., first=False):
        super().__init__()
        self.in_channels = in_channels
        self.resample = resample
        self.activation = activation

        self.residual_layer_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1
        )

        if resample is None:
            self.shortcut_layer = nn.Identity()
            self.residual_2_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1
            )
        elif resample == 'down':
            self.shortcut_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=2 * in_channels,
                kernel_size=3,
                stride=2,
                padding=1
            )
            self.residual_2_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=2 * in_channels,
                kernel_size=3,
                stride=2,
                padding=1
            )
        elif resample == 'up':
            self.shortcut_layer = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0 if first else 1
            )
            self.residual_2_layer = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0 if first else 1
            )

        if dropout_probability > 0:
            self.dropout = nn.Dropout(dropout_probability)
        else:
            self.dropout = None

    def forward(self, inputs):

        shortcut = self.shortcut_layer(inputs)
        residual_1 = self.activation(inputs)
        residual_1 = self.residual_layer_1(residual_1)
        if self.dropout is not None:
            residual_1 = self.dropout(residual_1)
        residual_2 = self.activation(residual_1)
        residual_2 = self.residual_2_layer(residual_2)

        return shortcut + residual_2


class ConvEncoder(nn.Module):
    def __init__(self, context_features, channels_multiplier,
                 activation=F.relu, dropout_probability=0.):
        super().__init__()
        self.context_features = context_features
        self.channels_multiplier = channels_multiplier
        self.activation = activation

        self.initial_layer = nn.Conv2d(1, channels_multiplier, kernel_size=1)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(in_channels=channels_multiplier,
                          dropout_probability=dropout_probability),
            ResidualBlock(in_channels=channels_multiplier, resample='down',
                          dropout_probability=dropout_probability),
            ResidualBlock(in_channels=channels_multiplier * 2,
                          dropout_probability=dropout_probability),
            ResidualBlock(in_channels=channels_multiplier * 2, resample='down',
                          dropout_probability=dropout_probability),
            ResidualBlock(in_channels=channels_multiplier * 4,
                          dropout_probability=dropout_probability),
            ResidualBlock(in_channels=channels_multiplier * 4, resample='down',
                          dropout_probability=dropout_probability)
        ])
        self.final_layer = nn.Linear(
            in_features=(4 * 4 * channels_multiplier * 8),
            out_features=context_features
        )

    def forward(self, inputs):
        temps = self.initial_layer(inputs)
        for residual_block in self.residual_blocks:
            temps = residual_block(temps)
        temps = self.activation(temps)
        outputs = self.final_layer(temps.reshape(-1, 4 * 4 * self.channels_multiplier * 8))
        return outputs


class ConvDecoder(nn.Module):
    def __init__(self, latent_features, channels_multiplier,
                 activation=F.relu, dropout_probability=0.):
        super().__init__()
        self.latent_features = latent_features
        self.channels_multiplier = channels_multiplier
        self.activation = activation

        self.initial_layer = nn.Linear(
            in_features=latent_features,
            out_features=(4 * 4 * channels_multiplier * 8)
        )
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(in_channels=channels_multiplier * 8,
                          dropout_probability=dropout_probability),
            ResidualBlock(in_channels=channels_multiplier * 8, resample='up', first=True,
                          dropout_probability=dropout_probability),
            ResidualBlock(in_channels=channels_multiplier * 4,
                          dropout_probability=dropout_probability),
            ResidualBlock(in_channels=channels_multiplier * 4, resample='up',
                          dropout_probability=dropout_probability),
            ResidualBlock(in_channels=channels_multiplier * 2,
                          dropout_probability=dropout_probability),
            ResidualBlock(in_channels=channels_multiplier * 2, resample='up',
                          dropout_probability=dropout_probability)
        ])
        self.final_layer = nn.Conv2d(
            in_channels=channels_multiplier,
            out_channels=1,
            kernel_size=1
        )

    def forward(self, inputs):
        temps = self.initial_layer(inputs).reshape(
            -1, self.channels_multiplier * 8, 4, 4
        )
        for residual_block in self.residual_blocks:
            temps = residual_block(temps)
        temps = self.activation(temps)
        outputs = self.final_layer(temps)
        return outputs


def main():
    batch_size, channels, width, height = 16, 1, 28, 28
    inputs = torch.rand(batch_size, channels, width, height)

    net = ConvEncoder(context_features=24, channels_multiplier=16)
    outputs = net(inputs)

    net = ConvDecoder(latent_features=24, channels_multiplier=16)
    outputs = net(outputs)


if __name__ == '__main__':
    main()
