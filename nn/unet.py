import numpy as np

from torch import nn
from torch.nn import functional as F

import utils


class UNet(nn.Module):
    def __init__(self,
                 in_features,
                 max_hidden_features,
                 num_layers,
                 out_features,
                 nonlinearity=F.relu):
        super().__init__()

        assert utils.is_power_of_two(max_hidden_features), \
            '\'max_hidden_features\' must be a power of two.'
        assert max_hidden_features // 2 ** num_layers > 1, \
            '\'num_layers\' must be {} or fewer'.format(int(np.log2(max_hidden_features) - 1))

        self.nonlinearity = nonlinearity
        self.num_layers = num_layers

        self.initial_layer = nn.Linear(in_features, max_hidden_features)

        self.down_layers = nn.ModuleList([
            nn.Linear(
                in_features=max_hidden_features // 2 ** i,
                out_features=max_hidden_features // 2 ** (i + 1)
            )
            for i in range(num_layers)
        ])

        self.middle_layer = nn.Linear(
            in_features=max_hidden_features // 2 ** num_layers,
            out_features=max_hidden_features // 2 ** num_layers)

        self.up_layers = nn.ModuleList([
            nn.Linear(
                in_features=max_hidden_features // 2 ** (i + 1),
                out_features=max_hidden_features // 2 ** i
            )
            for i in range(num_layers - 1, -1, -1)
        ])

        self.final_layer = nn.Linear(max_hidden_features, out_features)

    def forward(self, inputs):
        temps = self.initial_layer(inputs)
        temps = self.nonlinearity(temps)

        down_temps = []
        for layer in self.down_layers:
            temps = layer(temps)
            temps = self.nonlinearity(temps)
            down_temps.append(temps)

        temps = self.middle_layer(temps)
        temps = self.nonlinearity(temps)

        for i, layer in enumerate(self.up_layers):
            temps += down_temps[self.num_layers - i - 1]
            temps = self.nonlinearity(temps)
            temps = layer(temps)

        return self.final_layer(temps)
