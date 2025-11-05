import torch.nn as nn

from .layer import KANLayer


class KANet(nn.Module):
    def __init__(self, layers_config):
        super().__init__()
        self.layers = nn.ModuleList()
        for config in layers_config:
            layer = KANLayer(
                in_feat=config["in_feat"],
                out_feat=config["out_feat"],
                grid_range=config.get("grid_range", (-1, 1)),
                num_knots=config.get("num_knots", 10),
                degree=config.get("degree", 3),
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def grid_extension(self, x):
        for layer in self.layers:
            layer.grid_extension(x)

    def grid_widening(self, k_extend):
        for layer in self.layers:
            layer.grid_widening(k_extend)
