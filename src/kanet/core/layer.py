import torch
from torch import nn

from kanet.core.bspline import BSplineBasis
from kanet.utils.phi import phi


# Simple KAN layer
class KANLayer(nn.Module):
    def __init__(self, in_feat, out_feat, grid_range=(-1, 1), num_knots=10, degree=3):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_knots = num_knots
        self.degree = degree

        # Setting up the B-spline grid
        self.grid_min, self.grid_max = grid_range

        knots = torch.linspace(self.grid_min, self.grid_max, num_knots)
        print("Knots:", knots)
        # Register knots as a buffer to ensure proper device management
        self.register_buffer("knots", knots)

        # B-spline array should be (in_feat, out_feat) BSplineBasis instances
        self.bspline_array = [
            [BSplineBasis(self.knots.cpu().numpy(), degree) for _ in range(out_feat)]
            for _ in range(in_feat)
        ]

        self.coeffs = nn.Parameter(
            torch.randn(in_feat, out_feat, num_knots - degree - 1, dtype=torch.float32)
        )

        self.base_activation = nn.SiLU()
        self.base_weights = nn.Parameter(
            torch.randn(in_feat, out_feat, dtype=torch.float32) * 0.1
        )

    def forward(self, x):
        output = torch.zeros(self.out_feat, device=x.device)

        for i in range(self.out_feat):
            for j in range(self.in_feat):
                output[i] += phi(
                    x[j],
                    self.bspline_array[j][i],
                    self.coeffs[j, i],
                    self.base_activation,
                    self.base_weights[j, i],
                )

        return output
