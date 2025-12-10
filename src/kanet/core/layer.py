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

        knots = self.set_uniform_knots(num_knots, self.grid_min, self.grid_max, degree)
        # Register knots as a buffer to ensure proper device management
        self.register_buffer("knots", knots)

        # B-spline array should be (in_feat, out_feat) BSplineBasis instances
        self.bspline_array = nn.ModuleList(
            [
                nn.ModuleList(
                    [BSplineBasis(self.knots, degree) for _ in range(out_feat)]
                )
                for _ in range(in_feat)
            ]
        )

        # Coefficients are initialized with N(0, 0.1)
        self.coeffs = nn.Parameter(
            torch.randn(in_feat, out_feat, num_knots - degree - 1, dtype=torch.float32)
            * 0.1
        )

        self.base_activation = nn.SiLU()

        # Base weights initialized with Xavier initialisation
        self.base_weights = nn.Parameter(
            torch.zeros(in_feat, out_feat, dtype=torch.float32)
        )
        nn.init.xavier_uniform_(self.base_weights)

        # For pruning: track which neurons are active
        self._input_mask = None  # Mask for input features
        self._output_mask = None  # Mask for output features

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Make it (1, in_feat)

        batch_size = x.shape[0]

        output = torch.zeros(batch_size, self.out_feat, device=x.device)

        for i in range(self.out_feat):
            for j in range(self.in_feat):
                output[:, i] += phi(
                    x[:, j],
                    self.bspline_array[j][i],
                    self.coeffs[j, i],
                    self.base_activation,
                    self.base_weights[j, i],
                )

        return output

    def get_activations(self, x):
        """
        Get individual activations WITHOUT summing over inputs.
        Used for computing edge importance scores for pruning.

        Returns:
            Tensor of shape (batch_size, in_feat, out_feat) containing
            individual activation values for each edge.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        activations = torch.zeros(
            batch_size, self.in_feat, self.out_feat, device=x.device
        )

        for i in range(self.out_feat):
            for j in range(self.in_feat):
                activations[:, j, i] = phi(
                    x[:, j],
                    self.bspline_array[j][i],
                    self.coeffs[j, i],
                    self.base_activation,
                    self.base_weights[j, i],
                )

        return activations

    def set_uniform_knots(self, num_knots, grid_min, grid_max, degree):
        # Create uniform knots with clamping at the ends

        if num_knots < 2 * (degree + 1):
            # if num_knots < (degree + 1):
            raise ValueError("Number of knots must be at least 2 * (degree + 1)")

        device = (
            torch.device("cpu")
            if not torch.cuda.is_available()
            else torch.device("cuda")
        )

        left = torch.full((degree + 1,), grid_min, device=device)
        right = torch.full((degree + 1,), grid_max, device=device)

        internal_knots = num_knots - 2 * (degree + 1)

        if internal_knots > 0:
            internal = torch.linspace(
                grid_min, grid_max, internal_knots + 2, device=device
            )[1:-1]
            new_knots = torch.cat([left, internal, right])
        else:
            new_knots = torch.cat([left, right])
        return new_knots

    def _grid_extension(self):
        # Grid Extension: Extend the grid by adding extra knots in the middle of
        # each interval and uses least squares minimization to fit the spline
        # coefficients accordingly.

        original_knots = self.knots

        new_knots = []
        for i in range(len(original_knots) - 1):
            new_knots.append(original_knots[i])
            mid_point = (original_knots[i] + original_knots[i + 1]) / 2
            new_knots.append(mid_point)
        new_knots.append(original_knots[-1])

        new_knots = torch.tensor(new_knots, device=original_knots.device)
        return new_knots

    def _least_squares_fit(self, x, y, bspline):
        basis_values = bspline.evaluate(x)

        A = basis_values.T @ basis_values + 1e-6 * torch.eye(
            basis_values.shape[1], device=x.device
        )
        b = basis_values.T @ y.unsqueeze(-1)
        coeffs = torch.linalg.solve(A, b).squeeze(-1)
        # result = torch.linalg.lstsq(basis_values, y.unsqueeze(-1))
        # coeffs = result.solution.squeeze(-1)

        return coeffs

    def grid_extension(self, x):
        # 1. Extend the grid
        new_knots = self._grid_extension()

        # 2. Create new B-spline bases with the new knots
        new_bspline_array = nn.ModuleList(
            [
                nn.ModuleList(
                    [BSplineBasis(new_knots, self.degree) for _ in range(self.out_feat)]
                )
                for _ in range(self.in_feat)
            ]
        )

        # 3. Fit new coefficients using least squares
        new_num_coeffs = new_knots.shape[0] - self.degree - 1
        new_coeffs = torch.zeros(
            self.in_feat,
            self.out_feat,
            new_num_coeffs,
            device=x.device,
            dtype=self.coeffs.dtype,
        )

        with torch.no_grad():
            # Sample points across the grid

            num_samples = max(100, new_num_coeffs * 2)
            x_samples = torch.linspace(
                self.grid_min,
                self.grid_max,
                num_samples,
                device=x.device,
                dtype=x.dtype,
            )

            # 4. For each input-output, refit coeffs
            for i in range(self.out_feat):
                for j in range(self.in_feat):
                    old_basis = self.bspline_array[j][i].evaluate(x_samples)
                    y_spline = torch.einsum("nk,k->n", old_basis, self.coeffs[j, i])

                    new_coeffs[j, i] = self._least_squares_fit(
                        x_samples, y_spline, new_bspline_array[j][i]
                    )

        # 5. Update the layer's knots, bspline_array, and coeffs
        self.register_buffer(
            "knots", new_knots
        )  # Use register_buffer instead of direct assignment
        # self.knots = new_knots
        self.bspline_array = new_bspline_array
        self.coeffs = nn.Parameter(new_coeffs)
        self.num_knots = new_knots.shape[0]

        print("Grid extension completed. New number of knots:", self.num_knots)
        print(f"Knots: ({self.knots[0].item()}, {self.knots[-1].item()})")

    def grid_widening(self, k_extend):
        """
        Caution: This method is experimental and may require further testing.
        (Doesn't comes from KAN paper)
        """
        # Extend the grid by adding k_extend knots on each side with same spacing
        original_knots = self.knots

        spacing = (original_knots[-1] - original_knots[0]) / (len(original_knots) - 1)

        new_left_knots = original_knots[0] - spacing * torch.arange(
            k_extend, 0, -1, device=original_knots.device
        )
        new_right_knots = original_knots[-1] + spacing * torch.arange(
            1, k_extend + 1, device=original_knots.device
        )

        extended_knots = torch.cat([new_left_knots, original_knots, new_right_knots])

        # Calculate new number of basis functions
        new_num_basis = extended_knots.shape[0] - self.degree - 1
        old_num_basis = self.coeffs.shape[2]

        # Create new coefficient tensor with extended size
        new_coeffs = torch.zeros(
            self.in_feat,
            self.out_feat,
            new_num_basis,
            device=self.coeffs.device,
            dtype=self.coeffs.dtype,
        )

        # Copy old coefficients to the center, initialize new ones to zero
        # This preserves the learned spline in the original range
        start_idx = k_extend
        new_coeffs[:, :, start_idx : start_idx + old_num_basis] = self.coeffs.data

        # Update the layer
        self.register_buffer("knots", extended_knots)
        self.coeffs = nn.Parameter(new_coeffs)
        self.num_knots = extended_knots.shape[0]

        # Update B-spline array with new knots
        self.bspline_array = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        BSplineBasis(self.knots, self.degree)
                        for _ in range(self.out_feat)
                    ]
                )
                for _ in range(self.in_feat)
            ]
        )

        print(
            "Grid widening completed. New number of knots: "
            f"{self.num_knots - self.degree * 2 - 1}, basis functions: {new_num_basis}"
        )
        print(f"New grid range: ({self.knots[0].item()}, {self.knots[-1].item()})")
