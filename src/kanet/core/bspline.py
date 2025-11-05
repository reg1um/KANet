import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# No longer used: Directly implemented in evaluate
def bspline_basis(knot_interval, position, knots_array, control_array, degree):
    # deBoor algorithm to compute B-spline basis functions
    d = [control_array[j + knot_interval - degree] for j in range(degree + 1)]

    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            alpha = (position - knots_array[j + knot_interval - degree]) / (
                knots_array[j + 1 + knot_interval - r]
                - knots_array[j + knot_interval - degree]
            )
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

    return d[degree]


class BSplineBasis(nn.Module):
    """
    Args:
        knots: knots defines the separation of the B-spline segments
        degree: degree of the B-spline basis functions
        position: position at which to evaluate the basis functions
    Returns:
        basis_values: array, values of the basis functions at the given position
    """

    def __init__(self, knots, degree):
        super().__init__()
        self.register_buffer("knots", torch.as_tensor(knots, dtype=torch.float32))
        self.degree = degree
        self.n = len(knots) - degree - 1

    def evaluate(self, position):
        # position: scalar, 1-D tensor (B,), or any array-like
        x = torch.as_tensor(
            position, dtype=self.knots.dtype, device=self.knots.device
        ).view(-1)  # (B,)
        B = x.new_zeros(x.shape[0], self.n)  # (B, n)

        # Initialize degree 0 basis functions
        for i in range(self.n):
            t_i = self.knots[i]
            t_ip1 = self.knots[i + 1]
            if i < self.n - 1:
                mask = (x >= t_i) & (x < t_ip1)
            else:
                # include the right end for the last basis
                mask = (x >= t_i) & (x <= t_ip1)
            B[:, i] = mask.to(B.dtype)

        # Recurrence to degree p (Cox - deBoor)
        for p in range(1, self.degree + 1):
            Bp = x.new_zeros(x.shape[0], self.n)
            for i in range(self.n):
                # Left term
                denom_left = self.knots[i + p] - self.knots[i]
                left = 0.0
                if denom_left > 0:
                    left = ((x - self.knots[i]) / denom_left) * B[:, i]

                # Right term
                right = 0.0
                if i + 1 < self.n:
                    denom_right = self.knots[i + p + 1] - self.knots[i + 1]
                    if denom_right > 0:
                        right = ((self.knots[i + p + 1] - x) / denom_right) * B[
                            :, i + 1
                        ]

                Bp[:, i] = left + right
            B = Bp

        return B

    """
    def evaluate(self, position):

        batch_size = position.shape[0]


        # Find the knot interval (index of the knot just before the position)
        knot_interval = torch.searchsorted(self.knots, position, right=True) - 1
        knot_interval = torch.clamp(knot_interval, self.degree, self.n - 1)

        basis_values = torch.zeros((batch_size, self.n), device=position.device)

        # Create an identity matrix for control points
        # Control points are the points that shape the B-spline
        control_array = torch.eye(self.n, device=position.device)

        # Evaluate basis functions for each control point
        # (TODO: optimize later by batching)


        for i in range(self.n):
            for b in range(batch_size):
                if (
                    position[b] < self.knots[self.degree]
                    or position[b] > self.knots[self.n]
                ):
                    continue

                basis_values[b, i] = bspline_basis(
                    knot_interval[b],
                    position[b].item(),
                    self.knots,
                    control_array[:, i],
                    self.degree,
                )
        return basis_values
        """

    def visualize(self, n_points=200):
        # Define the domain for plotting
        x_min = self.knots[self.degree]
        x_max = self.knots[self.n]
        x_values = np.linspace(x_min, x_max, n_points)

        # Evaluate all basis functions across the entire domain
        basis_values_matrix = np.array([self.evaluate(x) for x in x_values])

        plt.figure(figsize=(12, 7))

        # Plot individual basis functions
        for i in range(self.n):
            plt.plot(
                x_values,
                basis_values_matrix[:, i],
                label=f"$B_{{{i},{self.degree}}}(x)$",
                alpha=0.7,
            )

        # Plot the "unity curve"
        sum_of_basis = np.sum(basis_values_matrix, axis=1)
        plt.plot(
            x_values,
            sum_of_basis,
            label="Sum of Basis (Partition of Unity)",
            color="black",
            linestyle="--",
            linewidth=2.5,
        )
        plt.title(f"B-spline Basis Functions (degree={self.degree})", fontsize=16)

        unique_knots = np.unique(self.knots)
        plt.vlines(
            unique_knots,
            ymin=0,
            ymax=1.1,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label="Knots",
        )
        plt.xlabel("Position (x)", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.ylim(-0.1, 1.2)
        plt.tight_layout()
        plt.show()
