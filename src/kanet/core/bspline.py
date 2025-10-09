import matplotlib.pyplot as plt
import numpy as np


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


class BSplineBasis:
    """
    Args:
        knots: knots defines the separation of the B-spline segments
        degree: degree of the B-spline basis functions
        position: position at which to evaluate the basis functions
    Returns:
        basis_values: array, values of the basis functions at the given position
    """

    def __init__(self, knots, degree):
        self.knots = np.asarray(knots)
        self.degree = degree
        self.n = len(knots) - degree - 1

    def evaluate(self, position):
        # If position is outside the valid range, return zero basis functions
        if (
            position < self.knots[self.degree]
            or position > self.knots[-self.degree - 1]
        ):
            return np.zeros(self.n)

        # Find the knot interval (index of the knot just before the position)
        knot_interval = np.searchsorted(self.knots, position) - 1
        # Create an identity matrix for control points
        # Control points are the points that shape the B-spline
        control_array = np.eye(self.n)

        # Evaluate basis functions for each control point
        # (TODO: optimize later by batching)
        basis_values = np.array(
            [
                bspline_basis(
                    knot_interval,
                    position,
                    self.knots,
                    control_array[:, i],
                    self.degree,
                )
                for i in range(self.n)
            ]
        )
        return basis_values

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
