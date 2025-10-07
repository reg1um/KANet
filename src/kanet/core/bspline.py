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
