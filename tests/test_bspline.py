import numpy as np

from src.kanet.core.bspline import BSplineBasis

# Nice website to visualize and understand B-Splines:
# https://www.desmos.com/calculator/ql6jqgdabs


# Test case 1: Basis vector sum equals to 1
def test_basis_vector_sum():
    degree = 3
    knots = [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]
    spline = BSplineBasis(knots, degree)
    u = 2
    basis_vector = spline.evaluate(u)
    assert abs(sum(basis_vector) - 1) < 1e-6, "Basis vector sum does not equal to 1"


# Basic Degree-0 Knot Test
## Degree 0 is a square wave, so at u=1.5,
## the only non-zero basis function is the one corresponding to the interval [1,2)
def test_degree_0():
    degree = 0
    knots = [0, 1, 2, 3]
    spline = BSplineBasis(knots, degree)
    u = 1.5
    basis_vector = spline.evaluate(u)
    expected = [0, 1, 0]
    print(basis_vector)
    assert np.allclose(
        basis_vector, expected
    ), f"Expected {expected}, got {basis_vector}"


# Basic Degree-1 Knot Test
## Degree 1 is a triangle wave, so at u=1.5,
## the basis functions corresponding to intervals [1,2) and
## [0,1) are non-zero (and we are in the middle of them
def test_degree_1():
    degree = 1
    knots = [0, 0, 1, 2, 3, 3]
    spline = BSplineBasis(knots, degree)
    u = 1.5
    basis_vector = spline.evaluate(u)
    expected = [0, 0.5, 0.5, 0]
    print(basis_vector)
    assert np.allclose(
        basis_vector, expected
    ), f"Expected {expected}, got {basis_vector}"


# Basic Degree-2 Knot Test
## Degree 2 is a quadratic wave, so at u=1.5,
## the basis functions corresponding to intervals
## [0,2) (a), [0,3) (b) and [1,3) (c) are non-zero
## and we are in the middle of the "intersection" of (a) and (c) (and peak of (b))
def test_degree_2():
    degree = 2
    knots = [0, 0, 0, 1, 2, 3, 3, 3]
    spline = BSplineBasis(knots, degree)
    u = 1.5
    basis_vector = spline.evaluate(u)
    expected = [0, 0.125, 0.75, 0.125, 0]
    print(basis_vector)
    assert np.allclose(
        basis_vector, expected
    ), f"Expected {expected}, got {basis_vector}"


# Basic Degree-3 Knot Test
## Degree 3 is a cubic wave, so at u=2,
## the basis functions corresponding to intervals
## [0.5,2.5) (a), [1,3) (b) and [1.5,3.5) (c) are non-zero
## and we are in the middle of the "intersection" of (a) and (c) (and peak of (b))
def test_degree_3():
    degree = 3
    knots = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    spline = BSplineBasis(knots, degree)
    u = 2
    basis_vector = spline.evaluate(u)
    expected = [0, 0.1666667, 0.6666667, 0.1666667, 0]
    print(basis_vector)
    assert np.allclose(
        basis_vector, expected
    ), f"Expected {expected}, got {basis_vector}"


# Local Support Test
def test_local_support():
    degree = 3
    knots = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    spline = BSplineBasis(knots, degree)
    u = 1.5
    basis_vector = spline.evaluate(u)
    # For degree 3 with the given knots, only the basis functions corresponding to
    # the intervals [0,2), [0.5,2.5), and [1,3) should be non-zero at u=1.5
    expected_non_zero_indices = [0, 1, 2]
    non_zero_indices = [i for i, val in enumerate(basis_vector) if val > 1e-6]
    assert (
        non_zero_indices == expected_non_zero_indices
    ), f"Expected non-zero indices {expected_non_zero_indices}, \
        got {non_zero_indices}"
