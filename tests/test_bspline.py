import numpy as np

from src.kanet.core.bspline import BSplineBasis


# Test case 1: Basis vector sum equals to 1
def test_basis_vector_sum():
    degree = 3
    knots = [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]
    spline = BSplineBasis(knots, degree)
    u = 2
    basis_vector = spline.evaluate(u)
    assert abs(sum(basis_vector) - 1) < 1e-6, "Basis vector sum does not equal to 1"


# Basic Degree-0 Knot Test
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


"""
# Basic Degree-3 Knot Test
def test_degree_3():
    degree = 3
    knots = [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]
    spline = BSplineBasis(knots, degree)
    u = 1.5
    basis_vector = spline.evaluate(u)
    # TODO: CHECK
    expected = [0, 0.0375, 0.5625, 0.375, 0.0625]
    print(basis_vector)
    assert np.allclose(
        basis_vector, expected
    ), f"Expected {expected}, got {basis_vector}"
"""
