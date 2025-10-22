import torch

# KAN phi functions


def phi(x, bspline_basis, coeffs, base_activation, base_weights):
    # Go through the B-spline basis to get basis values at x
    basis_eval = bspline_basis.evaluate(x.cpu().numpy())
    basis_values = torch.from_numpy(basis_eval).to(x.device)

    # basis_values * coeffs part
    spline_output = torch.dot(basis_values, coeffs)

    # base activation part
    spline_output += base_activation(x) * base_weights

    return spline_output
