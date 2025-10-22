import torch

# KAN phi functions


def phi(x, bspline_basis, coeffs, base_activation, base_weights, verbose=False):
    if verbose:
        print(f"phi({x.item():.4f})")

    # Go through the B-spline basis to get basis values at x
    basis_eval = bspline_basis.evaluate(x.cpu())
    basis_values = basis_eval.to(
        x.device
    )  # torch.from_numpy(basis_eval).to(dtype=torch.float32).to(x.device)

    if verbose:
        print(f"  basis values: {basis_values}")
        print(f"  coeffs: {coeffs}")

    # basis_values * coeffs part
    spline_output = torch.dot(basis_values, coeffs)

    # base activation part
    spline_output += base_activation(x) * base_weights

    if verbose:
        print(f"  spline output: {spline_output.item():.4f}")

    return spline_output
