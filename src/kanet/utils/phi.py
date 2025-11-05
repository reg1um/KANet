import torch

# KAN phi functions


def phi(x, bspline_basis, coeffs, base_activation, base_weights, verbose=False):
    if verbose:
        print(f"phi({x.item():.4f})")

    if x.dim() == 0:
        x = x.unsqueeze(0)

    """
    basis_values = torch.stack([
        bspline_basis.evaluate(x_val)
        for x_val in x
    ]).to(x.device)  # Shape: (batch_size, num_basis_functions)
    """

    basis_values = bspline_basis.evaluate(x.squeeze(-1))

    spline_output = torch.einsum("bn,n->b", basis_values, coeffs)

    spline_output += base_activation(x) * base_weights
    return spline_output


"""
    if x.dim() == 1:
        x = x.squeeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = x.shape[0]
    outputs = torch.zeros(batch_size, device=x.device)

    for b in range(batch_size):

        x_val = x[b]

        # Go through the B-spline basis to get basis values at x
        basis_eval = bspline_basis.evaluate(x_val)
        basis_values = basis_eval.to(x.device)

        if verbose:
            print(f"  basis values: {basis_values}")
            print(f"  coeffs: {coeffs}")

        # basis_values * coeffs part
        spline_output = torch.dot(basis_values, coeffs)

        # base activation part
        spline_output += base_activation(x_val) * base_weights

        outputs[b] = spline_output

        if verbose:
            print(f"  spline output: {spline_output.item():.4f}")

    if squeeze_output:
        outputs = outputs.squeeze(0)

    return outputs
    """
