"""
Visualization utilities for Kolmogorov-Arnold Networks (KANs).

This module provides functions to visualize:
1. The network architecture as a graph
2. The learned B-spline activation functions for each edge
3. Combined visualizations showing both structure and functions
4. Support for pruning masks to show only active nodes/edges
"""

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle


def _get_active_nodes(
    model,
    pruning_mask: dict[int, torch.Tensor] | None = None,
) -> dict[int, list[int]]:
    """
    Get the list of active (non-pruned) node indices for each layer.

    Args:
        model: A KANet model instance
        pruning_mask: Dictionary mapping layer index to boolean mask tensor.
                     True = keep node, False = prune node.
                     Layer indices refer to hidden layers (1, 2, ..., n_layers-1).
                     Input (layer 0) and output (last layer) are always kept.

    Returns:
        Dictionary mapping layer index to list of active node indices
    """
    n_layers = len(model.layers) + 1  # +1 for output layer
    active_nodes = {}

    # Input layer (layer 0) - always all active
    active_nodes[0] = list(range(model.layers[0].in_feat))

    # Hidden and output layers
    for layer_idx in range(1, n_layers):
        if layer_idx == n_layers - 1:
            # Output layer - always all active
            n_nodes = model.layers[-1].out_feat
            active_nodes[layer_idx] = list(range(n_nodes))
        else:
            # Hidden layer
            n_nodes = model.layers[layer_idx - 1].out_feat
            if pruning_mask is not None and layer_idx in pruning_mask:
                mask = pruning_mask[layer_idx]
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                active_nodes[layer_idx] = [i for i in range(n_nodes) if mask[i]]
            else:
                active_nodes[layer_idx] = list(range(n_nodes))

    return active_nodes


def plot_network_graph(
    model,
    figsize: tuple[int, int] = (12, 8),
    node_size: int = 800,
    title: str = "KAN Network Architecture",
    show_labels: bool = True,
    edge_width_scale: float = 2.0,
    cmap: str = "coolwarm",
    ax: Axes | None = None,
    pruning_mask: dict[int, torch.Tensor] | None = None,
) -> Figure:
    """
    Plot the KAN network architecture as a graph.

    Each layer is displayed as a column of nodes, with edges connecting
    nodes between consecutive layers. Edge widths and colors represent
    the magnitude of the learned weights.

    Args:
        model: A KANet model instance
        figsize: Figure size (width, height)
        node_size: Size of the nodes in the plot
        title: Title of the plot
        show_labels: Whether to show node labels
        edge_width_scale: Scale factor for edge widths
        cmap: Colormap for edge colors
        ax: Optional matplotlib axes to plot on
        pruning_mask: Optional dict mapping layer index to boolean mask.
                     If provided, only active (non-pruned) nodes are shown.

    Returns:
        matplotlib Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Get active nodes for each layer
    active_nodes = _get_active_nodes(model, pruning_mask)

    # Extract layer dimensions (only active nodes)
    layer_dims = [len(active_nodes[i]) for i in range(len(model.layers) + 1)]

    n_layers = len(layer_dims)
    max_nodes = max(layer_dims) if layer_dims else 1

    # Calculate node positions (only for active nodes)
    positions = {}
    for layer_idx, active_list in active_nodes.items():
        n_active = len(active_list)
        x = layer_idx / (n_layers - 1) if n_layers > 1 else 0.5
        for pos_idx, node_idx in enumerate(active_list):
            # Center nodes vertically based on active count
            y = (pos_idx - (n_active - 1) / 2) / max(max_nodes - 1, 1)
            positions[(layer_idx, node_idx)] = (x, y)

    # Draw edges with weights from the model
    colormap = plt.get_cmap(cmap)

    # Collect weights only for active edges
    all_weights = []
    for layer_idx, layer in enumerate(model.layers):
        coeffs = layer.coeffs.detach().cpu().numpy()
        active_in = active_nodes[layer_idx]
        active_out = active_nodes[layer_idx + 1]
        for in_idx in active_in:
            for out_idx in active_out:
                all_weights.append(np.abs(coeffs[in_idx, out_idx]).mean())

    if all_weights:
        weight_max = np.max(all_weights)
        weight_min = np.min(all_weights)
    else:
        weight_max, weight_min = 1.0, 0.0

    for layer_idx, layer in enumerate(model.layers):
        coeffs = layer.coeffs.detach().cpu()
        edge_weights = torch.abs(coeffs).mean(dim=-1).numpy()

        active_in = active_nodes[layer_idx]
        active_out = active_nodes[layer_idx + 1]

        for in_idx in active_in:
            for out_idx in active_out:
                start_pos = positions[(layer_idx, in_idx)]
                end_pos = positions[(layer_idx + 1, out_idx)]

                weight = edge_weights[in_idx, out_idx]

                if weight_max > weight_min:
                    norm_weight = (weight - weight_min) / (weight_max - weight_min)
                else:
                    norm_weight = 0.5

                color = colormap(norm_weight)
                line_width = 0.5 + edge_width_scale * norm_weight

                ax.plot(
                    [start_pos[0], end_pos[0]],
                    [start_pos[1], end_pos[1]],
                    color=color,
                    linewidth=line_width,
                    alpha=0.7,
                    zorder=1,
                )

    # Draw nodes (only active ones)
    for layer_idx, active_list in active_nodes.items():
        for node_idx in active_list:
            pos = positions[(layer_idx, node_idx)]

            if layer_idx == 0:
                color = "#4CAF50"  # Green for input
            elif layer_idx == n_layers - 1:
                color = "#2196F3"  # Blue for output
            else:
                color = "#FF9800"  # Orange for hidden

            circle = Circle(
                pos, radius=0.03, color=color, ec="black", linewidth=2, zorder=2
            )
            ax.add_patch(circle)

            if show_labels:
                ax.annotate(
                    f"{node_idx}",
                    pos,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white",
                    zorder=3,
                )

    # Add layer labels
    for layer_idx in range(n_layers):
        n_active = len(active_nodes[layer_idx])
        x = layer_idx / (n_layers - 1) if n_layers > 1 else 0.5
        if layer_idx == 0:
            label = f"Input\n({n_active})"
        elif layer_idx == n_layers - 1:
            label = f"Output\n({n_active})"
        else:
            label = f"Hidden {layer_idx}\n({n_active})"

        ax.text(
            x,
            -0.6,
            label,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
        )

    # Add legend
    legend_patches = [
        mpatches.Patch(color="#4CAF50", label="Input"),
        mpatches.Patch(color="#FF9800", label="Hidden"),
        mpatches.Patch(color="#2196F3", label="Output"),
    ]
    ax.legend(handles=legend_patches, loc="upper right")

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.8, 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_activation_functions(
    model,
    layer_idx: int,
    input_range: tuple[float, float] = (-1, 1),
    n_points: int = 200,
    figsize: tuple[int, int] | None = None,
    title: str | None = None,
    pruning_mask: dict[int, torch.Tensor] | None = None,
) -> Figure:
    """
    Plot all learned activation functions (φ_{j,i}) for a specific layer.

    Each subplot shows the activation function for one edge (input j -> output i).

    Args:
        model: A KANet model instance
        layer_idx: Index of the layer to visualize (0-indexed)
        input_range: Range of input values to plot
        n_points: Number of points to sample for plotting
        figsize: Figure size (width, height), auto-calculated if None
        title: Title for the figure
        pruning_mask: Optional dict mapping layer index to boolean mask.
                     If provided, only active (non-pruned) edges are shown.

    Returns:
        matplotlib Figure object
    """
    if layer_idx < 0 or layer_idx >= len(model.layers):
        raise ValueError(f"layer_idx must be in [0, {len(model.layers) - 1}]")

    layer = model.layers[layer_idx]

    # Get active nodes
    active_nodes = _get_active_nodes(model, pruning_mask)
    active_in = active_nodes[layer_idx]
    active_out = active_nodes[layer_idx + 1]

    n_active_in = len(active_in)
    n_active_out = len(active_out)

    if n_active_in == 0 or n_active_out == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No active edges in this layer", ha="center", va="center")
        ax.axis("off")
        return fig

    if figsize is None:
        figsize = (3 * n_active_out, 3 * n_active_in)

    fig, axes = plt.subplots(n_active_in, n_active_out, figsize=figsize, squeeze=False)

    x_vals = torch.linspace(input_range[0], input_range[1], n_points)
    device = next(model.parameters()).device
    x_vals = x_vals.to(device)

    with torch.no_grad():
        for row_idx, j in enumerate(active_in):
            for col_idx, i in enumerate(active_out):
                ax = axes[row_idx, col_idx]

                bspline = layer.bspline_array[j][i]
                coeffs = layer.coeffs[j, i]
                base_weight = layer.base_weights[j, i]
                base_activation = layer.base_activation

                basis_values = bspline.evaluate(x_vals)
                spline_output = torch.einsum("bn,n->b", basis_values, coeffs)
                base_output = base_activation(x_vals) * base_weight
                total_output = spline_output + base_output

                x_np = x_vals.cpu().numpy()
                y_spline = spline_output.cpu().numpy()
                y_base = base_output.cpu().numpy()
                y_total = total_output.cpu().numpy()

                ax.plot(
                    x_np, y_spline, "b-", label="B-spline", alpha=0.7, linewidth=1.5
                )
                ax.plot(
                    x_np, y_base, "g--", label="Base (SiLU)", alpha=0.7, linewidth=1.5
                )
                ax.plot(x_np, y_total, "r-", label="Total φ", linewidth=2)

                ax.set_xlabel("x", fontsize=9)
                ax.set_ylabel("φ(x)", fontsize=9)
                ax.set_title(f"φ_{{{j},{i}}}", fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color="k", linewidth=0.5, alpha=0.5)
                ax.axvline(x=0, color="k", linewidth=0.5, alpha=0.5)

                if row_idx == 0 and col_idx == 0:
                    ax.legend(fontsize=7, loc="best")

    if title is None:
        title = f"Activation Functions - Layer {layer_idx}"
        if pruning_mask:
            title += " (pruned)"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    return fig


def plot_all_activations(
    model,
    input_range: tuple[float, float] = (-1, 1),
    n_points: int = 200,
    figsize_per_cell: tuple[float, float] = (2.5, 2.5),
    pruning_mask: dict[int, torch.Tensor] | None = None,
) -> list[Figure]:
    """
    Plot activation functions for all layers in the network.

    Args:
        model: A KANet model instance
        input_range: Range of input values to plot
        n_points: Number of points to sample for plotting
        figsize_per_cell: Size of each subplot cell
        pruning_mask: Optional dict mapping layer index to boolean mask.

    Returns:
        List of matplotlib Figure objects, one per layer
    """
    active_nodes = _get_active_nodes(model, pruning_mask)

    figures = []
    for layer_idx in range(len(model.layers)):
        n_active_in = len(active_nodes[layer_idx])
        n_active_out = len(active_nodes[layer_idx + 1])

        figsize = (
            figsize_per_cell[0] * max(n_active_out, 1),
            figsize_per_cell[1] * max(n_active_in, 1),
        )
        fig = plot_activation_functions(
            model,
            layer_idx,
            input_range=input_range,
            n_points=n_points,
            figsize=figsize,
            pruning_mask=pruning_mask,
        )
        figures.append(fig)

    return figures


def plot_single_activation(
    model,
    layer_idx: int,
    in_idx: int,
    out_idx: int,
    input_range: tuple[float, float] = (-1, 1),
    n_points: int = 200,
    figsize: tuple[int, int] = (8, 6),
    show_components: bool = True,
    show_knots: bool = True,
    ax: Axes | None = None,
) -> Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    layer = model.layers[layer_idx]

    if in_idx >= layer.in_feat or out_idx >= layer.out_feat:
        raise ValueError(
            f"Invalid indices: in_idx={in_idx}, out_idx={out_idx}. "
            f"Layer has in_feat={layer.in_feat}, out_feat={layer.out_feat}"
        )

    device = next(model.parameters()).device
    x_vals = torch.linspace(input_range[0], input_range[1], n_points).to(device)

    with torch.no_grad():
        bspline = layer.bspline_array[in_idx][out_idx]
        coeffs = layer.coeffs[in_idx, out_idx]
        base_weight = layer.base_weights[in_idx, out_idx]
        base_activation = layer.base_activation

        basis_values = bspline.evaluate(x_vals)
        spline_output = torch.einsum("bn,n->b", basis_values, coeffs)
        base_output = base_activation(x_vals) * base_weight
        total_output = spline_output + base_output

        x_np = x_vals.cpu().numpy()
        y_spline = spline_output.cpu().numpy()
        y_base = base_output.cpu().numpy()
        y_total = total_output.cpu().numpy()

    if show_components:
        ax.fill_between(
            x_np, 0, y_spline, alpha=0.2, color="blue", label="B-spline component"
        )
        ax.fill_between(
            x_np, 0, y_base, alpha=0.2, color="green", label="Base component (SiLU)"
        )
        ax.plot(x_np, y_spline, "b--", linewidth=1.5, alpha=0.7)
        ax.plot(x_np, y_base, "g--", linewidth=1.5, alpha=0.7)

    ax.plot(x_np, y_total, "r-", linewidth=2.5, label="Total φ(x)")

    if show_knots:
        knots = layer.knots.cpu().numpy()
        unique_knots = np.unique(knots)
        for knot in unique_knots:
            if input_range[0] <= knot <= input_range[1]:
                ax.axvline(x=knot, color="gray", linestyle=":", alpha=0.5, linewidth=1)

    ax.axhline(y=0, color="k", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("φ(x)", fontsize=12)
    ax.set_title(
        f"Activation Function φ_{{{in_idx},{out_idx}}} (Layer {layer_idx})",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


def plot_network_with_activations(
    model,
    input_range: tuple[float, float] = (-1, 1),
    n_points: int = 100,
    figsize: tuple[int, int] = (16, 10),
    activation_size: float = 0.08,
    pruning_mask: dict[int, torch.Tensor] | None = None,
) -> Figure:
    """
    Plot the network architecture with small activation function plots on each edge.

    This provides a comprehensive visualization showing both the network structure
    and the learned functions simultaneously.

    Args:
        model: A KANet model instance
        input_range: Range of input values for activation plots
        n_points: Number of points for activation curves
        figsize: Figure size
        activation_size: Size of inset activation plots (as fraction of figure)
        pruning_mask: Optional dict mapping layer index to boolean mask.
                     If provided, only active (non-pruned) nodes/edges are shown.

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get active nodes for each layer
    active_nodes = _get_active_nodes(model, pruning_mask)

    # Extract layer dimensions (only active nodes)
    layer_dims = [len(active_nodes[i]) for i in range(len(model.layers) + 1)]

    n_layers = len(layer_dims)
    max_nodes = max(layer_dims) if layer_dims else 1

    # Calculate node positions (only for active nodes)
    positions = {}
    for layer_idx, active_list in active_nodes.items():
        n_active = len(active_list)
        x = 0.1 + 0.8 * layer_idx / (n_layers - 1) if n_layers > 1 else 0.5
        for pos_idx, node_idx in enumerate(active_list):
            y = 0.5 + 0.35 * (pos_idx - (n_active - 1) / 2) / max(max_nodes - 1, 1)
            positions[(layer_idx, node_idx)] = (x, y)

    device = next(model.parameters()).device
    x_vals = torch.linspace(input_range[0], input_range[1], n_points).to(device)

    # Draw edges and activation functions (only for active nodes)
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.layers):
            active_in = active_nodes[layer_idx]
            active_out = active_nodes[layer_idx + 1]

            for in_idx in active_in:
                for out_idx in active_out:
                    start_pos = positions[(layer_idx, in_idx)]
                    end_pos = positions[(layer_idx + 1, out_idx)]

                    # Draw edge
                    ax.plot(
                        [start_pos[0], end_pos[0]],
                        [start_pos[1], end_pos[1]],
                        color="lightgray",
                        linewidth=1,
                        alpha=0.5,
                        zorder=1,
                    )

                    # Compute activation function
                    bspline = layer.bspline_array[in_idx][out_idx]
                    coeffs = layer.coeffs[in_idx, out_idx]
                    base_weight = layer.base_weights[in_idx, out_idx]
                    base_activation = layer.base_activation

                    basis_values = bspline.evaluate(x_vals)
                    spline_output = torch.einsum("bn,n->b", basis_values, coeffs)
                    base_output = base_activation(x_vals) * base_weight
                    total_output = spline_output + base_output

                    x_np = x_vals.cpu().numpy()
                    y_np = total_output.cpu().numpy()

                    # Normalize for small inset plot
                    y_range = y_np.max() - y_np.min() if y_np.max() != y_np.min() else 1
                    y_norm = (y_np - y_np.min()) / y_range
                    x_norm = (x_np - x_np.min()) / (x_np.max() - x_np.min())

                    # Position inset at midpoint of edge
                    mid_x = (start_pos[0] + end_pos[0]) / 2
                    mid_y = (start_pos[1] + end_pos[1]) / 2

                    # Scale and offset for inset
                    inset_x = mid_x - activation_size / 2 + x_norm * activation_size
                    inset_y = mid_y - activation_size / 2 + y_norm * activation_size

                    # Draw inset border first (lower zorder)
                    rect = Rectangle(
                        (mid_x - activation_size / 2, mid_y - activation_size / 2),
                        activation_size,
                        activation_size,
                        fill=True,
                        facecolor="white",
                        edgecolor="gray",
                        linewidth=0.5,
                        alpha=0.9,
                        zorder=2,
                    )
                    ax.add_patch(rect)

                    # Draw activation function
                    ax.plot(inset_x, inset_y, "b-", linewidth=1, zorder=3)

    # Draw nodes (only active ones)
    for layer_idx, active_list in active_nodes.items():
        for node_idx in active_list:
            pos = positions[(layer_idx, node_idx)]

            if layer_idx == 0:
                color = "#4CAF50"
            elif layer_idx == n_layers - 1:
                color = "#2196F3"
            else:
                color = "#FF9800"

            circle = Circle(
                pos, radius=0.02, color=color, ec="black", linewidth=2, zorder=4
            )
            ax.add_patch(circle)

            ax.annotate(
                f"{node_idx}",
                pos,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
                zorder=5,
            )

    # Layer labels
    for layer_idx in range(n_layers):
        n_active = len(active_nodes[layer_idx])
        x = 0.1 + 0.8 * layer_idx / (n_layers - 1) if n_layers > 1 else 0.5
        if layer_idx == 0:
            label = f"Input ({n_active})"
        elif layer_idx == n_layers - 1:
            label = f"Output ({n_active})"
        else:
            label = f"Hidden {layer_idx} ({n_active})"
        ax.text(x, 0.05, label, ha="center", fontsize=10, fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    title = "KAN Network with Learned Activation Functions"
    if pruning_mask:
        title += " (pruned)"
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_basis_functions(
    model,
    layer_idx: int,
    in_idx: int,
    out_idx: int,
    input_range: tuple[float, float] | None = None,
    n_points: int = 200,
    figsize: tuple[int, int] = (10, 6),
) -> Figure:
    layer = model.layers[layer_idx]
    bspline = layer.bspline_array[in_idx][out_idx]
    coeffs = layer.coeffs[in_idx, out_idx].detach().cpu()
    knots = layer.knots.cpu().numpy()

    if input_range is None:
        input_range = (layer.grid_min, layer.grid_max)

    device = next(model.parameters()).device
    x_vals = torch.linspace(input_range[0], input_range[1], n_points).to(device)

    with torch.no_grad():
        basis_values = bspline.evaluate(x_vals).cpu().numpy()

    x_np = x_vals.cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    n_basis = basis_values.shape[1]
    colors = cm.get_cmap("viridis")(np.linspace(0, 1, n_basis))

    for i in range(n_basis):
        ax1.plot(x_np, basis_values[:, i], color=colors[i], label=f"B_{i}", alpha=0.7)

    basis_sum = basis_values.sum(axis=1)
    ax1.plot(x_np, basis_sum, "k--", linewidth=2, label="Sum", alpha=0.8)

    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("B(x)", fontsize=12)
    ax1.set_title("B-spline Basis Functions", fontsize=12)
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    unique_knots = np.unique(knots)
    for knot in unique_knots:
        if input_range[0] <= knot <= input_range[1]:
            ax1.axvline(x=knot, color="gray", linestyle=":", alpha=0.5)

    coeffs_np = coeffs.numpy()
    weighted_sum = np.zeros_like(x_np)

    for i in range(n_basis):
        weighted_basis = basis_values[:, i] * coeffs_np[i]
        ax2.plot(
            x_np,
            weighted_basis,
            color=colors[i],
            alpha=0.5,
            label=f"c_{i}={coeffs_np[i]:.2f}",
        )
        weighted_sum += weighted_basis

    ax2.plot(x_np, weighted_sum, "r-", linewidth=2.5, label="Spline output")

    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("c·B(x)", fontsize=12)
    ax2.set_title("Weighted Basis Functions", fontsize=12)
    ax2.legend(loc="best", fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="k", linewidth=0.5)

    fig.suptitle(
        f"Basis Function Analysis: Layer {layer_idx}, Edge ({in_idx}→{out_idx})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    return fig


def save_visualization(
    fig: Figure,
    filepath: str,
    dpi: int = 150,
    transparent: bool = False,
) -> None:
    fig.savefig(filepath, dpi=dpi, transparent=transparent, bbox_inches="tight")
    print(f"Saved visualization to {filepath}")
