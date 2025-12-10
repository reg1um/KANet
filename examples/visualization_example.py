"""
Example: Visualizing Kolmogorov-Arnold Networks (KANs).

This example demonstrates the visualization utilities for KANs:
1. Network architecture visualization
2. Learned activation function plots
3. B-spline basis function analysis
4. Combined network + activation visualization

These visualizations help understand what the network has learned
and are essential for the interpretability benefits of KANs.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from kanet.core.kanet import KANet
from kanet.utils import (
    plot_network_graph,
    plot_activation_functions,
    plot_all_activations,
    plot_single_activation,
    plot_network_with_activations,
    plot_basis_functions,
    save_visualization,
)


def create_sample_data(n_samples=1000, n_features=2):
    """Create a simple dataset: y = sin(x1) + cos(x2)"""
    x = torch.randn(n_samples, n_features)
    y = torch.sin(x[:, 0]) + torch.cos(x[:, 1])
    y = y.unsqueeze(-1)
    return x, y


def train_model(model, train_loader, n_epochs=100, lr=1e-2, verbose=True):
    """Train the model for a few epochs."""
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if verbose and (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    return model


def example_network_graph():
    """Demonstrate network architecture visualization."""
    print("\n" + "=" * 60)
    print("1. Network Architecture Visualization")
    print("=" * 60)
    
    # Create a model
    model = KANet(
        layers_config=[
            {"in_feat": 2, "out_feat": 4, "num_knots": 8, "degree": 3},
            {"in_feat": 4, "out_feat": 3, "num_knots": 8, "degree": 3},
            {"in_feat": 3, "out_feat": 1, "num_knots": 8, "degree": 3},
        ]
    )
    
    print("Model: 2 -> 4 -> 3 -> 1")
    print("Visualizing network structure...")
    
    # Plot the network graph
    fig = plot_network_graph(
        model,
        figsize=(10, 6),
        title="KAN Architecture: 2 → 4 → 3 → 1",
        edge_width_scale=2.0,
    )
    
    plt.show()
    return fig


def example_activation_functions():
    """Demonstrate activation function visualization after training."""
    print("\n" + "=" * 60)
    print("2. Activation Functions Visualization")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create and train a simple model
    model = KANet(
        layers_config=[
            {"in_feat": 2, "out_feat": 3, "num_knots": 8, "degree": 3},
            {"in_feat": 3, "out_feat": 1, "num_knots": 8, "degree": 3},
        ]
    ).to(device)
    
    # Create dataset
    x_train, y_train = create_sample_data(n_samples=500)
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print("Training model...")
    model = train_model(model, train_loader, n_epochs=100, verbose=True)
    
    # Plot activation functions for layer 0
    print("\nVisualizing activation functions for Layer 0...")
    fig = plot_activation_functions(
        model,
        layer_idx=0,
        input_range=(-2, 2),
        title="Layer 0: Input → Hidden Activations",
    )
    plt.show()
    
    # Plot activation functions for layer 1
    print("Visualizing activation functions for Layer 1...")
    fig = plot_activation_functions(
        model,
        layer_idx=1,
        input_range=(-2, 2),
        title="Layer 1: Hidden → Output Activations",
    )
    plt.show()
    
    return model


def example_single_activation(model):
    """Demonstrate detailed single activation visualization."""
    print("\n" + "=" * 60)
    print("3. Single Activation Function (Detailed)")
    print("=" * 60)
    
    print("Visualizing φ_{0,0} from Layer 0 with components and knots...")
    
    fig = plot_single_activation(
        model,
        layer_idx=0,
        in_idx=0,
        out_idx=0,
        input_range=(-2, 2),
        show_components=True,
        show_knots=True,
        figsize=(10, 7),
    )
    plt.show()
    
    return fig


def example_basis_functions(model):
    """Demonstrate B-spline basis function visualization."""
    print("\n" + "=" * 60)
    print("4. B-spline Basis Functions Analysis")
    print("=" * 60)
    
    print("Visualizing basis functions and coefficients for edge (0→0) in Layer 0...")
    
    fig = plot_basis_functions(
        model,
        layer_idx=0,
        in_idx=0,
        out_idx=0,
        input_range=(-2, 2),
        figsize=(12, 5),
    )
    plt.show()
    
    return fig


def example_network_with_activations(model):
    """Demonstrate combined network + activation visualization."""
    print("\n" + "=" * 60)
    print("5. Network with Inline Activation Functions")
    print("=" * 60)
    
    print("Creating comprehensive visualization with activation functions on edges...")
    
    fig = plot_network_with_activations(
        model,
        input_range=(-2, 2),
        n_points=50,
        figsize=(14, 9),
        activation_size=0.07,
    )
    plt.show()
    
    return fig


def example_all_activations(model):
    """Demonstrate plotting all activations across all layers."""
    print("\n" + "=" * 60)
    print("6. All Activation Functions (All Layers)")
    print("=" * 60)
    
    print("Generating activation plots for all layers...")
    
    figures = plot_all_activations(
        model,
        input_range=(-2, 2),
        figsize_per_cell=(3, 3),
    )
    
    for i, fig in enumerate(figures):
        print(f"  Showing Layer {i} activations...")
        plt.show()
    
    return figures


def example_save_visualizations(model):
    """Demonstrate saving visualizations to files."""
    print("\n" + "=" * 60)
    print("7. Saving Visualizations to Files")
    print("=" * 60)
    
    # Create output directory
    import os
    output_dir = "visualization_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save network graph
    fig1 = plot_network_graph(model, title="Trained KAN Architecture")
    save_visualization(fig1, f"{output_dir}/network_graph.png", dpi=150)
    plt.close(fig1)
    
    # Save activation functions
    fig2 = plot_activation_functions(model, layer_idx=0)
    save_visualization(fig2, f"{output_dir}/layer0_activations.png", dpi=150)
    plt.close(fig2)
    
    # Save combined visualization
    fig3 = plot_network_with_activations(model)
    save_visualization(fig3, f"{output_dir}/network_with_activations.png", dpi=200)
    plt.close(fig3)
    
    # Save as PDF (vector format)
    fig4 = plot_single_activation(model, 0, 0, 0)
    save_visualization(fig4, f"{output_dir}/single_activation.pdf")
    plt.close(fig4)
    
    print(f"\nAll visualizations saved to '{output_dir}/' directory")


def example_custom_subplot():
    """Demonstrate using visualization functions with custom subplot layouts."""
    print("\n" + "=" * 60)
    print("8. Custom Subplot Layout")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create a small model
    model = KANet(
        layers_config=[
            {"in_feat": 2, "out_feat": 2, "num_knots": 8, "degree": 3},
            {"in_feat": 2, "out_feat": 1, "num_knots": 8, "degree": 3},
        ]
    ).to(device)
    
    # Quick training
    x_train, y_train = create_sample_data(n_samples=200)
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    model = train_model(model, train_loader, n_epochs=50, verbose=False)
    
    # Create custom figure with multiple visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: Network graph
    plot_network_graph(model, ax=axes[0, 0], title="Network Structure")
    
    # Top-right: Single activation
    plot_single_activation(model, 0, 0, 0, ax=axes[0, 1], show_knots=True)
    
    # Bottom-left: Another activation
    plot_single_activation(model, 0, 1, 0, ax=axes[1, 0], show_knots=True)
    
    # Bottom-right: Output layer activation
    plot_single_activation(model, 1, 0, 0, ax=axes[1, 1], show_knots=True)
    
    fig.suptitle("KAN Visualization Dashboard", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig


def main():
    """Run all visualization examples."""
    print("=" * 60)
    print("KAN Visualization Examples")
    print("=" * 60)
    
    # Example 1: Basic network graph (untrained model)
    example_network_graph()
    
    # Example 2: Train a model and visualize activations
    trained_model = example_activation_functions()
    
    # Example 3: Detailed single activation
    example_single_activation(trained_model)
    
    # Example 4: Basis function analysis
    example_basis_functions(trained_model)
    
    # Example 5: Network with inline activations
    example_network_with_activations(trained_model)
    
    # Example 6: All activations
    example_all_activations(trained_model)
    
    # Example 7: Save visualizations
    example_save_visualizations(trained_model)
    
    # Example 8: Custom subplot layout
    example_custom_subplot()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
