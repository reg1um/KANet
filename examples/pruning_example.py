"""
Example: Training a KAN with pruning for improved interpretability.

This example demonstrates the pruning pipeline from the KAN paper:
1. Train with sparsification regularization (L1 + entropy)
2. Compute edge/node importance scores
3. Identify nodes to prune
4. Fine-tune the network

Pruning helps discover the minimal network structure needed for a task,
which is a key benefit of KANs for scientific discovery.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kanet.core.kanet import KANet
from kanet.training import KANPruner, KANTrainerWithPruning, SparsityRegularizer


def create_sample_data(n_samples=1000, n_features=2):
    """Create a simple dataset: y = sin(x1) + cos(x2)"""
    x = torch.randn(n_samples, n_features)
    y = torch.sin(x[:, 0]) + torch.cos(x[:, 1])
    y = y.unsqueeze(-1)
    return x, y


def main():
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create dataset
    x_train, y_train = create_sample_data(n_samples=1000)
    x_test, y_test = create_sample_data(n_samples=200)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Create KAN model: 2 inputs -> 5 hidden -> 5 hidden -> 1 output
    # We intentionally use more neurons than needed to demonstrate pruning
    model = KANet(
        layers_config=[
            {"in_feat": 2, "out_feat": 5, "num_knots": 8, "degree": 3},
            {"in_feat": 5, "out_feat": 5, "num_knots": 8, "degree": 3},
            {"in_feat": 5, "out_feat": 1, "num_knots": 8, "degree": 3},
        ]
    ).to(device)

    print(f"Model architecture: 2 -> 5 -> 5 -> 1")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Create trainer with pruning support
    trainer = KANTrainerWithPruning(
        model=model,
        optimizer=optimizer,
        device=device,
        lambda_l1=0.01,      # L1 regularization weight
        lambda_entropy=1.0,   # Entropy regularization weight
    )

    # Train with pruning pipeline
    pruning_mask = trainer.train_with_pruning(
        train_loader=train_loader,
        criterion=nn.MSELoss(),
        n_epochs_before_prune=100,
        n_epochs_after_prune=50,
        sparsity_weight=0.1,
        prune_threshold_percentile=20,  # Prune bottom 20% of nodes
        verbose=True,
    )

    # Evaluate final model
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x)
            test_loss += nn.MSELoss()(pred, batch_y).item()
        test_loss /= len(test_loader)
    
    print(f"\nFinal test loss: {test_loss:.4f}")

    # Print pruning recommendations
    print("\n" + "=" * 50)
    print("Pruning Recommendations")
    print("=" * 50)
    for layer_idx, mask in pruning_mask.items():
        keep_indices = torch.where(mask)[0].tolist()
        prune_indices = torch.where(~mask)[0].tolist()
        print(f"Layer {layer_idx}:")
        print(f"  Keep neurons: {keep_indices}")
        print(f"  Prune neurons: {prune_indices}")


def standalone_pruning_example():
    """
    Example of using the pruning utilities standalone (without the trainer).
    Useful if you have a pre-trained model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create a pre-trained model (in practice, you'd load your trained model)
    model = KANet(
        layers_config=[
            {"in_feat": 2, "out_feat": 4, "num_knots": 8, "degree": 3},
            {"in_feat": 4, "out_feat": 1, "num_knots": 8, "degree": 3},
        ]
    ).to(device)

    # Create sample data
    x, y = create_sample_data(n_samples=500)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32)

    # Create pruner
    pruner = KANPruner(model, device=device)

    # Compute importance scores
    print("Computing edge importance scores...")
    edge_scores = pruner.compute_edge_scores(dataloader)

    print("Computing node importance scores...")
    node_scores = pruner.compute_node_scores()

    # Print summary
    pruner.print_importance_summary()

    # Get pruning mask
    print("\nGetting pruning mask (prune bottom 25%)...")
    mask = pruner.get_pruning_mask(threshold_percentile=25)

    return mask


def sparsity_loss_example():
    """
    Example of using sparsity regularization in a custom training loop.
    """
    device = "cuda"
    
    model = KANet(
        layers_config=[
            {"in_feat": 2, "out_feat": 3, "num_knots": 8, "degree": 3},
            {"in_feat": 3, "out_feat": 1, "num_knots": 8, "degree": 3},
        ]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    
    # Create sparsity regularizer
    regularizer = SparsityRegularizer(lambda_l1=0.01, lambda_entropy=1.0)

    # Create data
    x, y = create_sample_data(n_samples=100)
    x, y = x.to(device), y.to(device)

    # Training loop with sparsity
    for epoch in range(10):
        optimizer.zero_grad()
        
        output = model(x)
        task_loss = criterion(output, y)
        
        # Add sparsity regularization
        sparsity_loss = regularizer(model, x)
        
        total_loss = task_loss + 0.1 * sparsity_loss
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Task={task_loss.item():.4f}, "
                  f"Sparsity={sparsity_loss.item():.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Full Training with Pruning Example")
    print("=" * 60)
    main()
    
    print("\n" + "=" * 60)
    print("Standalone Pruning Example")
    print("=" * 60)
    standalone_pruning_example()
    
    print("\n" + "=" * 60)
    print("Sparsity Regularization Example")
    print("=" * 60)
    sparsity_loss_example()
