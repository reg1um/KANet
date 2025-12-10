"""
Pruning utilities for Kolmogorov-Arnold Networks (KANs).

Based on the KAN paper, pruning is crucial for improving interpretability.
The approach combines sparsification regularization followed by pruning
of unimportant nodes/edges based on activation magnitudes.
"""

import torch
import torch.nn as nn


class SparsityRegularizer:
    """
    Implements sparsification regularization from the KAN paper.

    Uses L1 norm of spline activations to encourage sparse activation patterns,
    combined with entropy regularization to make the activation distribution
    more peaked (sparser).

    Args:
        lambda_l1: Weight for L1 regularization on activations
        lambda_entropy: Weight for entropy regularization
    """

    def __init__(self, lambda_l1: float = 0.01, lambda_entropy: float = 1.0):
        self.lambda_l1 = lambda_l1
        self.lambda_entropy = lambda_entropy

    def compute_regularization_from_activations(
        self, all_activations: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute L1 and entropy regularization from pre-computed activations.

        This is the optimized version that reuses activations computed during
        the forward pass, avoiding redundant computation.

        Args:
            all_activations: List of activation tensors, one per layer,
                           each of shape (batch, in_feat, out_feat)

        Returns:
            Tuple of (l1_loss, entropy_loss)
        """
        device = all_activations[0].device
        l1_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)

        for activations in all_activations:
            # L1 loss: mean of absolute activations
            l1_loss = l1_loss + torch.abs(activations).mean()

            # Entropy loss
            act_magnitudes = torch.abs(activations).mean(dim=0)  # (in_feat, out_feat)
            total = act_magnitudes.sum() + 1e-10
            probs = act_magnitudes / total
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            entropy_loss = entropy_loss + entropy

        return l1_loss, entropy_loss

    def __call__(
        self,
        model: nn.Module,
        x: torch.Tensor,
        precomputed_activations: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Compute total sparsification loss combining L1 and entropy.

        Args:
            model: KANet model
            x: Input tensor
            precomputed_activations: Optional pre-computed activations to avoid
                                    redundant forward passes

        Returns:
            Combined sparsification loss
        """
        if precomputed_activations is not None:
            l1, entropy = self.compute_regularization_from_activations(
                precomputed_activations
            )
        else:
            # Fallback to computing activations (slower path)
            all_activations = []
            current = x
            for layer in model.layers:
                activations = layer.get_activations(current)
                all_activations.append(activations)
                current = layer(current)
            l1, entropy = self.compute_regularization_from_activations(all_activations)

        return self.lambda_l1 * l1 + self.lambda_entropy * entropy


def forward_with_activations(
    model: nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Forward pass that also returns all layer activations.

    This is the key optimization: compute output AND activations in a single pass.

    Args:
        model: KANet model
        x: Input tensor

    Returns:
        Tuple of (output, list of activation tensors)
    """
    all_activations = []
    current = x

    for layer in model.layers:
        # Get activations for this layer (before summing)
        activations = layer.get_activations(current)
        all_activations.append(activations)
        # Sum activations to get layer output (this is what forward() does)
        current = activations.sum(dim=1)

    return current, all_activations


class KANPruner:
    """
    Pruning utilities for KAN networks.

    Computes edge and node importance scores based on activation magnitudes,
    then prunes unimportant nodes to create a smaller, more interpretable network.

    Args:
        model: KANet model to prune
        device: Device to use for computations
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.edge_scores: dict[int, torch.Tensor] = {}
        self.node_scores: dict[int, torch.Tensor] = {}

    def compute_edge_scores(
        self, dataloader: torch.utils.data.DataLoader
    ) -> dict[int, torch.Tensor]:
        """
        Compute edge importance scores based on activation magnitudes.

        Score of edge (l, j, i) = mean(|φ_{l,j,i}(x)|) over dataset

        Args:
            dataloader: DataLoader providing input samples

        Returns:
            Dictionary mapping layer index to edge score tensor
            of shape (in_feat, out_feat)
        """
        self.model.eval()
        edge_scores = {}

        # Initialize accumulators for each layer
        for layer_idx, layer in enumerate(self.model.layers):
            n_in, n_out = layer.in_feat, layer.out_feat
            edge_scores[layer_idx] = torch.zeros(n_in, n_out, device=self.device)

        n_samples = 0
        with torch.no_grad():
            for batch in dataloader:
                # Handle both (x,) and (x, y) formats
                if isinstance(batch, list | tuple):
                    batch_x = batch[0]
                else:
                    batch_x = batch

                batch_x = batch_x.to(self.device)
                # Use optimized forward pass
                _, activations = forward_with_activations(self.model, batch_x)

                for layer_idx, act in enumerate(activations):
                    # act shape: (batch, in_feat, out_feat)
                    edge_scores[layer_idx] += torch.abs(act).sum(dim=0)

                n_samples += batch_x.shape[0]

        # Normalize by number of samples
        for layer_idx in edge_scores:
            edge_scores[layer_idx] /= n_samples

        self.edge_scores = edge_scores
        return edge_scores

    def _get_layer_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass that captures individual spline activations.

        Args:
            x: Input tensor

        Returns:
            List of tensors, each of shape (batch, in_feat, out_feat)
        """
        _, activations = forward_with_activations(self.model, x)
        return activations

    def compute_node_scores(self) -> dict[int, torch.Tensor]:
        """
        Compute node importance scores.

        Node score = max(incoming_score, outgoing_score)
        - Incoming score I_{l,i} = Σ_j |φ_{l-1,j,i}|
        - Outgoing score O_{l,i} = Σ_k |φ_{l,i,k}|

        Returns:
            Dictionary mapping layer index to node score tensor
        """
        if not self.edge_scores:
            raise ValueError("Call compute_edge_scores first")

        node_scores = {}
        n_layers = len(self.model.layers)

        for layer_idx in range(n_layers + 1):
            if layer_idx == 0:
                # Input layer: only outgoing edges matter
                outgoing = self.edge_scores[0].sum(dim=1)  # Sum over output dim
                node_scores[layer_idx] = outgoing
            elif layer_idx == n_layers:
                # Output layer: only incoming edges matter
                incoming = self.edge_scores[layer_idx - 1].sum(
                    dim=0
                )  # Sum over input dim
                node_scores[layer_idx] = incoming
            else:
                # Hidden layers: max of incoming and outgoing
                incoming = self.edge_scores[layer_idx - 1].sum(dim=0)  # (n_hidden,)
                outgoing = self.edge_scores[layer_idx].sum(dim=1)  # (n_hidden,)
                node_scores[layer_idx] = torch.max(incoming, outgoing)

        self.node_scores = node_scores
        return node_scores

    def get_pruning_mask(
        self,
        threshold: float | None = None,
        threshold_percentile: float | None = None,
        threshold_ratio: float | None = None,
        threshold_gap: float | None = None,
    ) -> dict[int, torch.Tensor]:
        """
        Determine which nodes to keep based on importance scores.

        Four threshold modes (in order of priority):
        1. threshold: Absolute threshold value. Nodes with scores below this are pruned.
        2. threshold_gap: Adaptive gap-based threshold. Finds the largest relative gap
           in sorted importance scores and prunes nodes below that gap.
           Value is the minimum gap ratio to consider significant.
           (e.g., 0.3 = 30% drop).
           This is the RECOMMENDED mode for unknown functions.
        3. threshold_ratio: Dynamic threshold relative to max importance.
           Prunes nodes with score < threshold_ratio * max_score.
        4. threshold_percentile: Prune the bottom X% of nodes (fixed percentage).

        If none specified, defaults to threshold_gap=0.3

        Args:
            threshold: Absolute threshold value.
            threshold_gap: Minimum relative gap to consider significant (0.0 to 1.0).
                          Recommended: 0.3-0.5. Finds natural "elbow" in importance.
            threshold_ratio: Ratio of max importance (0.0 to 1.0). Recommended: 0.05-0.2
            threshold_percentile: Percentile of nodes to prune (0-100).

        Returns:
            Dictionary mapping layer index to boolean mask (True = keep, False = prune)
        """
        if not self.node_scores:
            self.compute_node_scores()

        # Collect all hidden layer scores
        hidden_scores = []
        for layer_idx in range(1, len(self.model.layers)):
            hidden_scores.append(self.node_scores[layer_idx])

        if not hidden_scores:
            return {}

        all_scores = torch.cat(hidden_scores)
        max_score = all_scores.max().item()

        # Determine threshold based on mode (priority:
        # absolute > gap > ratio > percentile)
        if threshold is not None:
            mode = "absolute"
            thresh_value = threshold
        elif threshold_gap is not None:
            mode = "gap"
            thresh_value = self._compute_gap_threshold(all_scores, threshold_gap)
        elif threshold_ratio is not None:
            mode = "ratio"
            thresh_value = threshold_ratio * max_score
        elif threshold_percentile is not None:
            mode = "percentile"
            thresh_value = torch.quantile(
                all_scores, threshold_percentile / 100.0
            ).item()
        else:
            # Default to gap-based with 0.3 minimum gap
            mode = "gap"
            threshold_gap = 0.3
            thresh_value = self._compute_gap_threshold(all_scores, threshold_gap)

        pruning_mask = {}
        total_kept = 0
        total_nodes = 0

        for layer_idx in range(1, len(self.model.layers)):
            mask = (
                self.node_scores[layer_idx] >= thresh_value
            )  # Changed > to >= to be inclusive
            pruning_mask[layer_idx] = mask

            n_kept = mask.sum().item()
            n_total = mask.numel()
            total_kept += n_kept
            total_nodes += n_total

            print(f"  Layer {layer_idx}: keeping {n_kept}/{n_total} nodes")

        # Print summary based on mode
        if mode == "gap":
            print(
                f"Threshold mode: gap={threshold_gap:.2f} "
                f"(threshold={thresh_value:.4f}, max={max_score:.4f})"
            )
        elif mode == "ratio":
            print(
                f"Threshold mode: ratio={threshold_ratio:.2f} "
                f"(threshold={thresh_value:.4f}, max={max_score:.4f})"
            )
        elif mode == "percentile":
            print(
                f"Threshold mode: percentile={threshold_percentile:.1f}% "
                f"(threshold={thresh_value:.4f})"
            )
        else:
            print(f"Threshold mode: absolute={threshold:.4f}")

        print(f"Total: keeping {total_kept}/{total_nodes} hidden nodes")

        return pruning_mask

    def _compute_gap_threshold(
        self, scores: torch.Tensor, min_gap_ratio: float = 0.3
    ) -> float:
        """
        Compute threshold based on the largest gap in sorted importance scores.

        This finds the natural "elbow" where importance drops significantly,
        adapting to the actual distribution of scores.

        Algorithm:
        1. Sort scores in descending order
        2. Compute relative gaps: (score[i] - score[i+1]) / score[i]
        3. Find the first gap that exceeds min_gap_ratio
        4. Threshold is the score just below that gap

        Args:
            scores: Tensor of importance scores
            min_gap_ratio: Minimum relative drop to consider significant (0.0 to 1.0)

        Returns:
            Threshold value
        """
        sorted_scores, _ = torch.sort(scores, descending=True)
        sorted_scores = sorted_scores.cpu().numpy()

        n = len(sorted_scores)
        if n <= 1:
            return 0.0

        # Find the largest significant gap
        best_gap_idx = n - 1  # Default: keep all nodes
        best_gap_value = 0.0

        for i in range(n - 1):
            if sorted_scores[i] > 1e-10:  # Avoid division by zero
                relative_gap = (
                    sorted_scores[i] - sorted_scores[i + 1]
                ) / sorted_scores[i]

                # Find first significant gap (elbow point)
                if relative_gap >= min_gap_ratio:
                    best_gap_idx = i
                    best_gap_value = relative_gap
                    break

        # Threshold is between the kept and pruned nodes
        # Use the score at best_gap_idx as minimum to keep
        threshold = (
            sorted_scores[best_gap_idx] - 1e-10
        )  # Slightly below to include this node

        print(
            f"  Gap analysis: found {best_gap_value:.1%} "
            f"gap after top {best_gap_idx + 1} nodes"
        )

        return float(threshold)

    def get_layer_widths(self) -> list[int]:
        """Get the width of each layer in the network."""
        widths = [self.model.layers[0].in_feat]
        for layer in self.model.layers:
            widths.append(layer.out_feat)
        return widths

    def print_importance_summary(self):
        """Print a summary of edge and node importance scores."""
        if not self.edge_scores:
            print("Edge scores not computed. Call compute_edge_scores first.")
            return

        print("\n=== Edge Importance Summary ===")
        for layer_idx, scores in self.edge_scores.items():
            print(
                f"Layer {layer_idx}: min={scores.min():.4f}, max={scores.max():.4f}, "
                f"mean={scores.mean():.4f}, std={scores.std():.4f}"
            )

        if self.node_scores:
            print("\n=== Node Importance Summary ===")
            for layer_idx, scores in self.node_scores.items():
                layer_type = (
                    "input"
                    if layer_idx == 0
                    else ("output" if layer_idx == len(self.model.layers) else "hidden")
                )
                print(
                    f"Layer {layer_idx} ({layer_type}): "
                    f"min={scores.min():.4f}, max={scores.max():.4f}, "
                    f"mean={scores.mean():.4f}"
                )


class KANTrainerWithPruning:
    """
    Training wrapper that incorporates sparsification and pruning.

    Implements the training pipeline from the KAN paper:
    1. Train with sparsification regularization
    2. Compute importance scores and prune
    3. Fine-tune the pruned model

    Args:
        model: KANet model
        optimizer: PyTorch optimizer
        device: Device to use
        lambda_l1: L1 regularization weight
        lambda_entropy: Entropy regularization weight
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        lambda_l1: float = 0.01,
        lambda_entropy: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.regularizer = SparsityRegularizer(lambda_l1, lambda_entropy)
        self.pruner = KANPruner(model, device)

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        sparsity_weight: float = 0.1,
    ) -> tuple[float, float]:
        """
        Train one epoch with sparsification regularization.

        Optimized to compute forward pass and activations in a single pass.

        Args:
            dataloader: Training data loader
            criterion: Loss function
            sparsity_weight: Weight for sparsification loss

        Returns:
            Tuple of (average task loss, average sparsity loss)
        """
        self.model.train()
        total_task_loss = 0.0
        total_sparsity_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            if isinstance(batch, list | tuple) and len(batch) >= 2:
                batch_x, batch_y = batch[0], batch[1]
            else:
                raise ValueError("Dataloader must provide (x, y) tuples")

            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            # OPTIMIZED: Single forward pass that returns both output and activations
            if sparsity_weight > 0:
                output, all_activations = forward_with_activations(self.model, batch_x)

                # Task loss
                task_loss = criterion(output, batch_y)

                # Sparsification loss using pre-computed activations
                sparsity_loss = self.regularizer(
                    self.model, batch_x, precomputed_activations=all_activations
                )

                # Combined loss
                loss = task_loss + sparsity_weight * sparsity_loss
                total_sparsity_loss += sparsity_loss.item()
            else:
                # No regularization - use standard forward pass
                output = self.model(batch_x)
                task_loss = criterion(output, batch_y)
                loss = task_loss

            loss.backward()
            self.optimizer.step()

            total_task_loss += task_loss.item()
            n_batches += 1

        avg_sparsity = total_sparsity_loss / n_batches if sparsity_weight > 0 else 0.0
        return total_task_loss / n_batches, avg_sparsity

    def analyze_for_pruning(
        self,
        dataloader: torch.utils.data.DataLoader,
        threshold_percentile: float = 10.0,
    ) -> dict[int, torch.Tensor]:
        """
        Analyze the network and determine which nodes to prune.

        Args:
            dataloader: Data to use for computing importance scores
            threshold_percentile: Prune the bottom X% of nodes

        Returns:
            Pruning mask dictionary
        """
        print("\nComputing edge importance scores...")
        self.pruner.compute_edge_scores(dataloader)

        print("Computing node importance scores...")
        self.pruner.compute_node_scores()

        self.pruner.print_importance_summary()

        print(
            f"\nDetermining pruning mask (threshold_percentile={threshold_percentile}%)"
            "..."
        )
        mask = self.pruner.get_pruning_mask(threshold_percentile=threshold_percentile)

        return mask

    def train_with_pruning(
        self,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        n_epochs_before_prune: int = 50,
        n_epochs_after_prune: int = 50,
        sparsity_weight: float = 0.1,
        prune_threshold_percentile: float = 10.0,
        verbose: bool = True,
    ) -> dict[int, torch.Tensor]:
        """
        Full training pipeline with pruning.

        1. Train with sparsification
        2. Analyze and get pruning recommendations
        3. Fine-tune (user should apply pruning manually if desired)

        Args:
            train_loader: Training data loader
            criterion: Loss function
            n_epochs_before_prune: Epochs to train before pruning analysis
            n_epochs_after_prune: Epochs to fine-tune after pruning analysis
            sparsity_weight: Weight for sparsification loss
            prune_threshold_percentile: Prune bottom X% of nodes
            verbose: Whether to print progress

        Returns:
            Pruning mask dictionary
        """
        # Phase 1: Train with sparsification
        if verbose:
            print("=" * 50)
            print("Phase 1: Training with sparsification...")
            print("=" * 50)

        for epoch in range(n_epochs_before_prune):
            task_loss, sparsity_loss = self.train_epoch(
                train_loader, criterion, sparsity_weight
            )
            if verbose and (epoch % 10 == 0 or epoch == n_epochs_before_prune - 1):
                print(
                    f"Epoch {epoch}: Task Loss = {task_loss:.4f}, "
                    f"Sparsity Loss = {sparsity_loss:.4f}"
                )

        # Phase 2: Analyze for pruning
        if verbose:
            print("\n" + "=" * 50)
            print("Phase 2: Analyzing for pruning...")
            print("=" * 50)

        pruning_mask = self.analyze_for_pruning(
            train_loader, threshold_percentile=prune_threshold_percentile
        )

        # Phase 3: Fine-tune (without sparsity regularization)
        if verbose:
            print("\n" + "=" * 50)
            print("Phase 3: Fine-tuning...")
            print("=" * 50)

        for epoch in range(n_epochs_after_prune):
            task_loss, _ = self.train_epoch(
                train_loader, criterion, sparsity_weight=0.0
            )
            if verbose and (epoch % 10 == 0 or epoch == n_epochs_after_prune - 1):
                print(f"Epoch {epoch}: Task Loss = {task_loss:.4f}")

        return pruning_mask
