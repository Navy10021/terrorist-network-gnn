"""
Training Module V2 - Enhanced Self-Supervised Learning
======================================================

Enhanced with:
1. Temporal autoencoder loss
2. Graph reconstruction loss
3. Improved negative sampling
4. Better training strategies

Author: Advanced GNN Research
Version: 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import time


class TemporalLinkPredictionLoss(nn.Module):
    """Loss for temporal link prediction task"""

    def __init__(self, negative_sampling_ratio: float = 1.0):
        super().__init__()
        self.negative_sampling_ratio = negative_sampling_ratio

    def forward(
        self, embeddings: torch.Tensor, pos_edges: torch.Tensor, neg_edges: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [num_nodes, hidden_dim]
            pos_edges: [2, num_pos_edges]
            neg_edges: [2, num_neg_edges]
        """
        pos_src = embeddings[pos_edges[0]]
        pos_dst = embeddings[pos_edges[1]]
        pos_scores = (pos_src * pos_dst).sum(dim=-1)

        neg_src = embeddings[neg_edges[0]]
        neg_dst = embeddings[neg_edges[1]]
        neg_scores = (neg_src * neg_dst).sum(dim=-1)

        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))

        return pos_loss + neg_loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for temporal consistency"""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings_t: torch.Tensor, embeddings_t1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings_t: [num_nodes, hidden_dim] at time t
            embeddings_t1: [num_nodes, hidden_dim] at time t+1
        """
        min_nodes = min(embeddings_t.size(0), embeddings_t1.size(0))
        embeddings_t = embeddings_t[:min_nodes]
        embeddings_t1 = embeddings_t1[:min_nodes]

        embeddings_t = F.normalize(embeddings_t, dim=-1)
        embeddings_t1 = F.normalize(embeddings_t1, dim=-1)

        similarity = torch.matmul(embeddings_t, embeddings_t1.t()) / self.temperature

        num_nodes = embeddings_t.size(0)
        labels = torch.arange(num_nodes, device=embeddings_t.device)

        loss = F.cross_entropy(similarity, labels)

        return loss


class NodeReconstructionLoss(nn.Module):
    """Loss for reconstructing node attributes"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            reconstructed: [num_nodes, num_features]
            original: [num_nodes, num_features]
            mask: [num_nodes]
        """
        if mask is not None:
            reconstructed = reconstructed[mask]
            original = original[mask]

        return F.mse_loss(reconstructed, original)


class TemporalAutoencoderLoss(nn.Module):
    """
    NEW: Temporal autoencoder loss - predict future state from current
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, embeddings_t: torch.Tensor, embeddings_t1: torch.Tensor) -> torch.Tensor:
        """
        Predict embeddings at t+1 from embeddings at t

        Args:
            embeddings_t: [num_nodes, hidden_dim]
            embeddings_t1: [num_nodes, hidden_dim]
        """
        min_nodes = min(embeddings_t.size(0), embeddings_t1.size(0))
        embeddings_t = embeddings_t[:min_nodes]
        embeddings_t1 = embeddings_t1[:min_nodes]

        predicted_t1 = self.predictor(embeddings_t)
        loss = F.mse_loss(predicted_t1, embeddings_t1)

        return loss


class GraphReconstructionLoss(nn.Module):
    """
    NEW: Graph structure reconstruction loss
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self, embeddings: torch.Tensor, pos_edges: torch.Tensor, neg_edges: torch.Tensor
    ) -> torch.Tensor:
        """
        Margin-based reconstruction loss

        Args:
            embeddings: [num_nodes, hidden_dim]
            pos_edges: [2, num_pos]
            neg_edges: [2, num_neg]
        """
        pos_score = (embeddings[pos_edges[0]] * embeddings[pos_edges[1]]).sum(-1)
        neg_score = (embeddings[neg_edges[0]] * embeddings[neg_edges[1]]).sum(-1)

        # Margin loss: pos_score should be higher than neg_score by margin
        loss = F.relu(self.margin - pos_score + neg_score).mean()

        return loss


class EnhancedNegativeSampler:
    """
    Enhanced negative sampling strategy
    """

    def __init__(self, strategy: str = "uniform"):
        """
        Args:
            strategy: 'uniform', 'degree_biased', 'hard_negative'
        """
        self.strategy = strategy

    def sample(
        self,
        num_nodes: int,
        pos_edges: torch.Tensor,
        num_neg_samples: int,
        embeddings: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Sample negative edges
        """
        if self.strategy == "uniform":
            return self._uniform_sampling(num_nodes, pos_edges, num_neg_samples, device)
        elif self.strategy == "hard_negative":
            return self._hard_negative_sampling(
                num_nodes, pos_edges, num_neg_samples, embeddings, device
            )
        else:
            return self._uniform_sampling(num_nodes, pos_edges, num_neg_samples, device)

    def _uniform_sampling(
        self, num_nodes: int, pos_edges: torch.Tensor, num_neg_samples: int, device: torch.device
    ) -> torch.Tensor:
        """Uniform random sampling"""
        pos_edge_set = set()
        for i in range(pos_edges.size(1)):
            src, dst = pos_edges[0, i].item(), pos_edges[1, i].item()
            pos_edge_set.add((src, dst))
            pos_edge_set.add((dst, src))

        neg_edges = []
        attempts = 0
        max_attempts = num_neg_samples * 100

        while len(neg_edges) < num_neg_samples and attempts < max_attempts:
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)

            if src != dst and (src, dst) not in pos_edge_set:
                neg_edges.append([src, dst])
            attempts += 1

        if len(neg_edges) == 0:
            neg_edges = torch.randint(0, num_nodes, (2, num_neg_samples))
            return neg_edges.to(device)

        neg_edges = torch.tensor(neg_edges, dtype=torch.long).t()
        return neg_edges.to(device)

    def _hard_negative_sampling(
        self,
        num_nodes: int,
        pos_edges: torch.Tensor,
        num_neg_samples: int,
        embeddings: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Hard negative sampling - sample nodes that are close in embedding space
        but not connected
        """
        if embeddings is None:
            return self._uniform_sampling(num_nodes, pos_edges, num_neg_samples, device)

        pos_edge_set = set()
        for i in range(pos_edges.size(1)):
            src, dst = pos_edges[0, i].item(), pos_edges[1, i].item()
            pos_edge_set.add((src, dst))
            pos_edge_set.add((dst, src))

        # Compute similarity matrix
        embeddings_norm = F.normalize(embeddings, dim=-1)
        similarity = torch.matmul(embeddings_norm, embeddings_norm.t())

        # Sample hard negatives
        neg_edges = []
        for _ in range(num_neg_samples):
            src = np.random.randint(0, min(num_nodes, embeddings.size(0)))

            # Get top-k similar nodes
            top_k = min(20, num_nodes)
            _, similar_nodes = similarity[src].topk(top_k)

            # Sample from similar nodes that are not connected
            for node in similar_nodes:
                dst = node.item()
                if dst != src and (src, dst) not in pos_edge_set:
                    neg_edges.append([src, dst])
                    break

        if len(neg_edges) < num_neg_samples:
            # Fill remaining with uniform sampling
            remaining = num_neg_samples - len(neg_edges)
            uniform_negs = self._uniform_sampling(num_nodes, pos_edges, remaining, device)
            neg_edges.extend(uniform_negs.t().tolist())

        neg_edges = torch.tensor(neg_edges[:num_neg_samples], dtype=torch.long).t()
        return neg_edges.to(device)


class EnhancedTemporalGNNTrainer:
    """
    Enhanced trainer with additional self-supervised tasks
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: torch.device = None,
        negative_sampling_strategy: str = "uniform",
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Scheduler for learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Loss functions
        self.link_loss = TemporalLinkPredictionLoss()
        self.contrast_loss = ContrastiveLoss(temperature=0.5)
        self.recon_loss = NodeReconstructionLoss()
        self.temporal_autoencoder_loss = TemporalAutoencoderLoss(
            hidden_dim=model.output_dim if hasattr(model, "output_dim") else 128
        ).to(self.device)
        self.graph_recon_loss = GraphReconstructionLoss(margin=1.0)

        # Enhanced negative sampler
        self.negative_sampler = EnhancedNegativeSampler(strategy=negative_sampling_strategy)

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "link_loss": [],
            "contrast_loss": [],
            "recon_loss": [],
            "temporal_ae_loss": [],
            "graph_recon_loss": [],
        }

    def train_epoch(
        self, train_networks: List, loss_weights: Dict[str, float] = None
    ) -> Dict[str, float]:
        """Train for one epoch"""
        if loss_weights is None:
            loss_weights = {
                "link": 1.0,
                "contrast": 0.5,
                "recon": 0.3,
                "temporal_ae": 0.4,  # NEW
                "graph_recon": 0.3,  # NEW
            }

        self.model.train()

        total_loss = 0.0
        total_link_loss = 0.0
        total_contrast_loss = 0.0
        total_recon_loss = 0.0
        total_temporal_ae_loss = 0.0
        total_graph_recon_loss = 0.0
        num_batches = 0

        for network in train_networks:
            num_timesteps = len(network.layers_history)

            for t in range(num_timesteps - 1):
                layers_t = network.get_timestep(t)
                layers_t1 = network.get_timestep(t + 1)
                agg_edge_index_t, _ = network.get_aggregated_network(t)
                num_nodes_t = layers_t[0].node_features.size(0)
                num_nodes_t1 = layers_t1[0].node_features.size(0)

                # Get embeddings
                num_ts = len(network.layers_history)
                timestamps = torch.arange(num_ts, dtype=torch.float32, device=self.device)
                embeddings = self.model(network, timestamps)

                embeddings_t = embeddings[:num_nodes_t]
                embeddings_t1 = embeddings[:num_nodes_t1]

                # Get target edges
                agg_edge_index_t1, _ = network.get_aggregated_network(t + 1)
                agg_edge_index_t1 = agg_edge_index_t1.to(self.device)

                # Negative sampling
                num_pos_edges = agg_edge_index_t1.size(1)
                num_neg_edges = int(num_pos_edges * 1.0)
                neg_edges = self.negative_sampler.sample(
                    num_nodes_t1, agg_edge_index_t1, num_neg_edges, embeddings_t1, self.device
                )

                # Compute losses
                link_loss = self.link_loss(embeddings_t1, agg_edge_index_t1, neg_edges)
                contrast_loss = self.contrast_loss(embeddings_t, embeddings_t1)

                # Temporal autoencoder loss (NEW)
                temporal_ae_loss = self.temporal_autoencoder_loss(embeddings_t, embeddings_t1)

                # Graph reconstruction loss (NEW)
                graph_recon_loss = self.graph_recon_loss(
                    embeddings_t1, agg_edge_index_t1, neg_edges
                )

                # Node reconstruction loss
                original_features = layers_t[0].node_features.to(self.device)
                num_nodes_t = original_features.size(0)

                if embeddings_t.size(0) >= num_nodes_t:
                    reconstructed = self.model.reconstruct_features(embeddings_t[:num_nodes_t])
                    recon_loss = self.recon_loss(reconstructed, original_features)
                else:
                    recon_loss = torch.tensor(0.0, device=self.device)

                # Combined loss
                loss = (
                    loss_weights["link"] * link_loss
                    + loss_weights["contrast"] * contrast_loss
                    + loss_weights["recon"] * recon_loss
                    + loss_weights["temporal_ae"] * temporal_ae_loss
                    + loss_weights["graph_recon"] * graph_recon_loss
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Accumulate losses
                total_loss += loss.item()
                total_link_loss += link_loss.item()
                total_contrast_loss += contrast_loss.item()
                total_recon_loss += recon_loss.item()
                total_temporal_ae_loss += temporal_ae_loss.item()
                total_graph_recon_loss += graph_recon_loss.item()
                num_batches += 1

        # Average losses
        avg_loss = total_loss / max(num_batches, 1)
        avg_link_loss = total_link_loss / max(num_batches, 1)
        avg_contrast_loss = total_contrast_loss / max(num_batches, 1)
        avg_recon_loss = total_recon_loss / max(num_batches, 1)
        avg_temporal_ae_loss = total_temporal_ae_loss / max(num_batches, 1)
        avg_graph_recon_loss = total_graph_recon_loss / max(num_batches, 1)

        return {
            "loss": avg_loss,
            "link_loss": avg_link_loss,
            "contrast_loss": avg_contrast_loss,
            "recon_loss": avg_recon_loss,
            "temporal_ae_loss": avg_temporal_ae_loss,
            "graph_recon_loss": avg_graph_recon_loss,
        }

    @torch.no_grad()
    def evaluate(
        self, val_networks: List, loss_weights: Dict[str, float] = None
    ) -> Dict[str, float]:
        """Evaluate on validation set"""
        if loss_weights is None:
            loss_weights = {"link": 1.0, "contrast": 0.5, "temporal_ae": 0.4}

        self.model.eval()

        total_loss = 0.0
        total_link_loss = 0.0
        num_batches = 0

        for network in val_networks:
            num_timesteps = len(network.layers_history)

            for t in range(num_timesteps - 1):
                layers_t = network.get_timestep(t)
                layers_t1 = network.get_timestep(t + 1)
                num_nodes_t = layers_t[0].node_features.size(0)
                num_nodes_t1 = layers_t1[0].node_features.size(0)

                # Get embeddings
                num_ts = len(network.layers_history)
                timestamps = torch.arange(num_ts, dtype=torch.float32, device=self.device)
                embeddings = self.model(network, timestamps)

                embeddings_t = embeddings[:num_nodes_t]
                embeddings_t1 = embeddings[:num_nodes_t1]

                # Get target edges
                agg_edge_index_t1, _ = network.get_aggregated_network(t + 1)
                agg_edge_index_t1 = agg_edge_index_t1.to(self.device)

                # Negative sampling
                num_pos_edges = agg_edge_index_t1.size(1)
                num_neg_edges = int(num_pos_edges * 1.0)
                neg_edges = self.negative_sampler.sample(
                    num_nodes_t1, agg_edge_index_t1, num_neg_edges, device=self.device
                )

                # Compute loss
                link_loss = self.link_loss(embeddings_t1, agg_edge_index_t1, neg_edges)
                contrast_loss = self.contrast_loss(embeddings_t, embeddings_t1)
                temporal_ae_loss = self.temporal_autoencoder_loss(embeddings_t, embeddings_t1)

                loss = (
                    loss_weights["link"] * link_loss
                    + loss_weights["contrast"] * contrast_loss
                    + loss_weights["temporal_ae"] * temporal_ae_loss
                )

                total_loss += loss.item()
                total_link_loss += link_loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_link_loss = total_link_loss / max(num_batches, 1)

        return {"loss": avg_loss, "link_loss": avg_link_loss}

    def fit(
        self,
        train_networks: List,
        val_networks: List,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> Dict:
        """Full training loop with early stopping"""
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        if verbose:
            print("=" * 80)
            print("TRAINING TEMPORAL GNN V2 - ENHANCED")
            print("=" * 80)
            print(f"Training networks: {len(train_networks)}")
            print(f"Validation networks: {len(val_networks)}")
            print(f"Epochs: {num_epochs}")
            print(f"Device: {self.device}")
            print("\nEnhancements:")
            print("  ✓ Temporal Autoencoder Loss")
            print("  ✓ Graph Reconstruction Loss")
            print("  ✓ Enhanced Negative Sampling")
            print()

        for epoch in range(num_epochs):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(train_networks)

            # Validate
            val_metrics = self.evaluate(val_networks)

            # Update learning rate
            self.scheduler.step(val_metrics["loss"])

            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["link_loss"].append(train_metrics["link_loss"])
            self.history["contrast_loss"].append(train_metrics["contrast_loss"])
            self.history["recon_loss"].append(train_metrics["recon_loss"])
            self.history["temporal_ae_loss"].append(train_metrics["temporal_ae_loss"])
            self.history["graph_recon_loss"].append(train_metrics["graph_recon_loss"])

            epoch_time = time.time() - start_time

            if verbose and (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch+1:03d}/{num_epochs} | "
                    f"Train: {train_metrics['loss']:.4f} | "
                    f"Val: {val_metrics['loss']:.4f} | "
                    f"Link: {train_metrics['link_loss']:.4f} | "
                    f"TAE: {train_metrics['temporal_ae_loss']:.4f} | "
                    f"GR: {train_metrics['graph_recon_loss']:.4f} | "
                    f"Time: {epoch_time:.2f}s"
                )

            # Early stopping
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        if verbose:
            print("\n" + "=" * 80)
            print("TRAINING COMPLETE")
            print("=" * 80)
            print(f"Best validation loss: {best_val_loss:.4f}")

        return self.history


# Backward compatibility
class TemporalGNNTrainer(EnhancedTemporalGNNTrainer):
    """Alias for backward compatibility"""

    pass


# Test code
if __name__ == "__main__":
    print("=" * 80)
    print("Training Module V2 - Enhanced")
    print("=" * 80)
    print("\nEnhancements:")
    print("  ✓ Temporal Autoencoder Loss")
    print("  ✓ Graph Reconstruction Loss")
    print("  ✓ Hard Negative Sampling")
    print("  ✓ Learning Rate Scheduling")
    print("  ✓ Improved Training Loop")
    print("\n✓ Enhanced training module ready!")
