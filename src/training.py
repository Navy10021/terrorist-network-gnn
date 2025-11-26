"""
Training Module for Temporal GNN with Self-Supervised Learning
==============================================================

Implements:
1. Self-supervised learning via temporal link prediction
2. Contrastive learning for temporal consistency
3. Node attribute reconstruction
4. Training utilities and metrics

Author: Advanced GNN Research
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
    """
    Loss for temporal link prediction task
    
    Given edges at time t, predict edges at time t+1
    """
    def __init__(self, negative_sampling_ratio: float = 1.0):
        super().__init__()
        self.negative_sampling_ratio = negative_sampling_ratio
        
    def forward(
        self,
        embeddings: torch.Tensor,
        pos_edges: torch.Tensor,
        neg_edges: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [num_nodes, hidden_dim]
            pos_edges: [2, num_pos_edges] - positive edges
            neg_edges: [2, num_neg_edges] - negative edges
        """
        # Positive edge scores
        pos_src = embeddings[pos_edges[0]]
        pos_dst = embeddings[pos_edges[1]]
        pos_scores = (pos_src * pos_dst).sum(dim=-1)
        
        # Negative edge scores
        neg_src = embeddings[neg_edges[0]]
        neg_dst = embeddings[neg_edges[1]]
        neg_scores = (neg_src * neg_dst).sum(dim=-1)
        
        # Binary cross entropy loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )
        
        return pos_loss + neg_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for temporal consistency
    
    Embeddings at nearby timesteps should be similar
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        embeddings_t: torch.Tensor,
        embeddings_t1: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings_t: [num_nodes, hidden_dim] at time t
            embeddings_t1: [num_nodes, hidden_dim] at time t+1
        """
        # Handle size mismatch (network size can change over time)
        min_nodes = min(embeddings_t.size(0), embeddings_t1.size(0))
        embeddings_t = embeddings_t[:min_nodes]
        embeddings_t1 = embeddings_t1[:min_nodes]
        
        # Normalize embeddings
        embeddings_t = F.normalize(embeddings_t, dim=-1)
        embeddings_t1 = F.normalize(embeddings_t1, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings_t, embeddings_t1.t()) / self.temperature
        
        # Diagonal elements are positive pairs
        num_nodes = embeddings_t.size(0)
        labels = torch.arange(num_nodes, device=embeddings_t.device)
        
        # Cross entropy loss
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class NodeReconstructionLoss(nn.Module):
    """
    Loss for reconstructing node attributes
    """
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            reconstructed: [num_nodes, num_features]
            original: [num_nodes, num_features]
            mask: [num_nodes] - which nodes to compute loss for
        """
        if mask is not None:
            reconstructed = reconstructed[mask]
            original = original[mask]
        
        return F.mse_loss(reconstructed, original)


class TemporalGNNTrainer:
    """
    Trainer for Temporal GNN with self-supervised learning
    """
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: torch.device = None
    ):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss functions
        self.link_loss = TemporalLinkPredictionLoss()
        self.contrast_loss = ContrastiveLoss(temperature=0.5)
        self.recon_loss = NodeReconstructionLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'link_loss': [],
            'contrast_loss': [],
            'recon_loss': []
        }
        
    def negative_sampling(
        self,
        num_nodes: int,
        pos_edges: torch.Tensor,
        num_neg_samples: int
    ) -> torch.Tensor:
        """
        Sample negative edges that don't exist in positive edges
        """
        # Create set of positive edges for fast lookup
        pos_edge_set = set()
        for i in range(pos_edges.size(1)):
            src, dst = pos_edges[0, i].item(), pos_edges[1, i].item()
            pos_edge_set.add((src, dst))
            pos_edge_set.add((dst, src))
        
        # Sample negative edges
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
            # Fallback: just sample random edges
            neg_edges = torch.randint(0, num_nodes, (2, num_neg_samples))
            return neg_edges.to(self.device)
        
        neg_edges = torch.tensor(neg_edges, dtype=torch.long).t()
        return neg_edges.to(self.device)
    
    def train_epoch(
        self,
        train_networks: List,
        loss_weights: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_networks: List of MultiLayerTemporalNetwork
            loss_weights: Weights for different loss components
        """
        if loss_weights is None:
            loss_weights = {
                'link': 1.0,
                'contrast': 0.5,
                'recon': 0.3
            }
        
        self.model.train()
        
        total_loss = 0.0
        total_link_loss = 0.0
        total_contrast_loss = 0.0
        total_recon_loss = 0.0
        num_batches = 0
        
        for network in train_networks:
            num_timesteps = len(network.layers_history)
            
            # Process each timestep pair (t, t+1)
            for t in range(num_timesteps - 1):
                # Get data at time t and t+1
                layers_t = network.get_timestep(t)
                layers_t1 = network.get_timestep(t + 1)
                agg_edge_index_t, _ = network.get_aggregated_network(t)
                num_nodes_t = layers_t[0].node_features.size(0)
                num_nodes_t1 = layers_t1[0].node_features.size(0)
                
                # Get embeddings using full temporal sequence
                num_ts = len(network.layers_history)
                timestamps = torch.arange(num_ts, dtype=torch.float32, device=self.device)
                embeddings = self.model(network, timestamps)
                
                # embeddings is for the last timestep (with most nodes)
                # Extract embeddings for nodes at t and t+1
                embeddings_t = embeddings[:num_nodes_t]
                embeddings_t1 = embeddings[:num_nodes_t1]
                
                # Get target edges at t+1
                agg_edge_index_t1, _ = network.get_aggregated_network(t + 1)
                
                # Move edges to device
                agg_edge_index_t1 = agg_edge_index_t1.to(self.device)
                
                # Negative sampling
                num_pos_edges = agg_edge_index_t1.size(1)
                num_neg_edges = int(num_pos_edges * 1.0)
                neg_edges = self.negative_sampling(num_nodes_t1, agg_edge_index_t1, num_neg_edges)
                
                # Compute losses
                link_loss = self.link_loss(embeddings_t1, agg_edge_index_t1, neg_edges)
                contrast_loss = self.contrast_loss(embeddings_t, embeddings_t1)
                
                # Node reconstruction loss (reconstruct features at t from embeddings)
                # Note: embeddings may be from last timestep, so we need to match sizes
                original_features = layers_t[0].node_features.to(self.device)
                num_nodes_t = original_features.size(0)
                
                # Only reconstruct for nodes that exist at timestep t
                if embeddings_t.size(0) >= num_nodes_t:
                    reconstructed = self.model.reconstruct_features(embeddings_t[:num_nodes_t])
                    recon_loss = self.recon_loss(reconstructed, original_features)
                else:
                    # Skip reconstruction if embedding size doesn't match
                    recon_loss = torch.tensor(0.0, device=self.device)
                
                # Combined loss
                loss = (
                    loss_weights['link'] * link_loss +
                    loss_weights['contrast'] * contrast_loss +
                    loss_weights['recon'] * recon_loss
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
                num_batches += 1
        
        # Average losses
        avg_loss = total_loss / max(num_batches, 1)
        avg_link_loss = total_link_loss / max(num_batches, 1)
        avg_contrast_loss = total_contrast_loss / max(num_batches, 1)
        avg_recon_loss = total_recon_loss / max(num_batches, 1)
        
        return {
            'loss': avg_loss,
            'link_loss': avg_link_loss,
            'contrast_loss': avg_contrast_loss,
            'recon_loss': avg_recon_loss
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        val_networks: List,
        loss_weights: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Evaluate on validation set
        """
        if loss_weights is None:
            loss_weights = {
                'link': 1.0,
                'contrast': 0.5,
                'recon': 0.3
            }
        
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
                
                # Get embeddings using full temporal sequence
                num_ts = len(network.layers_history)
                timestamps = torch.arange(num_ts, dtype=torch.float32, device=self.device)
                embeddings = self.model(network, timestamps)
                
                # Extract embeddings for specific timesteps
                embeddings_t = embeddings[:num_nodes_t]
                embeddings_t1 = embeddings[:num_nodes_t1]
                
                # Get target edges
                agg_edge_index_t1, _ = network.get_aggregated_network(t + 1)
                agg_edge_index_t1 = agg_edge_index_t1.to(self.device)
                
                # Negative sampling
                num_pos_edges = agg_edge_index_t1.size(1)
                num_neg_edges = int(num_pos_edges * 1.0)
                neg_edges = self.negative_sampling(num_nodes_t1, agg_edge_index_t1, num_neg_edges)
                
                # Compute loss
                link_loss = self.link_loss(embeddings_t1, agg_edge_index_t1, neg_edges)
                contrast_loss = self.contrast_loss(embeddings_t, embeddings_t1)
                
                loss = loss_weights['link'] * link_loss + loss_weights['contrast'] * contrast_loss
                
                total_loss += loss.item()
                total_link_loss += link_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_link_loss = total_link_loss / max(num_batches, 1)
        
        return {
            'loss': avg_loss,
            'link_loss': avg_link_loss
        }
    
    def fit(
        self,
        train_networks: List,
        val_networks: List,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Full training loop with early stopping
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        if verbose:
            print("="*80)
            print("TRAINING TEMPORAL GNN")
            print("="*80)
            print(f"Training networks: {len(train_networks)}")
            print(f"Validation networks: {len(val_networks)}")
            print(f"Epochs: {num_epochs}")
            print(f"Device: {self.device}")
            print()
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_networks)
            
            # Validate
            val_metrics = self.evaluate(val_networks)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['link_loss'].append(train_metrics['link_loss'])
            self.history['contrast_loss'].append(train_metrics['contrast_loss'])
            self.history['recon_loss'].append(train_metrics['recon_loss'])
            
            epoch_time = time.time() - start_time
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:03d}/{num_epochs} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Link: {train_metrics['link_loss']:.4f} | "
                      f"Contrast: {train_metrics['contrast_loss']:.4f} | "
                      f"Recon: {train_metrics['recon_loss']:.4f} | "
                      f"Time: {epoch_time:.2f}s")
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
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
            print("\n" + "="*80)
            print("TRAINING COMPLETE")
            print("="*80)
            print(f"Best validation loss: {best_val_loss:.4f}")
        
        return self.history


# Test and demonstration
if __name__ == "__main__":
    print("="*80)
    print("Training Module - Self-Supervised Learning")
    print("="*80)
    
    # This will be tested in the main experiment
    print("\nKey Components:")
    print("  ✓ Temporal Link Prediction Loss")
    print("  ✓ Contrastive Loss for Temporal Consistency")
    print("  ✓ Node Reconstruction Loss")
    print("  ✓ Negative Sampling")
    print("  ✓ Training Loop with Early Stopping")
    print("\nReady for integration!")
