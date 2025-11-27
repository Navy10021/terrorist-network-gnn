"""
Advanced Temporal Graph Neural Network (T-GNN) V2 - Enhanced Version
====================================================================

Enhanced with:
1. Hierarchical Temporal Pooling
2. Multi-scale Feature Extraction
3. Improved Temporal Modeling
4. Better Memory Mechanisms

Author: Advanced GNN Research
Version: 2.0
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, TransformerConv
from torch_geometric.utils import to_dense_batch


class HierarchicalTemporalPooling(nn.Module):
    """
    Hierarchical temporal pooling for multi-scale temporal features
    Captures patterns at different time scales
    """

    def __init__(self, hidden_dim: int, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales

        # Multi-scale convolutions
        self.scales = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=2**i,
                    stride=2**i,
                    padding=2 ** (i - 1) if i > 0 else 0,
                )
                for i in range(num_scales)
            ]
        )

        # Scale fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, hidden_dim, seq_len]
        Returns:
            pooled: [batch, hidden_dim]
        """
        multi_scale = []

        for scale in self.scales:
            scaled = scale(x)
            # Adaptive pooling to fixed size
            pooled = F.adaptive_avg_pool1d(scaled, 1)
            multi_scale.append(pooled.squeeze(-1))

        # Concatenate and fuse
        fused = torch.cat(multi_scale, dim=-1)
        output = self.fusion(fused)
        output = self.norm(output)

        return output


class AdaptiveTimeEncoding(nn.Module):
    """
    Adaptive time encoding that learns continuous time representations
    """

    def __init__(self, d_model: int, max_timescale: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_timescale = max_timescale

        # Learnable frequency and phase parameters
        self.frequency_scale = nn.Parameter(torch.ones(d_model // 2))
        self.phase_shift = nn.Parameter(torch.zeros(d_model // 2))

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: [batch_size, seq_len] or [batch_size]
        Returns:
            time_encoding: [batch_size, seq_len, d_model] or [batch_size, d_model]
        """
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(-1)

        # Create frequency bands
        position = timestamps.unsqueeze(-1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model // 2, dtype=torch.float32, device=timestamps.device)
            * -(math.log(self.max_timescale) / (self.d_model // 2))
        )

        # Apply learnable scaling and phase shift
        div_term = div_term * self.frequency_scale

        # Compute sine and cosine encodings
        pe_sin = torch.sin(position * div_term + self.phase_shift)
        pe_cos = torch.cos(position * div_term + self.phase_shift)

        # Concatenate
        pe = torch.cat([pe_sin, pe_cos], dim=-1)

        return pe


class EnhancedTemporalMemoryBank(nn.Module):
    """
    Enhanced memory bank with attention-based read/write
    """

    def __init__(self, memory_size: int, memory_dim: int, num_heads: int = 4):
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_heads = num_heads

        # Memory storage
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim))

        # Query, Key, Value projections
        self.query_proj = nn.Linear(memory_dim, memory_dim)
        self.key_proj = nn.Linear(memory_dim, memory_dim)
        self.value_proj = nn.Linear(memory_dim, memory_dim)

        # Output projection
        self.out_proj = nn.Linear(memory_dim, memory_dim)

        # Memory update gate with context
        self.update_gate = nn.Sequential(
            nn.Linear(memory_dim * 3, memory_dim), nn.Sigmoid()  # query, old_memory, retrieved
        )

        # Usage tracking
        self.usage = nn.Parameter(torch.zeros(memory_size), requires_grad=False)

    def forward(self, query: torch.Tensor, update: bool = False) -> torch.Tensor:
        """
        Args:
            query: [batch_size, memory_dim]
            update: whether to update memory
        Returns:
            retrieved: [batch_size, memory_dim]
        """
        batch_size = query.size(0)

        # Compute attention over memory
        Q = self.query_proj(query)  # [batch_size, memory_dim]
        K = self.key_proj(self.memory)  # [memory_size, memory_dim]
        V = self.value_proj(self.memory)  # [memory_size, memory_dim]

        # Multi-head attention
        head_dim = self.memory_dim // self.num_heads
        Q = Q.view(batch_size, self.num_heads, head_dim)
        K = K.view(1, self.memory_size, self.num_heads, head_dim).transpose(1, 2)
        V = V.view(1, self.memory_size, self.num_heads, head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q.unsqueeze(2), K.transpose(-2, -1))
        scores = scores / math.sqrt(head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        # Retrieve from memory
        retrieved = torch.matmul(attn_weights, V)
        retrieved = retrieved.squeeze(2).reshape(batch_size, self.memory_dim)
        retrieved = self.out_proj(retrieved)

        # Update memory if training
        if update and self.training:
            self._update_memory(query, retrieved, attn_weights.squeeze(2).mean(1))

        return retrieved

    def _update_memory(
        self, query: torch.Tensor, retrieved: torch.Tensor, attn_weights: torch.Tensor
    ):
        """
        Update memory based on new information with least-recently-used policy
        """
        batch_size = query.size(0)

        for i in range(batch_size):
            # Find least used memory slot
            _, lru_idx = self.usage.min(0)

            # Also consider attention weights for importance
            max_attn_idx = attn_weights[i].argmax()

            # Use the one with higher attention if it's underutilized
            if self.usage[max_attn_idx] < self.usage.mean():
                update_idx = max_attn_idx
            else:
                update_idx = lru_idx

            # Compute update gate
            old_memory = self.memory[update_idx]
            gate_input = torch.cat([query[i], old_memory, retrieved[i]], dim=-1)
            gate = self.update_gate(gate_input)

            # Update memory
            self.memory.data[update_idx] = gate * query[i] + (1 - gate) * old_memory

            # Update usage
            self.usage.data[update_idx] += 1.0


class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-head attention mechanism for temporal sequences
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model]
            mask: [batch_size, seq_len_q, seq_len_k]
        Returns:
            output: [batch_size, seq_len_q, d_model]
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(context)

        return output


class CausalTemporalConv(nn.Module):
    """
    Causal temporal convolution that respects temporal ordering
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_channels, seq_len]
        Returns:
            output: [batch_size, out_channels, seq_len]
        """
        x = self.conv(x)
        # Remove future information
        if self.padding > 0:
            x = x[:, :, : -self.padding]
        return x


class EnhancedTemporalBlock(nn.Module):
    """
    Enhanced temporal convolutional block with residual connections
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv1 = CausalTemporalConv(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalTemporalConv(out_channels, out_channels, kernel_size, dilation)

        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual connection with projection if needed
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

        # Squeeze-and-Excitation for channel attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, out_channels // 4, 1),
            nn.ReLU(),
            nn.Conv1d(out_channels // 4, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_channels, seq_len]
        Returns:
            output: [batch_size, out_channels, seq_len]
        """
        residual = x

        # First convolution
        out = self.conv1(x)
        out = out.transpose(1, 2)
        out = self.norm1(out)
        out = out.transpose(1, 2)
        out = self.relu(out)
        out = self.dropout(out)

        # Second convolution
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = self.relu(out)
        out = self.dropout(out)

        # Squeeze-and-Excitation
        se_weight = self.se(out)
        out = out * se_weight

        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.relu(out + residual)


class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer layer with multi-head attention over graph structure
    """

    def __init__(
        self, d_model: int, num_heads: int, edge_dim: Optional[int] = None, dropout: float = 0.1
    ):
        super().__init__()

        self.transformer_conv = TransformerConv(
            d_model,
            d_model // num_heads,
            heads=num_heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, d_model]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_dim]
        Returns:
            output: [num_nodes, d_model]
        """
        # Graph attention with residual
        residual = x
        x = self.transformer_conv(x, edge_index, edge_attr)
        x = self.norm1(x + residual)

        # Feed-forward with residual
        residual = x
        x = self.ffn(x)
        x = self.norm2(x + residual)

        return x


class AdvancedTemporalGNN(nn.Module):
    """
    Advanced Temporal Graph Neural Network V2 - Enhanced Version

    New Features:
    - Hierarchical temporal pooling
    - Enhanced memory bank with LRU policy
    - Squeeze-and-Excitation in temporal blocks
    - Better feature fusion
    """

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int = 128,
        num_temporal_layers: int = 3,
        num_graph_layers: int = 3,
        num_attention_heads: int = 8,
        memory_size: int = 100,
        dropout: float = 0.1,
        max_seq_len: int = 50,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_temporal_layers = num_temporal_layers
        self.num_graph_layers = num_graph_layers

        # Input projection
        self.node_encoder = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Adaptive time encoding
        self.time_encoder = AdaptiveTimeEncoding(hidden_dim)

        # Enhanced memory bank
        self.memory_bank = EnhancedTemporalMemoryBank(memory_size, hidden_dim, num_attention_heads)

        # Enhanced temporal convolution blocks
        self.temporal_conv_blocks = nn.ModuleList(
            [
                EnhancedTemporalBlock(
                    hidden_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    dilation=2**i,
                    dropout=dropout,
                )
                for i in range(num_temporal_layers)
            ]
        )

        # Hierarchical temporal pooling
        self.hierarchical_pooling = HierarchicalTemporalPooling(hidden_dim, num_scales=3)

        # Multi-head temporal attention
        self.temporal_attention = nn.ModuleList(
            [
                MultiHeadTemporalAttention(hidden_dim, num_attention_heads, dropout)
                for _ in range(num_temporal_layers)
            ]
        )

        # Graph transformer layers
        self.graph_layers = nn.ModuleList(
            [
                GraphTransformerLayer(hidden_dim, num_attention_heads, hidden_dim, dropout)
                for _ in range(num_graph_layers)
            ]
        )

        # Cross-time message passing
        self.cross_time_attention = MultiHeadTemporalAttention(
            hidden_dim, num_attention_heads, dropout
        )

        # Positional encoding for temporal order
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))

        # Output layers with better fusion
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # temporal + memory + hierarchical
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Node feature reconstruction head
        self.node_reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_node_features),
        )

    def forward(
        self,
        node_features_seq: List[torch.Tensor],
        edge_indices_seq: List[torch.Tensor],
        edge_features_seq: List[torch.Tensor],
        timestamps: torch.Tensor,
        batch_seq: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features_seq: List of [num_nodes_t, num_node_features] for each timestep
            edge_indices_seq: List of [2, num_edges_t] for each timestep
            edge_features_seq: List of [num_edges_t, num_edge_features] for each timestep
            timestamps: [num_timesteps] - timestamp for each snapshot
            batch_seq: Optional list of batch assignments for each timestep

        Returns:
            embeddings: [num_nodes_total, hidden_dim]
        """
        num_timesteps = len(node_features_seq)
        device = node_features_seq[0].device

        # Encode nodes and edges at each timestep
        encoded_nodes_seq = []
        encoded_edges_seq = []

        for t in range(num_timesteps):
            # Node encoding
            node_emb = self.node_encoder(node_features_seq[t])

            # Add time encoding
            time_emb = self.time_encoder(timestamps[t].unsqueeze(0))
            time_emb = time_emb.squeeze(0)
            time_emb = time_emb.expand(node_emb.size(0), -1)
            node_emb = node_emb + time_emb

            # Add positional encoding
            if t < self.positional_encoding.size(1):
                node_emb = node_emb + self.positional_encoding[:, t, :].expand(node_emb.size(0), -1)

            encoded_nodes_seq.append(node_emb)

            # Edge encoding
            if edge_features_seq[t].size(0) > 0:
                edge_emb = self.edge_encoder(edge_features_seq[t])
                encoded_edges_seq.append(edge_emb)
            else:
                encoded_edges_seq.append(None)

        # Apply graph layers at each timestep
        graph_outputs_seq = []

        for t in range(num_timesteps):
            x = encoded_nodes_seq[t]
            edge_index = edge_indices_seq[t]
            edge_attr = encoded_edges_seq[t]

            # Apply graph transformer layers
            for graph_layer in self.graph_layers:
                x = graph_layer(x, edge_index, edge_attr)

            graph_outputs_seq.append(x)

        # Temporal modeling with enhanced blocks
        max_nodes = max(x.size(0) for x in graph_outputs_seq)
        temporal_input = torch.zeros(max_nodes, self.hidden_dim, num_timesteps, device=device)

        for t, x in enumerate(graph_outputs_seq):
            temporal_input[: x.size(0), :, t] = x

        # Multi-scale temporal convolution
        temporal_features = temporal_input
        for conv_block in self.temporal_conv_blocks:
            temporal_features = conv_block(temporal_features)

        # Hierarchical temporal pooling
        hierarchical_features = self.hierarchical_pooling(temporal_features)

        # Temporal attention over sequence
        temporal_features = temporal_features.transpose(1, 2)

        for attn_layer in self.temporal_attention:
            temporal_features = attn_layer(temporal_features, temporal_features, temporal_features)

        # Memory-augmented retrieval
        memory_output = []
        for node_idx in range(temporal_features.size(0)):
            node_temporal = temporal_features[node_idx].mean(0)
            memory_retrieved = self.memory_bank(node_temporal.unsqueeze(0), update=self.training)
            memory_output.append(memory_retrieved)

        memory_output = torch.cat(memory_output, dim=0)

        # Cross-time message passing
        cross_time_features = self.cross_time_attention(
            temporal_features.view(-1, num_timesteps, self.hidden_dim).mean(1, keepdim=True),
            temporal_features.view(-1, num_timesteps, self.hidden_dim),
            temporal_features.view(-1, num_timesteps, self.hidden_dim),
        ).squeeze(1)

        # Enhanced feature fusion: temporal + memory + hierarchical
        final_features = torch.cat(
            [temporal_features.mean(1), memory_output, hierarchical_features], dim=-1
        )

        final_output = self.output_projection(final_features)
        final_output = self.output_norm(final_output)

        # Extract actual node embeddings
        actual_outputs = []
        for t, x in enumerate(graph_outputs_seq):
            actual_outputs.append(final_output[: x.size(0)])

        return actual_outputs[-1]

    def reconstruct_features(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Reconstruct node features from embeddings"""
        return self.node_reconstructor(embeddings)

    def get_temporal_embeddings(
        self,
        node_features_seq: List[torch.Tensor],
        edge_indices_seq: List[torch.Tensor],
        edge_features_seq: List[torch.Tensor],
        timestamps: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Get embeddings for all timesteps"""
        num_timesteps = len(node_features_seq)
        device = node_features_seq[0].device

        encoded_nodes_seq = []

        for t in range(num_timesteps):
            node_emb = self.node_encoder(node_features_seq[t])
            time_emb = self.time_encoder(timestamps[t].unsqueeze(0))
            time_emb = time_emb.squeeze(0)
            time_emb = time_emb.expand(node_emb.size(0), -1)
            node_emb = node_emb + time_emb

            if t < self.positional_encoding.size(1):
                node_emb = node_emb + self.positional_encoding[:, t, :].expand(node_emb.size(0), -1)

            encoded_nodes_seq.append(node_emb)

        embeddings_seq = []

        for t in range(num_timesteps):
            x = encoded_nodes_seq[t]
            edge_index = edge_indices_seq[t]
            edge_attr = (
                self.edge_encoder(edge_features_seq[t])
                if edge_features_seq[t].size(0) > 0
                else None
            )

            for graph_layer in self.graph_layers:
                x = graph_layer(x, edge_index, edge_attr)

            embeddings_seq.append(x)

        return embeddings_seq


# Test code
if __name__ == "__main__":
    print("=" * 80)
    print("Advanced Temporal GNN V2 - Enhanced Version")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Model configuration
    num_node_features = 64
    num_edge_features = 32
    hidden_dim = 128

    model = AdvancedTemporalGNN(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=hidden_dim,
        num_temporal_layers=3,
        num_graph_layers=3,
        num_attention_heads=8,
        memory_size=100,
        dropout=0.1,
    ).to(device)

    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nEnhancements:")
    print("  ✓ Hierarchical Temporal Pooling")
    print("  ✓ Enhanced Memory Bank with LRU")
    print("  ✓ Squeeze-and-Excitation Blocks")
    print("  ✓ Improved Feature Fusion")
    print("\n✓ Enhanced model ready!")
