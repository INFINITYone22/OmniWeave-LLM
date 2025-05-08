# Copyright 2025 INFINITYone22 github. All rights reserved.

import torch
import torch.nn as nn
import math # Ensure math is imported for math.sqrt
from typing import Optional # Ensure Optional is imported

class LargeTransformer(nn.Module):
    """
    Defines a large-scale Transformer encoder model.

    This class implements a standard Transformer encoder architecture with
    configurable dimensions, suitable for large language modeling tasks.
    It uses learned positional embeddings.
    """
    def __init__(self, num_layers: int = 262, d_model: int = 15360, num_heads: int = 120,
                 d_ff: int = 61440, vocab_size: int = 128000, max_seq_len: int = 1024, dropout: float = 0.1):
        """
        Initializes the LargeTransformer.

        Args:
            num_layers: Number of transformer encoder layers.
            d_model: Dimensionality of the model (embeddings and hidden states).
            num_heads: Number of attention heads.
            d_ff: Dimensionality of the feed-forward network.
            vocab_size: Vocabulary size.
            max_seq_len: Maximum sequence length for positional embeddings.
            dropout: Dropout probability.
        """
        super().__init__()
        self.d_model = d_model

        # Input embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding (learned positional embeddings)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True  # Expects input shape (batch, seq, feature)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer (for simplicity, projecting back to vocab size)
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        Initializes the weights of model layers.

        Applies normal initialization to Linear and Embedding layers.
        Initializes LayerNorm biases to zero and weights to one.
        More specific initializations can be added for TransformerEncoderLayer components if needed.

        Args:
            module: The module to initialize.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
             # Initialize LayerNorm
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # Example of more specific initialization (can be expanded)
        # elif isinstance(module, nn.TransformerEncoderLayer):
        #     if hasattr(module.self_attn, 'in_proj_weight'):
        #          module.self_attn.in_proj_weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs the forward pass of the Transformer encoder.

        Args:
            src: Input tensor (batch_size, sequence_length) of token IDs.
            src_mask: Additive mask for the src sequence (optional). Shape (sequence_length, sequence_length).
            src_key_padding_mask: Mask for src keys per batch (optional). Shape (batch_size, sequence_length).
        Returns:
            Output tensor (batch_size, sequence_length, vocab_size).
        """
        seq_len = src.size(1)
        max_pos_len = self.pos_embedding.size(1)
        if seq_len > max_pos_len:
            raise ValueError(f"Input sequence length ({seq_len}) exceeds maximum positional "
                             f"embedding length ({max_pos_len})")

        # Embed the input tokens and add positional embeddings
        src_emb = self.embedding(src) * math.sqrt(self.d_model) # Scale embedding
        src_emb = src_emb + self.pos_embedding[:, :seq_len, :]

        # Pass through the Transformer encoder
        # Note: PyTorch TransformerEncoder expects mask shapes:
        # src_mask: (S, S)
        # src_key_padding_mask: (N, S) where N=batch_size, S=seq_len
        output = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        # Project to vocabulary size
        output = self.output_layer(output)
        return output

# Model Configuration & Operational Notes:
# The LargeTransformer class is designed with the following considerations,
# assuming a configuration similar to the __init__ defaults (e.g., ~742B parameters if d_model=15360, num_layers=262, etc.).
#
# 1. Target Precision: FP4
#    - Parameter Memory: For a ~742B model, FP4 (0.5 bytes/param) implies ~345.6 GB for parameters.
#    - System Assumption: FP4 precision is assumed to be handled by underlying hardware
#      (e.g., NVIDIA GB200 Superchip's Transformer Engine) and specialized libraries.
#    - Model Code: The Python code uses standard PyTorch modules (FP32/FP16).
#      Actual FP4 quantization requires external mechanisms (not implemented in this script).
#
# 2. Memory and Parallelism:
#    - Given typical HBM capacities (e.g., 384 GB on a GB200 accelerator node),
#      activations and workspace must fit alongside parameters.
#    - Requires model parallelism (e.g., tensor, pipeline, sequence parallelism using
#      frameworks like Megatron-LM or DeepSpeed) across multiple accelerators.
#
# 3. Performance Profile:
#    - Aimed to be compute-bottlenecked on suitable hardware.
#    - For example, with batch_size=1, seq_len=1024, the arithmetic intensity
#      is expected to be high (> 2500 ops/byte), suitable for GB200-like systems.
#
# Note: The hyperparameter values in __init__ are illustrative for a very large model.
# Actual deployment would require careful tuning and resource management.
# (End of illustrative example code and notes from original file)
