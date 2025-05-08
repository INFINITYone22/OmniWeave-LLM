# Copyright 2025 INFINITYone22 github. All rights reserved.

import torch
import torch.nn as nn
import math
from typing import Optional, List
from dataclasses import dataclass

# --- Configuration ---
@dataclass
class ModelConfig:
    """Configuration for the Multimodal LLM."""
    text_vocab_size: int = 128000  # Vocabulary size for text tokens
    dim: int = 8192
    num_layers: int = 110
    num_heads: int = 64
    ffn_dim: int = 32768
    max_seq_len: int = 8192
    dropout: float = 0.1
    # VQ-VAE specific
    vq_codebook_size: int = 8192  # Defines the number of image tokens
    vq_codebook_dim: int = 512
    # Image Encoder specific (ViT)
    image_size: int = 256 # Example image size
    patch_size: int = 16
    # Audio Encoder specific
    audio_conv_kernel: int = 10
    audio_conv_stride: int = 5

# --- Model Components ---

# FlashAttention implementation (simplified, assumes library availability)
class FlashAttention(nn.Module):
    """
    Simplified FlashAttention with Multi-Query Attention (MQA).

    This implementation assumes the availability of an optimized FlashAttention kernel.
    It uses a single key/value head shared across all query heads for efficiency.
    """
    def __init__(self, dim: int, num_heads: int, head_dim: int):
        """
        Initialize FlashAttention.

        Args:
            dim: Input dimension.
            num_heads: Number of query heads.
            head_dim: Dimension of each head.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Multi-Query Attention: one key/value head shared across all query heads
        self.q_proj = nn.Linear(dim, num_heads * head_dim)
        self.k_proj = nn.Linear(dim, head_dim)  # Single key projection
        self.v_proj = nn.Linear(dim, head_dim)  # Single value projection
        self.out_proj = nn.Linear(num_heads * head_dim, dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Applies FlashAttention to the input tensor.

        Args:
            x: Input tensor (batch, seq_len, dim).
            mask: An optional mask to prevent attention to certain positions.

        Returns:
            Output tensor (batch, seq_len, dim).
        """
        batch, seq_len, dim = x.shape
        # Project queries, keys, values
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, 1, self.head_dim)  # Single key head
        v = self.v_proj(x).view(batch, seq_len, 1, self.head_dim)  # Single value head
        # FlashAttention logic (simplified, assumes optimized kernel)
        scores = torch.einsum("bnhd,bm1d->bnhm", q, k) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bnhm,bm1d->bnhd", attn, v)
        out = out.view(batch, seq_len, -1)
        return self.out_proj(out)

# Transformer Layer
class TransformerLayer(nn.Module):
    """
    Single transformer decoder layer with FlashAttention and MQA.
    """
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        """
        Initialize TransformerLayer.

        Args:
            dim: Input dimension.
            num_heads: Number of attention heads.
            ffn_dim: Dimension of the feed-forward network.
            dropout: The dropout probability.
        """
        super().__init__()
        self.attn = FlashAttention(dim, num_heads, dim // num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Applies the transformer layer to the input tensor.

        Args:
            x: The input tensor.
            mask: An optional mask.

        Returns:
            Output tensor.
        """
        # Attention with residual connection
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        # Feed-forward with residual connection
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

# VQ-VAE for Image Generation
class VQVAE(nn.Module):
    """
    VQ-VAE for encoding/decoding images into discrete tokens.
    """
    def __init__(self, config: ModelConfig):
        """
        Initialize VQ-VAE from ModelConfig.

        Args:
            config: Model configuration object.
        """
        super().__init__()
        self.codebook_size = config.vq_codebook_size
        self.codebook_dim = config.vq_codebook_dim

        # Encoder: CNN with residual blocks
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.Conv2d(256, config.vq_codebook_dim, kernel_size=3, stride=2, padding=1), # Output spatial dim depends on input size and strides
        )

        # Codebook
        self.codebook = nn.Parameter(torch.randn(config.vq_codebook_size, config.vq_codebook_dim))

        # Decoder: Transposed CNN
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(config.vq_codebook_dim, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # Output RGB values in [0, 1]
        )

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """
        Quantizes the input tensor to the nearest codebook entry.

        Args:
            z: The input tensor.

        Returns:
            Tuple of (quantized_tensor, indices of codebook_entries).
        """
        z_flat = z.flatten(0, -2)  # [batch*H*W, codebook_dim]
        distances = torch.cdist(z_flat, self.codebook)  # [batch*H*W, codebook_size]
        indices = distances.argmin(dim=-1)  # [batch*H*W]
        z_quantized = self.codebook[indices].view(z.shape)
        return z_quantized, indices

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Encodes the input image to tokens and decodes back to the image.

        Args:
            x: The input image.

        Returns:
            Tuple of (reconstructed_image, indices of codebook_entries).
        """
        z = self.encoder(x)  # [batch, codebook_dim, H', W']
        z_quantized, indices = self.quantize(z)
        x_recon = self.decoder(z_quantized)
        return x_recon, indices

# Modality Encoders
class TextEncoder(nn.Module):
    """Encodes text tokens into embeddings with positional encoding."""
    def __init__(self, config: ModelConfig):
        """
        Initialize TextEncoder from ModelConfig.

        Args:
            config: Model configuration object.
        """
        super().__init__()
        self.embedding = nn.Embedding(config.text_vocab_size + config.vq_codebook_size, config.dim)
        # Create positional encoding up to max_seq_len
        self.pos_encoding = self._create_pos_encoding(config.max_seq_len, config.dim)
        # Register buffer makes it part of the state_dict but not trainable
        self.register_buffer('pe', self.pos_encoding)

    def _create_pos_encoding(self, max_len: int, dim: int) -> torch.Tensor:
        """
        Creates sinusoidal positional encodings.

        Args:
            max_len: The maximum sequence length.
            dim: The dimension of the embeddings.

        Returns:
            Positional encodings tensor.
        """
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input text tokens into embeddings.

        Args:
            tokens: The input text tokens.

        Returns:
            Embeddings tensor.
        """
        x = self.embedding(tokens)
        x = x + self.pos_encoding[:tokens.size(1), :].to(x.device)
        return x

class ImageEncoder(nn.Module):
    """Vision Transformer (ViT) like encoder for image input."""
    def __init__(self, config: ModelConfig):
        """
        Initialize ImageEncoder (ViT-like) from ModelConfig.

        Args:
            config: Model configuration object.
        """
        super().__init__()
        image_height, image_width = config.image_size, config.image_size
        patch_height, patch_width = config.patch_size, config.patch_size
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = 3 * patch_height * patch_width # Channels * H * W

        # Using a simple linear projection for patches instead of Conv2d for flexibility
        self.patch_proj = nn.Linear(patch_dim, config.dim)
        self.patch_size = config.patch_size
        self.num_patches = num_patches

        # CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, config.dim))

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Reshapes image into patches."""
        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, f"Image dimensions ({H}, {W}) must be divisible by patch size ({p})."
        num_patches_h = H // p
        num_patches_w = W // p
        # Reshape: (B, C, H, W) -> (B, C, num_patches_h, p, num_patches_w, p)
        x = x.view(B, C, num_patches_h, p, num_patches_w, p)
        # Permute: (B, num_patches_h, num_patches_w, C, p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        # Flatten: (B, num_patches_h * num_patches_w, C * p * p)
        x = x.view(B, num_patches_h * num_patches_w, C * p * p)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input image into patch embeddings with CLS token and positional info.

        Args:
            x: Input image tensor of shape (batch, channels, height, width).

        Returns:
            Image embeddings tensor (batch, num_patches + 1, dim).
        """
        B = x.shape[0]
        # Convert image to patches and project
        patches = self._to_patches(x)
        x = self.patch_proj(patches) # [B, num_patches, dim]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, dim]
        x = torch.cat((cls_tokens, x), dim=1) # [B, num_patches + 1, dim]

        # Add positional embeddings
        x = x + self.pos_embedding

        return x

class AudioEncoder(nn.Module):
    """Simplified 1D CNN encoder for audio."""
    def __init__(self, config: ModelConfig):
        """
        Initialize AudioEncoder (simplified 1D CNN) from ModelConfig.

        Args:
            config: Model configuration object.
        """
        super().__init__()
        # Simple 1D convolution to extract features
        self.conv = nn.Conv1d(1, config.dim, kernel_size=config.audio_conv_kernel, stride=config.audio_conv_stride)
        # Optional: Add more layers or a projection layer if needed
        # self.proj = nn.Linear(config.dim, config.dim) # Example projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input audio waveform into embeddings.

        Args:
            x: Input audio tensor of shape (batch, 1, audio_seq_len).

        Returns:
            Audio embeddings tensor (batch, output_seq_len, dim).
        """
        # Apply convolution
        x = self.conv(x)  # [batch, dim, output_seq_len]
        # Transpose to (batch, output_seq_len, dim) for transformer compatibility
        x = x.transpose(1, 2)
        # Apply projection if defined
        # if hasattr(self, 'proj'):
        #     x = self.proj(x)
        return x

# --- Main Multimodal LLM ---
class MultimodalLLM(nn.Module):
    """
    Multimodal Large Language Model combining text, image, and audio processing.

    This model uses separate encoders for each modality and a shared Transformer
    backbone to process the combined sequence. It can generate text and potentially
    image tokens (via VQ-VAE).
    """
    def __init__(self, config: ModelConfig):
        """
        Initialize MultimodalLLM from ModelConfig.

        Args:
            config: Model configuration object.
        """
        super().__init__()
        self.config = config

        # Modality encoders
        self.text_encoder = TextEncoder(config)
        self.image_encoder = ImageEncoder(config) # Using ViT-like encoder
        self.audio_encoder = AudioEncoder(config)
        self.vqvae = VQVAE(config) # For image tokenization/generation

        # Transformer backbone
        self.layers = nn.ModuleList([
            TransformerLayer(config.dim, config.num_heads, config.ffn_dim, dropout=config.dropout)
            for _ in range(config.num_layers)
        ])

        # Final layer norm and output head
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.text_vocab_size + config.vq_codebook_size) # Predicts next token ID

    def _prepare_inputs(self, text_tokens: Optional[torch.Tensor] = None,
                        images: Optional[torch.Tensor] = None,
                        audio: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encodes and concatenates inputs from different modalities."""
        embeddings = []
        current_seq_len = 0

        if text_tokens is not None:
            text_emb = self.text_encoder(text_tokens)
            embeddings.append(text_emb)
            current_seq_len += text_tokens.size(1)

        if images is not None:
            # Option 1: Use ViT-like encoder output directly
            image_emb = self.image_encoder(images) # [B, num_patches+1, dim]
            embeddings.append(image_emb)
            current_seq_len += image_emb.size(1)
            # Option 2: Use VQ-VAE tokens (requires mapping tokens to embeddings)
            # _, image_indices = self.vqvae(images) # [B, H', W']
            # image_indices_flat = image_indices.view(images.size(0), -1) # [B, num_image_tokens]
            # image_emb = self.text_encoder.embedding(image_indices_flat + text_vocab_offset) # Need offset
            # embeddings.append(image_emb)
            # current_seq_len += image_indices_flat.size(1)

        if audio is not None:
            audio_emb = self.audio_encoder(audio) # [B, audio_seq_len', dim]
            embeddings.append(audio_emb)
            current_seq_len += audio_emb.size(1)

        if not embeddings:
             # Handle case with no inputs (e.g., unconditional generation)
             # This might need adjustment based on specific use case
             return torch.zeros(1, 0, self.config.dim, device=self.head.weight.device) # Return empty sequence

        # Concatenate embeddings along the sequence dimension
        x = torch.cat(embeddings, dim=1)

        # Ensure sequence length does not exceed max_seq_len
        if current_seq_len > self.config.max_seq_len:
            # Simple truncation, could be more sophisticated
            x = x[:, :self.config.max_seq_len, :]
            print(f"Warning: Input sequence truncated from {current_seq_len} to {self.config.max_seq_len}")

        return x

    def forward(self, text_tokens: Optional[torch.Tensor] = None,
                images: Optional[torch.Tensor] = None,
                audio: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs the forward pass of the MultimodalLLM.

        Args:
            text_tokens: Input text token IDs (batch, text_seq_len).
            images: Input image tensors (batch, channels, height, width).
            audio: Input audio waveforms (batch, 1, audio_seq_len).
            attention_mask: Optional custom attention mask. If None, a causal mask is generated.

        Returns:
            Output logits (batch, combined_seq_len, vocab_size).
        """
        # 1. Encode and combine inputs
        x = self._prepare_inputs(text_tokens, images, audio)
        batch_size, seq_len, _ = x.shape

        # 2. Create attention mask if not provided (default: causal mask)
        if attention_mask is None and seq_len > 0:
            # Standard causal mask for autoregressive decoding
            attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
            # Expand mask to match expected shape for nn.TransformerEncoderLayer (needs adjustments if using custom attention)
            # PyTorch's TransformerEncoderLayer expects mask shape (S, S) or (N*num_heads, S, S)
            # For simplicity, we'll pass None here and assume the layer handles causal masking if needed,
            # or modify TransformerLayer to accept and apply the mask correctly.
            # Let's assume TransformerLayer handles the causal mask internally for now.
            # TODO: Verify mask handling in TransformerLayer/FlashAttention
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))


        # 3. Pass through Transformer layers
        # Note: Mask handling needs careful verification based on TransformerLayer implementation
        for layer in self.layers:
             # Pass causal_mask if TransformerLayer expects it
             x = layer(x, mask=causal_mask) # Assuming layer handles causal mask

        # 4. Final normalization and output projection
        x = self.norm(x)
        logits = self.head(x) # [batch, seq_len, vocab_size]

        return logits

    @torch.no_grad()
    def generate(self, text_prompt: Optional[torch.Tensor] = None,
                 image_prompt: Optional[torch.Tensor] = None,
                 audio_prompt: Optional[torch.Tensor] = None,
                 max_new_tokens: int = 50,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None) -> torch.Tensor:
        """
        Generates token sequences autoregressively based on prompts.

        Args:
            text_prompt: Text token IDs to start generation (batch, text_seq_len).
            image_prompt: Image tensor prompt (batch, channels, height, width).
            audio_prompt: Audio waveform prompt (batch, 1, audio_seq_len).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature. 1.0 means no change, < 1.0 sharpens, > 1.0 flattens.
            top_k: If set, only sample from the top k most likely tokens.

        Returns:
            Generated token IDs (batch, total_seq_len).
        """
        self.eval() # Set model to evaluation mode

        # Prepare initial input sequence from prompts
        prompt_embeddings = self._prepare_inputs(text_prompt, image_prompt, audio_prompt)
        current_tokens = prompt_embeddings # Use embeddings directly if generation starts from them
        # If starting generation *after* prompts, need the initial token IDs
        # For simplicity, let's assume text_prompt provides the starting tokens for generation
        generated_sequence = text_prompt if text_prompt is not None else torch.empty((prompt_embeddings.size(0), 0), dtype=torch.long, device=prompt_embeddings.device)


        for _ in range(max_new_tokens):
            # Prepare input for the next step (use the current sequence)
            # Need to handle the sequence length limit
            input_sequence = generated_sequence[:, -self.config.max_seq_len:] # Use last max_seq_len tokens

            # Get logits for the next token
            # Need to pass appropriate inputs based on how _prepare_inputs and forward work
            # This part needs careful implementation based on whether we pass tokens or embeddings
            # Assuming we pass token IDs for simplicity here:
            logits = self(text_tokens=input_sequence) # Pass only the current sequence
            next_token_logits = logits[:, -1, :] # Logits for the last token position

            # Apply temperature scaling
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

            # Sample the next token ID
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            # Append the new token ID to the sequence
            generated_sequence = torch.cat((generated_sequence, next_token_id), dim=1)

            # TODO: Add stopping criteria (e.g., EOS token)

        return generated_sequence


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Initialize configuration
    config = ModelConfig(
        # Smaller dimensions for feasible local testing
        dim=512,
        num_layers=4,
        num_heads=8,
        ffn_dim=2048,
        max_seq_len=512,
        text_vocab_size=20000, # Smaller text vocab for example
        vq_codebook_size=1024, # Using the example vq_codebook_size for image tokens
        vq_codebook_dim=128,
        image_size=64, # Smaller image
        patch_size=16
    )
    print("Model Configuration:", config)

    # 2. Initialize model
    model = MultimodalLLM(config)
    print(f"\nModel Initialized. Parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


    # 3. Create Dummy Inputs
    batch_size = 2
    text_seq_len = 32
    image_size = config.image_size
    audio_seq_len = 2000 # Example audio length

    dummy_text = torch.randint(0, config.text_vocab_size, (batch_size, text_seq_len))
    dummy_image = torch.randn(batch_size, 3, image_size, image_size)
    dummy_audio = torch.randn(batch_size, 1, audio_seq_len)

    print(f"\nDummy Input Shapes:")
    print(f"  Text: {dummy_text.shape}")
    print(f"  Image: {dummy_image.shape}")
    print(f"  Audio: {dummy_audio.shape}")

    # 4. Forward Pass (Training Mode Example)
    print("\nTesting Forward Pass...")
    try:
        logits = model(text_tokens=dummy_text, images=dummy_image, audio=dummy_audio)
        print(f"  Output Logits Shape: {logits.shape}") # Should be (batch_size, combined_seq_len, config.text_vocab_size + config.vq_codebook_size)
    except Exception as e:
        print(f"  Forward pass failed: {e}")
        import traceback
        traceback.print_exc()


    # 5. Generation Example (Inference Mode)
    print("\nTesting Generation...")
    try:
        # Generate starting from a text prompt
        prompt = torch.randint(0, config.text_vocab_size, (1, 10)) # Single prompt, length 10
        generated_tokens = model.generate(text_prompt=prompt, max_new_tokens=20)
        print(f"  Generated Token Sequence Shape: {generated_tokens.shape}")
        print(f"  Generated Tokens (first prompt): {generated_tokens[0].tolist()}")

        # Example generating with image prompt (requires careful handling of input sequence)
        # generated_from_image = model.generate(image_prompt=dummy_image[:1], max_new_tokens=15)
        # print(f"  Generated from Image Shape: {generated_from_image.shape}")

    except Exception as e:
        print(f"  Generation failed: {e}")
        import traceback
        traceback.print_exc()
