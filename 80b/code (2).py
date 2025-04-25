import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# --- Configuration ---
class OmniWeaveConfig:
    """Configuration for the OmniWeave 80B Model"""
    def __init__(self, **kwargs):
        # Core Decoder Config
        self.d_model = 8192           # Hidden dimension
        self.num_layers = 96          # Number of decoder layers (adjust slightly if needed for exact 80B)
        self.num_heads = 64           # Number of attention heads (d_head = 128)
        self.d_ffn = 28672            # Intermediate dim in FFN (Approx 3.5 * d_model, often related to SwiGLU needs)
        self.vocab_size = 128000      # Text vocabulary size (adjust based on tokenizer)
        self.image_vocab_size = 16384 # VQ-GAN codebook size (common value)
        self.max_seq_len = 8192       # Max context length (can be extended with techniques)
        self.dropout_rate = 0.0       # Dropout often set to 0 for very large models pre-training
        self.norm_eps = 1e-5          # Epsilon for RMSNorm stability
        self.rope_theta = 10000.0     # RoPE base frequency

        # Input Encoder Configs (Simplified examples)
        self.vit_config = {
            "d_model": 1024, "layers": 24, "heads": 16, "patch_size": 14, "img_size": 224
        }
        self.audio_config = {
            "d_model": 1024, "layers": 24, "heads": 16 # For transformer part
            # + CNN feature extractor config needed (e.g., output dim before projection)
        }

        # Precision (Conceptual placeholder - Execution detail)
        self.precision = "fp4"

        # Derived/Fixed
        self.d_head = self.d_model // self.num_heads

        # Update attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

# --- Helper Modules ---

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # Learnable gain

    def _norm(self, x):
        # Calculate RMS and normalize
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # RMSNorm is often computed in float32 for stability
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        # Precompute frequencies (inv_freq)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cosine and sine tables
        t = torch.arange(max_seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int):
        # Return precomputed cos/sin values sliced to the current seq_len
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies RoPE to input tensor x"""
    # Reshape x for rotation: (..., seq_len, ..., dim) -> (..., seq_len, ..., dim/2, 2)
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x_reshaped.unbind(-1)
    # Apply rotation using complex number multiplication formula: (x1+ix2) * (c+is) = (x1c - x2s) + i(x1s + x2c)
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    # Combine back and reshape
    rotated_x = torch.stack((rotated_x1, rotated_x2), dim=-1).reshape_as(x)
    return rotated_x.type_as(x)


class SelfAttention(nn.Module):
    """Multi-Head Self Attention with RoPE"""
    def __init__(self, config: OmniWeaveConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.d_head = config.d_head

        # Linear layers for Q, K, V, and Output
        # In an FP4 context, these linear layers would utilize quantized weights/compute
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.rotary_emb = RotaryEmbedding(self.d_head, config.max_seq_len, theta=config.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None, # Needed if not inferring pos from seq len
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        B, T, C = hidden_states.size() # Batch, Sequence Length, Channels (d_model)

        # 1. Project Q, K, V
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.d_head).transpose(1, 2) # [B, nh, T, hd]
        k = self.k_proj(hidden_states).view(B, T, self.num_heads, self.d_head).transpose(1, 2) # [B, nh, T, hd]
        v = self.v_proj(hidden_states).view(B, T, self.num_heads, self.d_head).transpose(1, 2) # [B, nh, T, hd]

        # 2. Apply RoPE
        past_len = 0
        if kv_cache is not None:
             past_len = kv_cache[0].shape[2] # Get T from K cache: [B, nh, T_past, hd]

        cos, sin = self.rotary_emb(T + past_len) # Get cos/sin for the full sequence length
        # Apply RoPE to the *new* query/key tokens based on their absolute positions
        q = apply_rotary_emb(q, cos=cos[..., past_len:T+past_len, :], sin=sin[..., past_len:T+past_len, :])
        k = apply_rotary_emb(k, cos=cos[..., past_len:T+past_len, :], sin=sin[..., past_len:T+past_len, :])

        # 3. KV Caching (for efficient generation)
        if kv_cache is not None:
            # Concatenate past K, V with current K, V
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        # Update cache for next iteration
        current_kv_cache = (k, v)

        # 4. Attention Calculation
        # *** CRITICAL NOTE ***: Replace with optimized attention like FlashAttention
        # The following is standard scaled dot-product attention, **infeasible** at 80B scale
        # without optimization due to memory (T^2) and compute requirements.
        # attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.d_head) # [B, nh, T, T_kv]
        #
        # if attention_mask is not None:
        #     attn_weights = attn_weights + attention_mask # Apply mask (e.g., causal)
        #
        # attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        # attn_output = torch.matmul(attn_weights, v) # [B, nh, T, hd]

        # --- Placeholder for Optimized Attention ---
        # Assumes flash_attn_func(q, k, v, attention_mask) handles masking internally
        # Requires installing and importing flash_attn library
        try:
            from flash_attn import flash_attn_func
            # Note: flash_attn expects specific shapes and handles causal masking via argument
            is_causal = attention_mask is not None # Simplistic check, proper causal flag needed
            attn_output = flash_attn_func(q, k, v, causal=is_causal) # [B, nh, T, hd]
        except ImportError:
            print("WARNING: flash_attn not found. Using placeholder for attention output.")
            # Fallback (inefficient): Use torch.nn.functional.scaled_dot_product_attention if available (PyTorch >= 2.0)
            if hasattr(F, 'scaled_dot_product_attention'):
                 attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=is_causal)
            else:
                # Absolute fallback - just zero tensor (will not work for training)
                print("ERROR: No suitable attention implementation found!")
                attn_output = torch.zeros_like(q)
        # --- End Optimized Attention Placeholder ---


        # 5. Reshape and Output Projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C) # [B, T, C]
        attn_output = self.o_proj(attn_output)

        return attn_output, current_kv_cache


class SwiGLUFFN(nn.Module):
    """Swish Gated Linear Unit Feed-Forward Network"""
    def __init__(self, config: OmniWeaveConfig):
        super().__init__()
        # In FP4 context, these linear layers would use quantized weights/compute
        self.w1 = nn.Linear(config.d_model, config.d_ffn, bias=False) # Feed-forward
        self.w3 = nn.Linear(config.d_model, config.d_ffn, bias=False) # Gating mechanism
        self.w2 = nn.Linear(config.d_ffn, config.d_model, bias=False) # Output projection
        self.activation = F.silu # Swish activation (SiLU)

    def forward(self, x):
        # Compute feed-forward and gate, apply activation to feed-forward, multiply, project down
        return self.w2(self.activation(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """A single block of the Transformer Decoder"""
    def __init__(self, config: OmniWeaveConfig):
        super().__init__()
        self.attention = SelfAttention(config)
        self.feed_forward = SwiGLUFFN(config)
        self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.norm2 = RMSNorm(config.d_model, eps=config.norm_eps)
        # Dropout is often skipped in large model pre-training or applied elsewhere
        # self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        # Pre-Normalization: Normalize -> Attention -> Add Residual
        residual = hidden_states
        normed_hidden_states = self.norm1(hidden_states)
        attn_output, current_kv_cache = self.attention(
            normed_hidden_states, attention_mask=attention_mask, position_ids=position_ids, kv_cache=kv_cache
        )
        # Add residual connection (No dropout here based on config.dropout_rate=0)
        hidden_states = residual + attn_output

        # Pre-Normalization: Normalize -> FFN -> Add Residual
        residual = hidden_states
        normed_hidden_states = self.norm2(hidden_states)
        ffn_output = self.feed_forward(normed_hidden_states)
        # Add residual connection
        hidden_states = residual + ffn_output

        return hidden_states, current_kv_cache


# --- Input Encoders (Structural Placeholders) ---

class ViTEncoder(nn.Module):
    """Placeholder Vision Transformer Encoder"""
    def __init__(self, vit_config, output_dim):
        super().__init__()
        self.config = vit_config
        self.output_dim = output_dim # The dimension of the main OmniWeave model (d_model)
        # --- Assume full ViT Implementation exists here ---
        # (Patch Embedding, Positional Embedding, Transformer Blocks, Final Norm)
        # Example: Dummy layer to represent the ViT processing
        self.dummy_vit = nn.Linear(vit_config['d_model'], vit_config['d_model'])
        print(f"Placeholder ViT: Input dim {vit_config['d_model']}, Layers {vit_config['layers']}")
        # --- Projection Layer ---
        self.projection = nn.Linear(vit_config['d_model'], output_dim, bias=False)
        print(f"ViT Projection: {vit_config['d_model']} -> {output_dim}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Input: images [B, C, H, W]
        # 1. Patchify and embed images
        # 2. Pass through ViT transformer blocks
        # 3. Extract final sequence of patch features [B, NumPatches, D_vit]
        print("Warning: Using placeholder ViTEncoder forward pass.")
        B = images.size(0)
        num_patches = (self.config['img_size'] // self.config['patch_size']) ** 2
        # Simulate ViT output
        vit_features = torch.randn(B, num_patches, self.config['d_model'], device=images.device)
        # 4. Project features to the main model's dimension
        projected_features = self.projection(vit_features) # [B, NumPatches, D_omniweave]
        return projected_features


class AudioEncoder(nn.Module):
    """Placeholder Audio Encoder (Wav2Vec2/HuBERT style)"""
    def __init__(self, audio_config, output_dim):
        super().__init__()
        self.config = audio_config
        self.output_dim = output_dim
        # --- Assume full Wav2Vec2/HuBERT Implementation exists here ---
        # (CNN Feature Extractor, Positional Embedding, Transformer Blocks, Final Norm)
        # Example: Dummy layer to represent audio processing
        self.dummy_audio_transformer = nn.Linear(audio_config['d_model'], audio_config['d_model'])
        print(f"Placeholder Audio Encoder: Input dim {audio_config['d_model']}, Layers {audio_config['layers']}")
        # --- Projection Layer ---
        self.projection = nn.Linear(audio_config['d_model'], output_dim, bias=False)
        print(f"Audio Projection: {audio_config['d_model']} -> {output_dim}")

    def forward(self, audio_waveforms: torch.Tensor) -> torch.Tensor:
        # Input: waveforms [B, NumSamples] or spectrograms
        # 1. Extract features using CNNs
        # 2. Pass through Transformer blocks
        # 3. Extract final sequence of audio features [B, NumAudioTokens, D_audio]
        print("Warning: Using placeholder AudioEncoder forward pass.")
        B = audio_waveforms.size(0)
        # Simulate Audio Transformer output (sequence length depends on CNN strides)
        num_audio_tokens = 256 # Example value
        audio_features = torch.randn(B, num_audio_tokens, self.config['d_model'], device=audio_waveforms.device)
        # 4. Project features to the main model's dimension
        projected_features = self.projection(audio_features) # [B, NumAudioTokens, D_omniweave]
        return projected_features


# --- VQ-GAN Tokenizer (Conceptual Placeholder) ---
class VQGANTokenizer(nn.Module):
    """Placeholder for pre-trained VQ-GAN used for image tokenization"""
    def __init__(self, config: OmniWeaveConfig):
        super().__init__()
        # Typically loaded from a separate pre-trained checkpoint
        self.codebook_size = config.image_vocab_size
        self.latent_dim = 256 # Example latent dim for VQ codes (internal detail)
        self.image_size = 256 # Example target image size for VQGAN
        self.latent_grid_size = 16 # Example H/W of the latent grid (sqrt(NumPatches))
        print(f"Placeholder VQGAN Tokenizer: Codebook size {self.codebook_size}, Grid {self.latent_grid_size}x{self.latent_grid_size}")
        # Components would be:
        # self.encoder = ... CNN Encoder ...
        # self.decoder = ... CNN Decoder ...
        # self.quantize = ... VectorQuantizer layer (codebook lookup) ...

    def encode_to_tokens(self, images: torch.Tensor) -> torch.Tensor:
        # Input: images [B, C, H, W] (e.g., 256x256)
        # Output: token_ids [B, H_latent * W_latent] (e.g., [B, 256])
        print("Warning: Using placeholder VQGAN encode_to_tokens.")
        B = images.size(0)
        num_tokens = self.latent_grid_size * self.latent_grid_size
        dummy_token_ids = torch.randint(0, self.codebook_size, (B, num_tokens), device=images.device)
        return dummy_token_ids

    def decode_from_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Input: token_ids [B, H_latent * W_latent] (e.g., [B, 256])
        # Output: reconstructed_images [B, C, H, W]
        print("Warning: Using placeholder VQGAN decode_from_tokens.")
        B = token_ids.size(0)
        # Reshape IDs to grid if decoder expects it
        # token_grid = token_ids.view(B, self.latent_grid_size, self.latent_grid_size)
        # Lookup embeddings from codebook
        # Pass through CNN decoder
        dummy_reconstruction = torch.rand(B, 3, self.image_size, self.image_size, device=token_ids.device)
        return dummy_reconstruction


# --- Core Multimodal Model ---

class OmniWeaveModel(nn.Module):
    """The OmniWeave 80B Dense Multimodal Model"""
    def __init__(self, config: OmniWeaveConfig):
        super().__init__()
        self.config = config

        # --- Embeddings ---
        self.text_embed = nn.Embedding(config.vocab_size, config.d_model)
        # Embedding layer for discrete image tokens (from VQ-GAN)
        self.image_token_embed = nn.Embedding(config.image_vocab_size, config.d_model)
        # Note: Positional embeddings (RoPE) are applied inside attention

        # --- Input Processing Modules ---
        self.image_encoder = ViTEncoder(config.vit_config, config.d_model)
        self.audio_encoder = AudioEncoder(config.audio_config, config.d_model)
        # VQGAN is external but needed for image generation logic/token embedding
        # self.vqgan = VQGANTokenizer(config) # Conceptually separate

        # --- Core Decoder Blocks ---
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps) # Final normalization

        # --- Output Heads ---
        # Text Head (predicts next text token)
        self.text_lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Image Generation Head (predicts next VQ-GAN image token)
        self.image_lm_head = nn.Linear(config.d_model, config.image_vocab_size, bias=False)

        # --- Weight Tying ---
        self.text_embed.weight = self.text_lm_head.weight # Tie text embeddings & LM head
        # Optionally tie image token embeddings if used for input conditioning:
        # self.image_token_embed.weight = self.image_lm_head.weight

        print(f"\n--- OmniWeaveModel Initialized ---")
        print(f" Layers: {config.num_layers}, Dim: {config.d_model}, Heads: {config.num_heads}")
        print(f" FFN Dim: {config.d_ffn}, Text Vocab: {config.vocab_size}, Image Vocab: {config.image_vocab_size}")
        print(f" Target Precision: {config.precision} (Requires specific runtime support)")
        print(f" Using RoPE, RMSNorm, SwiGLU")
        print(f" WARNING: Requires optimized attention (e.g., FlashAttention) for feasibility.")
        print(f" WARNING: Input encoders and interleaving logic are placeholders.")
        print(f"-------------------------------------\n")

    def forward(
        self,
        # Inputs are assumed to be pre-processed into a single sequence
        # containing interleaved embeddings of different modalities.
        input_embeddings: torch.Tensor, # [B, T_combined, D]
        attention_mask: Optional[torch.Tensor] = None, # Combined causal/padding mask [B, 1, T, T] or similar
        position_ids: Optional[torch.Tensor] = None, # Positions [B, T] if needed by RoPE impl
        use_cache: bool = False, # Whether to use and return KV cache
        past_kv_caches: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> dict:
        """
        Main forward pass through the decoder stack.
        Assumes input_embeddings are already constructed from various modalities.
        """
        hidden_states = input_embeddings
        current_kv_caches = [] if use_cache else None

        # Pass through Transformer Decoder Layers
        for i, layer in enumerate(self.layers):
            past_kv = past_kv_caches[i] if past_kv_caches is not None else None
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=past_kv
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                current_kv_caches.append(layer_outputs[1])

        # Final Normalization
        hidden_states = self.norm(hidden_states)

        # --- Calculate Logits ---
        # Logits are calculated for all positions; loss is masked during training.
        text_logits = self.text_lm_head(hidden_states)       # [B, T_combined, vocab_size]
        image_logits = self.image_lm_head(hidden_states)    # [B, T_combined, image_vocab_size]

        output = {
            "last_hidden_state": hidden_states,
            "text_logits": text_logits,
            "image_logits": image_logits,
        }
        if use_cache:
             output["kv_caches"] = tuple(current_kv_caches) # type: ignore

        return output

    def prepare_inputs_for_generation(self, input_ids, past_kv_caches=None, **kwargs):
        """ Prepares inputs for the `generate` loop (simplified). """
        # In a real scenario, this handles embedding lookup and retrieving cache length
        token_embeddings = self.text_embed(input_ids) # Assuming text-only for simplicity here
        # Need to handle position_ids and attention_mask correctly based on cache length
        # ... logic to get correct position_ids and mask ...
        position_ids = None # Placeholder
        attention_mask = None # Placeholder
        return {
            "input_embeddings": token_embeddings,
            "past_kv_caches": past_kv_caches,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": kwargs.get("use_cache", True)
        }


    @torch.no_grad()
    def generate(
        self,
        # --- Inputs ---
        # This needs a sophisticated way to handle mixed initial prompts
        # For simplicity, let's assume it starts with text tokens for now
        input_ids: Optional[torch.Tensor] = None,
        # OR provide initial embeddings directly
        inputs_embeds: Optional[torch.Tensor] = None,

        # --- Control Parameters ---
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 50,
        do_sample: bool = True,
        # Special tokens for control
        eos_token_id: Optional[int] = None, # End Of Sequence token ID
        image_token_start_id: Optional[int] = None, # Token indicating image generation should start
        image_token_end_id: Optional[int] = None, # Token indicating image generation should end
        max_image_tokens: int = 256, # e.g., 16x16 grid

        # --- Internal State ---
        past_kv_caches=None,
        attention_mask=None,
        position_ids=None,
        **kwargs # Other model forward args
    ):
        """
        Autoregressive generation loop for text and optionally image tokens.
        NOTE: This is a highly simplified conceptual implementation.
        Real generation requires careful handling of multimodal prompts, state
        management (text vs image generation mode), masking, KV caching,
        stopping conditions, and sampling strategies.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            # Prepare initial embeddings if IDs are given
            # This is simplified - needs proper handling of multimodal prep
            model_inputs = self.prepare_inputs_for_generation(input_ids, past_kv_caches, **kwargs)
            current_embeddings = model_inputs["input_embeddings"]
        else:
            batch_size, seq_len, _ = inputs_embeds.shape
            current_embeddings = inputs_embeds
            # Assume other inputs (mask, pos_ids, cache) are passed correctly if using embeds directly
            model_inputs = {
                "input_embeddings": current_embeddings,
                "past_kv_caches": past_kv_caches,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "use_cache": kwargs.get("use_cache", True)
            }


        generated_token_ids = input_ids.tolist()[0] if input_ids is not None else [] # Keep track of generated IDs
        num_generated = 0
        in_image_generation = False
        num_image_tokens_generated = 0

        while num_generated < max_new_tokens:

            # Prepare model inputs for the next step
            # Only pass the *last* token's embedding for generation step
            step_inputs = {
                "input_embeddings": current_embeddings[:, -1:, :], # [B, 1, D]
                "past_kv_caches": model_inputs.get("past_kv_caches"),
                "attention_mask": model_inputs.get("attention_mask"), # Needs careful update for new token
                "position_ids": model_inputs.get("position_ids"),   # Needs careful update for new token
                "use_cache": True
            }

            # Forward pass for the next token prediction
            outputs = self.forward(**step_inputs)
            next_token_logits_text = outputs["text_logits"][:, -1, :]   # [B, vocab_size]
            next_token_logits_image = outputs["image_logits"][:, -1, :] # [B, image_vocab_size]

            # Update KV cache for the next iteration
            model_inputs["past_kv_caches"] = outputs.get("kv_caches")

            # --- Determine generation mode (Text or Image) ---
            # This logic is crucial and depends on the specific control tokens used
            # Simplified logic: if we are in image generation mode, predict image tokens
            if in_image_generation:
                next_token_logits = next_token_logits_image
                vocab_size = self.config.image_vocab_size
                current_eos_id = image_token_end_id
            else:
                next_token_logits = next_token_logits_text
                vocab_size = self.config.vocab_size
                current_eos_id = eos_token_id

            # --- Sampling ---
            if do_sample:
                # Apply temperature scaling
                logits = next_token_logits / temperature
                # Apply Top-K filtering
                if top_k > 0:
                    top_k = min(top_k, vocab_size)
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float("Inf")
                # Sample from the filtered distribution
                probs = F.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).squeeze(1) # [B]
            else:
                # Greedy decoding
                next_token_id = torch.argmax(next_token_logits, dim=-1)

            # --- Check Stopping Conditions & Mode Switching ---
            # Stop if EOS token is generated
            if current_eos_id is not None and next_token_id.item() == current_eos_id:
                 if in_image_generation:
                     in_image_generation = False # Finished image block
                     print("Exiting image generation mode.")
                     # Optionally predict a text token immediately after? Or just stop?
                 else:
                     print("EOS token generated. Stopping.")
                     break # Stop text generation

            # Switch to image generation mode if start token is generated
            if not in_image_generation and image_token_start_id is not None and next_token_id.item() == image_token_start_id:
                print("Entering image generation mode.")
                in_image_generation = True
                num_image_tokens_generated = 0
                # Don't increment num_generated for the control token itself? Or start predicting image?
                # Let's predict the first image token in the *next* iteration

            # Stop image generation if max image tokens reached
            if in_image_generation:
                num_image_tokens_generated += 1
                if num_image_tokens_generated >= max_image_tokens:
                    in_image_generation = False
                    print(f"Max image tokens ({max_image_tokens}) reached. Exiting image mode.")
                    # Optionally predict image_token_end_id here? Or move to text?


            # --- Prepare input for the *next* iteration ---
            # Get embedding for the newly generated token
            if in_image_generation and next_token_id.item() != image_token_start_id: # Embed as image token unless it was the start token
                next_token_embed = self.image_token_embed(next_token_id).unsqueeze(1)
            else: # Embed as text token
                next_token_embed = self.text_embed(next_token_id).unsqueeze(1)

            # Append the new embedding
            current_embeddings = torch.cat([current_embeddings, next_token_embed], dim=1)

            # Update attention mask and position IDs (CRITICAL, but complex logic omitted)
            # model_inputs["attention_mask"] = updated_mask
            # model_inputs["position_ids"] = updated_position_ids

            generated_token_ids.append(next_token_id.item())
            num_generated += 1

        return generated_token_ids


# --- Main Execution Block (Conceptual & Parameter Count) ---
if __name__ == '__main__':
    config = OmniWeaveConfig()

    # Instantiate the main model
    model = OmniWeaveModel(config)

    # --- Parameter Count Check (Approximate) ---
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_params = count_parameters(model)
    print(f"\nApproximate Total Trainable Parameters in OmniWeaveModel: {total_params / 1e9:.3f} Billion")

    # Estimate parameters per component (rough):
    params_decoder = sum(count_parameters(layer) for layer in model.layers)
    params_embed = count_parameters(model.text_embed) + count_parameters(model.image_token_embed)
    params_heads = count_parameters(model.text_lm_head) + count_parameters(model.image_lm_head) # Note weight tying
    params_norms = count_parameters(model.norm) + sum(count_parameters(l.norm1) + count_parameters(l.norm2) for l in model.layers)
    params_encoders = count_parameters(model.image_encoder) + count_parameters(model.audio_encoder)

    print(f"  Decoder Layers Approx: {params_decoder / 1e9:.3f} B")
    print(f"  Embeddings Approx:     {params_embed / 1e9:.3f} B")
    print(f"  LM Heads Approx:       {params_heads / 1e9:.3f} B (Tied weights counted in Embeddings)")
    print(f"  Norms Approx:          {params_norms / 1e9:.3f} B")
    print(f"  Encoders Approx:       {params_encoders / 1e9:.3f} B (Placeholders)")
    print(f"Note: Actual VQ-GAN parameters are separate and assumed pre-trained.")

    # --- Conceptual Usage Example ---
    print("\n--- Conceptual Usage ---")
    # model.eval() # Set to eval mode for generation

    # VQGAN needed for actual image generation/display
    # vqgan = VQGANTokenizer(config)

    # Example 1: Text Generation
    # prompt_text = "The latest discovery in astrophysics is"
    # tokenizer = ... # Load your tokenizer
    # input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    # generated_ids = model.generate(input_ids, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id)
    # generated_text = tokenizer.decode(generated_ids)
    # print(f"Generated Text: {generated_text}")

    # Example 2: Image Generation (Highly Conceptual)
    # prompt_for_image = "Generate an image of a futuristic cityscape at sunset [IMG_START]"
    # input_ids = tokenizer(prompt_for_image, return_tensors="pt")["input_ids"]
    # Need special token IDs configured in config/tokenizer
    # img_start_token = tokenizer.convert_tokens_to_ids("[IMG_START]")
    # img_end_token = tokenizer.convert_tokens_to_ids("[IMG_END]") # Assuming VQGAN tokens don't overlap
    # generated_ids = model.generate(
    #      input_ids,
    #      max_new_tokens=300, # Needs to be > max_image_tokens
    #      image_token_start_id=img_start_token,
    #      image_token_end_id=img_end_token,
    #      max_image_tokens=config.image_vocab_size # e.g. 256
    # )
    # # Extract image tokens based on start/end tokens
    # image_token_sequence = extract_image_tokens(generated_ids, img_start_token, img_end_token)
    # if image_token_sequence:
    #      image_tokens_tensor = torch.tensor(image_token_sequence).unsqueeze(0)
    #      reconstructed_image = vqgan.decode_from_tokens(image_tokens_tensor)
    #      # Display or save reconstructed_image
    #      print("Image generated (conceptual).")