import abc

import torch
import torch.nn as nn


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10, num_layers: int = 4, nhead: int = 8, dim_feedforward: int = 512):
        super().__init__()
        
        # Claude Sonnet 4.5
        self.d_latent = d_latent
        self.n_tokens = n_tokens
        
        # 1. Token embedding: maps integer tokens to d_latent dimensions
        self.token_embedding = nn.Embedding(n_tokens, d_latent)
        
        # 2. Optional: Positional embedding (can help but not required)
        # For images up to 30x20 = 600 tokens
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, d_latent) * 0.02)
        
        # 3. Start token embedding (for the first prediction)
        self.start_token = nn.Parameter(torch.randn(1, 1, d_latent))
        
        # 4. Transformer encoder layers (used as decoder with causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.0,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Output projection: d_latent -> n_tokens (logits for each token)
        self.output_proj = nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        #print("Input shape:", x.shape) # used for debugging
        # Claude Sonnet 4.5
        B, h, w = x.shape
        seq_len = h * w
        
        # 1. Flatten tokens: (B, h, w) -> (B, seq_len)
        x_flat = x.view(B, seq_len)
        
        # 2. Embed tokens: (B, seq_len) -> (B, seq_len, d_latent)
        x_embed = self.token_embedding(x_flat)
        
        # 3. Add positional embeddings
        x_embed = x_embed + self.pos_embedding[:, :seq_len, :]
        
        # 4. Shift right by 1: prepend start token, remove last token
        # This ensures position i predicts token i (doesn't see it as input)
        start_tokens = self.start_token.expand(B, -1, -1)  # (B, 1, d_latent)
        x_shifted = torch.cat([start_tokens, x_embed[:, :-1, :]], dim=1)  # (B, seq_len, d_latent)
        
        # 5. Create causal mask (upper triangular)
        # Mask shape: (seq_len, seq_len)
        # True means "cannot attend", False means "can attend"
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        
        # 6. Apply transformer with causal mask
        x_transformed = self.transformer(x_shifted, mask=causal_mask, is_causal=True)
        
        # 7. Project to logits: (B, seq_len, d_latent) -> (B, seq_len, n_tokens)
        logits_flat = self.output_proj(x_transformed)
        
        # 8. Reshape back to image format: (B, seq_len, n_tokens) -> (B, h, w, n_tokens)
        logits = logits_flat.view(B, h, w, self.n_tokens)
        
        return logits, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        # Claude Sonnet 4.5
        if device is None:
            device = next(self.parameters()).device
        
        seq_len = h * w
        
        # Start with empty sequence
        generated = torch.zeros(B, seq_len, dtype=torch.long, device=device)
        
        # Generate tokens one by one
        for i in range(seq_len):
            # Get embeddings for generated tokens so far
            if i == 0:
                # First token: just use start token
                x_embed = self.start_token.expand(B, 1, -1)
            else:
                # Embed generated tokens
                x_embed = self.token_embedding(generated[:, :i])
                # Add positional embeddings
                x_embed = x_embed + self.pos_embedding[:, :i, :]
                # Prepend start token
                start_tokens = self.start_token.expand(B, -1, -1)
                x_embed = torch.cat([start_tokens, x_embed], dim=1)
            
            # Create causal mask for current sequence length
            current_len = i + 1
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                current_len, device=device
            )
            
            # Forward pass through transformer
            x_transformed = self.transformer(x_embed, mask=causal_mask, is_causal=True)
            
            # Get logits for next token (last position)
            logits = self.output_proj(x_transformed[:, -1, :])  # (B, n_tokens)
            
            # CORRECT (stochastic - creates diverse samples):
            temperature = 1.0
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # Store generated token
            generated[:, i] = next_token

            
        
        # Reshape to image format: (B, seq_len) -> (B, h, w)
        generated = generated.view(B, h, w)
        
        return generated
