import abc

import torch
import torch.nn as nn

from .ae import PatchAutoEncoder


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        
        # Claude Sonnet 4.5
        self._codebook_bits = codebook_bits
        self.embedding_dim = embedding_dim
        
        # Down-projection: embedding_dim -> codebook_bits
        self.down_proj = nn.Linear(embedding_dim, codebook_bits)
        
        # Up-projection: codebook_bits -> embedding_dim
        self.up_proj = nn.Linear(codebook_bits, embedding_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ encoder:
        - A linear down-projection into codebook_bits dimensions
        - L2 normalization
        - differentiable sign
        """
        
        # Claude Sonnet 4.5
        # 1. Down-project to codebook_bits dimensions
        x = self.down_proj(x)
        
        # 2. L2 normalization (per feature vector)
        # Normalize along the last dimension
        x = x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-8)
        
        # 3. Differentiable binarization (sign with straight-through)
        x = diff_sign(x)
        
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ decoder:
        - A linear up-projection into embedding_dim should suffice
        """
        
        # Claude Sonnet 4.5
        # Up-project back to embedding_dim
        return self.up_proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run BQS and encode the input tensor x into a set of integer tokens
        """
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of integer tokens into an image.
        """
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * 2 ** torch.arange(x.size(-1)).to(x.device)).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self._codebook_bits).to(x.device))) > 0).float() - 1


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """
    Combine your PatchAutoEncoder with BSQ to form a Tokenizer.

    Hint: The hyper-parameters below should work fine, no need to change them
          Changing the patch-size of codebook-size will complicate later parts of the assignment.
    """

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim)
        
        # Claude Sonnet 4.5
        self.codebook_bits = codebook_bits
        
        # BSQ quantizer - operates on the bottleneck dimension from parent
        # The parent's bottleneck dimension should match latent_dim for BSQ to work
        self.bsq = BSQ(codebook_bits=codebook_bits, embedding_dim=self.bottleneck)

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        # Claude Sonnet 4.5
        # 1. Encode to features
        z = super().encode(x)  # (B, h, w, latent_dim)
        
        # 2. Convert to tokens
        tokens = self.bsq.encode_index(z)  # (B, h, w)
        
        return tokens

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        #Claude Sonnet 4.5
        # 1. Convert tokens to features
        z = self.bsq.decode_index(x)  # (B, h, w, latent_dim)
        
        # 2. Decode to image
        return super().decode(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Claude Sonnet 4.5
        # 1. Run through patch encoder
        z = super().encode(x)  # (B, h, w, latent_dim)
        
        # 2. Apply BSQ quantization
        z_quantized = self.bsq(z)  # (B, h, w, latent_dim)
        
        return z_quantized

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        # Claude Sonnet 4.5
        return super().decode(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        Hint: It can be helpful to monitor the codebook usage with

              cnt = torch.bincount(self.encode_index(x).flatten(), minlength=2**self.codebook_bits)

              and returning

              {
                "cb0": (cnt == 0).float().mean().detach(),
                "cb2": (cnt <= 2).float().mean().detach(),
                ...
              }
        """
        # Claude Sonnet 4.5
        # Encode and decode
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        
        # Monitor codebook usage
        with torch.no_grad():
            # Get token indices
            tokens = self.encode_index(x).flatten()
            
            # Count usage of each code
            cnt = torch.bincount(tokens, minlength=2**self.codebook_bits)
            
            # Track statistics
            loss_dict = {
                "cb0": (cnt == 0).float().mean(),      # Fraction of unused codes
                "cb2": (cnt <= 2).float().mean(),       # Fraction of rarely used codes
                "cb10": (cnt <= 10).float().mean(),     # Fraction of codes used ≤10 times
                "cb_mean": cnt.float().mean(),          # Average usage
                "cb_max": cnt.float().max(),            # Max usage
            }
        
        return x_reconstructed, loss_dict
