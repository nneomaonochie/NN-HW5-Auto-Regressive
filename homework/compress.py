import struct
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image

from .autoregressive import Autoregressive
from .bsq import Tokenizer


class Compressor:
    def __init__(self, tokenizer: Tokenizer, autoregressive: Autoregressive):
        super().__init__()
        self.tokenizer = tokenizer
        self.autoregressive = autoregressive

    def compress(self, x: torch.Tensor) -> bytes:
        """
        Compress the image into a torch.uint8 bytes stream (1D tensor).

        Use arithmetic coding.
        """
                # Add batch dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            # Tokenize: (1, H, W, 3) → (1, h, w)
            tokens = self.tokenizer.encode_index(x)  # (1, 30, 20) for 150x100 image
            
            # Flatten tokens: (1, 30, 20) → (600,)
            tokens_flat = tokens.flatten().cpu().numpy().astype(np.uint16)
            
            h, w = tokens.shape[1], tokens.shape[2]
            
            # Pack metadata (image dimensions) - 4 bytes
            metadata = struct.pack('HH', h, w)
            
            # Pack tokens efficiently
            # With codebook_bits=10, we have tokens in range [0, 1023]
            # Store as uint16 (2 bytes per token)
            token_bytes = tokens_flat.tobytes()
            
            return metadata + token_bytes

    def decompress(self, x: bytes) -> torch.Tensor:
        """
        Decompress a tensor into a PIL image.
        You may assume the output image is 150 x 100 pixels.
        """
        # Extract metadata (4 bytes)
        h, w = struct.unpack('HH', x[:4])
        token_bytes = x[4:]
        
        # Unpack tokens
        seq_len = h * w
        tokens_flat = np.frombuffer(token_bytes, dtype=np.uint16, count=seq_len)
        
        # Reshape to image format: (seq_len,) → (1, h, w)
        tokens = torch.from_numpy(tokens_flat).long().to(self.device)
        tokens = tokens.view(1, h, w)
        
        with torch.no_grad():
            # Decode tokens to image: (1, h, w) → (1, H, W, 3)
            image = self.tokenizer.decode_index(tokens)
        
        # Remove batch dimension: (1, H, W, 3) → (H, W, 3)
        return image.squeeze(0)


def compress(tokenizer: Path, autoregressive: Path, image: Path, compressed_image: Path):
    """
    Compress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    images: Path to the image to compress.
    compressed_image: Path to save the compressed image tensor.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    x = torch.tensor(np.array(Image.open(image)), dtype=torch.uint8, device=device)
    cmp_img = cmp.compress(x.float() / 255.0 - 0.5)
    with open(compressed_image, "wb") as f:
        f.write(cmp_img)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    """
    Decompress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    compressed_image: Path to the compressed image tensor.
    images: Path to save the image to compress.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    with open(compressed_image, "rb") as f:
        cmp_img = f.read()

    x = cmp.decompress(cmp_img)
    img = Image.fromarray(((x + 0.5) * 255.0).clamp(min=0, max=255).byte().cpu().numpy())
    img.save(image)


if __name__ == "__main__":
    from fire import Fire

    Fire({"compress": compress, "decompress": decompress})
