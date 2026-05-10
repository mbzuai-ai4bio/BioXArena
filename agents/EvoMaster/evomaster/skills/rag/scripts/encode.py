#!/usr/bin/env python3
"""Text Encoder - standalone text-to-embedding utility.

Provides independent text encoding functionality, converting text into vectors.
Supports both local transformer models and the OpenAI embedding API.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# Import embedder-related classes from search.py.
from search import create_embedder, BaseEmbedder


class TextEncoder:
    """Text encoder supporting both local models and the OpenAI API."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
        embedding_type: str = "auto",
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        embedding_dimensions: int | None = None,
    ):
        """Initialize the encoder.
        
        Args:
            model_name: Model name or path.
            device: Compute device (``'cpu'`` or ``'cuda'``), used only for local models.
            embedding_type: One of ``"local"``, ``"openai"``, or ``"auto"`` (auto-detect).
            embedding_api_key: OpenAI API key (required only for ``openai`` type).
            embedding_base_url: OpenAI API base URL (required only for ``openai`` type).
            embedding_dimensions: Embedding dimension (only supported by OpenAI text-embedding-3-* models).
        """
        self.model_name = model_name
        self.device = device
        
        # Use the shared embedder creation helper.
        self.embedder = create_embedder(
            model=model_name,
            embedding_type=embedding_type,
            api_key=embedding_api_key,
            base_url=embedding_base_url,
            dimensions=embedding_dimensions,
            device=device,
        )
        logger.info(f"Initialized encoder with model: {model_name}")

    def encode(
        self,
        text: str,
        max_length: int = 512,
        normalize: bool = False
    ) -> np.ndarray:
        """Encode a single piece of text.
        
        Args:
            text: Input text.
            max_length: Max length (only used by local models).
            normalize: Whether to L2-normalize the vector.
        
        Returns:
            Encoded vector.
        """
        emb = self.embedder.encode(text)
        
        # Ensure a 1D vector.
        if emb.ndim > 1:
            emb = emb[0]

        # Optional normalization.
        if normalize:
            norm = np.linalg.norm(emb)
            emb = emb / (norm + 1e-8)

        return emb

    def encode_batch(
        self,
        texts: list[str],
        max_length: int = 512,
        normalize: bool = False,
        batch_size: int = 32
    ) -> np.ndarray:
        """Encode a batch of texts.
        
        Args:
            texts: List of texts.
            max_length: Max length (only used by local models).
            normalize: Whether to L2-normalize vectors.
            batch_size: Batch size.
        
        Returns:
            Array of encoded vectors with shape ``(n_texts, embedding_dim)``.
        """
        all_embeddings = []

        for text in texts:
            emb = self.encode(text, max_length=max_length, normalize=normalize)
            all_embeddings.append(emb)

        return np.vstack(all_embeddings)


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Text Encoder CLI")
    parser.add_argument(
        "--model",
        default=None,
        help="Embedding model path, HuggingFace model name, or OpenAI model name "
             "(required for local embedding; optional for OpenAI when using defaults)",
    )
    parser.add_argument("--text", help="Text to encode")
    parser.add_argument("--file", help="File containing text (one per line)")
    parser.add_argument("--output", help="Output file for embeddings (.npy)")
    parser.add_argument("--max_length", type=int, default=512, help="Max length")
    parser.add_argument("--normalize", action="store_true", help="Normalize vectors")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    # OpenAI embedding parameters.
    parser.add_argument(
        "--embedding_type",
        choices=["auto", "local", "openai"],
        default="auto",
        help="Embedding type: 'local' for transformer models, 'openai' for OpenAI API, 'auto' to detect (default: auto)",
    )
    parser.add_argument(
        "--embedding_api_key",
        help="OpenAI API key for embedding (can also use OPENAI_EMBEDDING_API_KEY env var)",
    )
    parser.add_argument(
        "--embedding_base_url",
        help="OpenAI API base URL for embedding (can also use OPENAI_EMBEDDING_BASE_URL env var)",
    )
    parser.add_argument(
        "--embedding_dimensions",
        type=int,
        help="Embedding dimensions for text-embedding-3-* models (default: 3072)",
    )

    args = parser.parse_args()

    # Initialize encoder.
    encoder = TextEncoder(
        model_name=args.model,
        embedding_type=args.embedding_type,
        embedding_api_key=args.embedding_api_key,
        embedding_base_url=args.embedding_base_url,
        embedding_dimensions=args.embedding_dimensions,
    )

    # Read input text.
    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # Read from stdin.
        texts = [line.strip() for line in sys.stdin if line.strip()]

    if not texts:
        print("Error: No text provided", file=sys.stderr)
        sys.exit(1)

    # Encode.
    if len(texts) == 1:
        embedding = encoder.encode(texts[0], max_length=args.max_length, normalize=args.normalize)
    else:
        embedding = encoder.encode_batch(
            texts,
            max_length=args.max_length,
            normalize=args.normalize,
            batch_size=args.batch_size
        )

    # Output.
    if args.output:
        np.save(args.output, embedding)
        print(f"Saved embeddings to {args.output}")
    else:
        # Print to stdout (human-readable format).
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding:\n{embedding}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
