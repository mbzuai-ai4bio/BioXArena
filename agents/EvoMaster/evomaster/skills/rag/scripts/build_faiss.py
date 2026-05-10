#!/usr/bin/env python3
"""Build FAISS index from embeddings.npy.

Generate ``faiss.index`` from ``embeddings.npy`` under ``vec_dir`` for ``search.py`` when ``--use_faiss`` is enabled.
All vectors are L2-normalized first and then written into an ``IndexFlatIP`` index (inner product = cosine similarity).
"""

import logging
import sys
from pathlib import Path

import numpy as np

# Use the same path resolution helpers as in search.py.
from search import _find_project_root, _resolve_path

logger = logging.getLogger(__name__)


def build_faiss_index(vec_dir: str | Path, project_root: Path | None = None) -> Path:
    """Build ``vec_dir/faiss.index`` from ``vec_dir/embeddings.npy``.

    Args:
        vec_dir: Vector store directory containing ``embeddings.npy``.
        project_root: Project root used to resolve ``evomaster/``-prefixed relative paths; auto-detected if None.

    Returns:
        Path to the written ``faiss.index`` file.

    Raises:
        FileNotFoundError: ``embeddings.npy`` does not exist.
        ImportError: ``faiss`` package is not installed.
        RuntimeError: Vector dimension/shape is invalid.
    """
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "Building faiss.index requires the faiss package. "
            "Install with: pip install faiss-cpu  (or faiss-gpu for GPU)"
        ) from None

    vec_dir = Path(vec_dir)
    if not vec_dir.is_absolute() and str(vec_dir).replace("\\", "/").startswith("evomaster/"):
        root = project_root or _find_project_root()
        vec_dir = _resolve_path(str(vec_dir), root)

    vec_dir = vec_dir.resolve()
    emb_path = vec_dir / "embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")

    emb = np.load(emb_path)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    if emb.ndim != 2:
        raise RuntimeError(f"Expected 2D array (n_vectors, dim), got shape {emb.shape}")

    # Normalize so that inner product corresponds to cosine similarity.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb = emb.astype(np.float32) / norms

    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    out_path = vec_dir / "faiss.index"
    faiss.write_index(index, str(out_path))
    logger.info(f"Built faiss.index from {emb_path} (shape {emb.shape}) -> {out_path}")
    return out_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build faiss.index from embeddings.npy in a vector store directory."
    )
    parser.add_argument(
        "--vec_dir",
        required=True,
        help="Vector store directory containing embeddings.npy (evomaster/ prefix resolved relative to project root)",
    )
    args = parser.parse_args()

    project_root = _find_project_root()
    vec_dir_resolved = str(_resolve_path(args.vec_dir, project_root))

    try:
        build_faiss_index(vec_dir_resolved, project_root=project_root)
        print(f"Done. faiss.index written to {vec_dir_resolved}/faiss.index")
    except (FileNotFoundError, ImportError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
