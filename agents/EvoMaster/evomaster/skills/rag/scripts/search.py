#!/usr/bin/env python3
"""RAG Searcher - vector retrieval utility.

Provides semantic vector search based on FAISS and transformer embeddings.
Supports both local transformer models and the OpenAI embedding API.

Design goal: a generic component for "vector retrieval + (optional) original content fetch".
- Vector retrieval: depends on ``embeddings.npy`` and ``nodes.jsonl`` under ``vec_dir``; whether to load ``faiss.index``
  is explicitly controlled by the caller via ``use_faiss`` (default: not loaded).
- Content retrieval: optionally load ``nodes_data.json`` and extract fields via a dotted ``content_path``.
"""

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any
from abc import ABC, abstractmethod

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None  # Optional: use FAISS index when available, otherwise fall back to embeddings.npy only.

logger = logging.getLogger(__name__)


# ============================================
# Embedding abstraction and implementations
# ============================================

class BaseEmbedder(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Encode text into a vector."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        pass


class LocalTransformerEmbedder(BaseEmbedder):
    """Embedder backed by a local Transformer model (HuggingFace)."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        self.model_name = model_name
        self.device = device
        
        # Load model quietly (suppress stderr noise).
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stderr(devnull):
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Get embedding dimension.
        self._dimension = self.model.config.hidden_size
        logger.info(f"Initialized local transformer embedder: {model_name} on {device}, dim={self._dimension}")
    
    def encode(self, text: str) -> np.ndarray:
        import torch
        
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            h = outputs.last_hidden_state
            attn = inputs["attention_mask"].unsqueeze(-1)
            # Mean pooling with attention weights
            emb = (h * attn).sum(dim=1) / attn.sum(dim=1)
        
        return emb.cpu().numpy()
    
    def get_dimension(self) -> int:
        return self._dimension


class OpenAIEmbedder(BaseEmbedder):
    """Embedder using the OpenAI Embedding API."""
    
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        self.model = model
        self.dimensions = dimensions
        
        # Prefer explicit parameters, otherwise fall back to environment variables.
        self.api_key = api_key or os.environ.get("OPENAI_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_EMBEDDING_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set via parameter, OPENAI_EMBEDDING_API_KEY or OPENAI_API_KEY env var.")
        
        # Initialize OpenAI client.
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        self.client = OpenAI(**client_kwargs)
        
        # Default embedding dimension (text-embedding-3-large defaults to 3072, and can be customized).
        self._dimension = dimensions or 3072
        logger.info(f"Initialized OpenAI embedder: {model}, base_url={base_url}, dim={self._dimension}")
    
    def encode(self, text: str) -> np.ndarray:
        """Call the OpenAI embedding API."""
        kwargs = {
            "model": self.model,
            "input": text,
        }
        # text-embedding-3-* models support the ``dimensions`` parameter.
        if self.dimensions and "text-embedding-3" in self.model:
            kwargs["dimensions"] = self.dimensions
        
        response = self.client.embeddings.create(**kwargs)
        embedding = response.data[0].embedding
        return np.array([embedding], dtype=np.float32)
    
    def get_dimension(self) -> int:
        return self._dimension


def create_embedder(
    model: str | None = None,
    embedding_type: str = "auto",
    api_key: str | None = None,
    base_url: str | None = None,
    dimensions: int | None = None,
    device: str = "cpu",
) -> BaseEmbedder:
    """Create an embedder instance.
    
    Args:
        model: Model name or path.
        embedding_type: One of ``"local"``, ``"openai"``, or ``"auto"`` (auto-detect).
        api_key: OpenAI API key (required only for ``openai`` type).
        base_url: OpenAI API base URL (required only for ``openai`` type).
        dimensions: Embedding dimension (only supported by OpenAI text-embedding-3-* models).
        device: Compute device (only relevant for ``local`` type).
    
    Returns:
        A ``BaseEmbedder`` instance.
    """
    # Auto-detect embedding type.
    if embedding_type == "auto":
        if model and ("text-embedding" in model or model.startswith("openai/")):
            embedding_type = "openai"
        elif api_key or os.environ.get("OPENAI_EMBEDDING_API_KEY"):
            embedding_type = "openai"
        else:
            embedding_type = "local"
    
    if embedding_type == "openai":
        return OpenAIEmbedder(
            model=model or "text-embedding-3-large",
            api_key=api_key,
            base_url=base_url,
            dimensions=dimensions,
        )
    else:
        # Local model: do not fall back to any project-internal default path; require explicit model name/path.
        if not model:
            raise ValueError(
                "Local embedding requires an explicit 'model' name/path. "
                "Please configure it in your embedding settings or CLI arguments."
            )
        return LocalTransformerEmbedder(
            model_name=model,
            device=device,
        )


def _find_project_root() -> Path:
    """Find the project root directory (the directory that contains ``evomaster``)."""
    script_path = Path(__file__).resolve()
    current = script_path.parent
    while current != current.parent:
        if (current / "evomaster").exists() and (current / "evomaster").is_dir():
            return current
        current = current.parent
    cwd = Path.cwd()
    current = cwd
    while current != current.parent:
        if (current / "evomaster").exists() and (current / "evomaster").is_dir():
            return current
        current = current.parent
    if "EvoMaster_ROOT" in os.environ:
        root = Path(os.environ["EvoMaster_ROOT"])
        if root.exists() and (root / "evomaster").exists():
            return root
    raise RuntimeError(
        "Failed to locate project root. Make sure you are running inside the EvoMaster project structure, "
        "or set the EvoMaster_ROOT environment variable."
    )


def _resolve_path(path_str: str, project_root: Path | None = None) -> Path:
    """Resolve a path string to an absolute path; ``evomaster/``-prefixed paths are resolved relative to the project root."""
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    path_str_normalized = str(path).replace("\\", "/")
    if path_str_normalized.startswith("evomaster/"):
        if project_root is None:
            project_root = _find_project_root()
        return (project_root / path).resolve()
    return path.resolve()


class RAGSearcher:
    """
    Generic RAG Searcher providing vector retrieval capabilities.
    Supports both local transformer models and the OpenAI embedding API.
    """

    def __init__(
        self,
        vec_dir: str,
        model_name: str | None = None,
        nodes_data_json: str | None = None,
        device: str = "cpu",
        node_id_key: str = "node_id",
        use_faiss: bool = False,
        # OpenAI embedding parameters
        embedding_type: str = "auto",
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        embedding_dimensions: int | None = None,
    ):
        """Initialize the RAG Searcher.
        
        Args:
            vec_dir: Vector database directory path (must contain ``embeddings.npy`` and ``nodes.jsonl``; whether to use
                ``faiss.index`` is controlled by ``use_faiss``).
            model_name: Model name for encoding (local path or OpenAI model name).
            nodes_data_json: Path to a JSON file with node data (optional, used to fetch knowledge/content).
            device: Compute device (``'cpu'`` or ``'cuda'``), used only for local models.
            node_id_key: Field name in each ``nodes.jsonl`` JSON object used as the ID (default: ``'node_id'``).
            use_faiss: Whether to load and use ``faiss.index`` under ``vec_dir`` (default False; only used when FAISS
                is installed and the file exists).
            embedding_type: One of ``"local"``, ``"openai"``, or ``"auto"`` (auto-detect).
            embedding_api_key: OpenAI API key (only required for ``openai`` type).
            embedding_base_url: OpenAI API base URL (only required for ``openai`` type).
            embedding_dimensions: Embedding dimension (only supported by OpenAI text-embedding-3-* models).
        """
        self.vec_dir = Path(vec_dir)
        self.model_name = model_name
        self.device = device
        self.node_id_key = node_id_key

        # Load FAISS index only when use_faiss=True, FAISS is available, and the index file exists.
        index_path = self.vec_dir / "faiss.index"
        if use_faiss and faiss is not None and index_path.exists():
            self.index = faiss.read_index(str(index_path))
            logger.info(f"Loaded FAISS index from {index_path}")
        else:
            self.index = None
            if use_faiss:
                if faiss is None:
                    logger.warning("use_faiss=True but FAISS not installed, using embeddings.npy only")
                elif not index_path.exists():
                    logger.warning(f"use_faiss=True but FAISS index not found at {index_path}, using embeddings.npy only")

        # Load embeddings and pre-compute normalized vectors (for cosine similarity).
        emb_path = self.vec_dir / "embeddings.npy"
        if emb_path.exists():
            self.emb = np.load(emb_path)
            # When there is a single embedding, it is 1D (dim,) and must be reshaped to (1, dim) for downstream computation.
            if self.emb.ndim == 1:
                self.emb = self.emb.reshape(1, -1)
            norms = np.linalg.norm(self.emb, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self.emb_normalized = self.emb / norms
            logger.info(f"Loaded embeddings from {emb_path}, shape={self.emb.shape}")
        else:
            self.emb = None
            self.emb_normalized = None
            logger.warning(f"Embeddings file not found: {emb_path}")

        # Load node_id mapping.
        nodes_jsonl_path = self.vec_dir / "nodes.jsonl"
        self.node_ids = []
        if nodes_jsonl_path.exists():
            with open(nodes_jsonl_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if line.strip():
                        obj = json.loads(line)
                        # Prefer the configured node_id_key; if missing, try task_name; otherwise fall back to the index.
                        if self.node_id_key in obj:
                            node_id = obj[self.node_id_key]
                        elif "task_name" in obj:
                            node_id = obj["task_name"]
                            logger.debug(f"Using 'task_name' as node_id for line {idx}: {node_id}")
                        else:
                            # Use the index as node_id.
                            node_id = str(idx)
                            logger.debug(f"Using index as node_id for line {idx}: {node_id}")
                        self.node_ids.append(node_id)
            logger.info(f"Loaded {len(self.node_ids)} node IDs from {nodes_jsonl_path}")
        else:
            logger.warning(f"Nodes JSONL file not found: {nodes_jsonl_path}")

        # Load nodes_data (if provided).
        self.nodes_data = {}
        if nodes_data_json:
            nodes_data_path = Path(nodes_data_json)
            if nodes_data_path.exists():
                with open(nodes_data_path, "r", encoding="utf-8") as f:
                    self.nodes_data = json.load(f)
                logger.info(f"Loaded nodes data from {nodes_data_path}")
            else:
                logger.warning(f"Nodes data file not found: {nodes_data_path}")

        # Initialize embedding model (supports both local models and the OpenAI API).
        self.embedder = create_embedder(
            model=model_name,
            embedding_type=embedding_type,
            api_key=embedding_api_key,
            base_url=embedding_base_url,
            dimensions=embedding_dimensions,
            device=device,
        )

    @staticmethod
    def _get_by_dotted_path(obj: Any, dotted_path: str, default: Any = None) -> Any:
        """Get a value from a dict/object via a dotted path, e.g. ``'content.text'``."""
        if dotted_path is None or dotted_path == "":
            return obj
        cur: Any = obj
        for key in dotted_path.split("."):
            if cur is None:
                return default
            if isinstance(cur, dict):
                cur = cur.get(key, default)
            else:
                cur = getattr(cur, key, default)
        return cur

    def _default_content_candidates(self) -> list[str]:
        # Common fallback fields (kept generic to avoid binding to any specific project).
        return [
            "content.text",
            "content.page_content",
            "content.knowledge",
            "content.data",
            "content",
            "text",
            "page_content",
            "knowledge",
            "data",
        ]

    def encode(self, text: str) -> np.ndarray:
        """Encode text into a vector.
        
        Args:
            text: Input text.
        
        Returns:
            Encoded vector (numpy array).
        """
        return self.embedder.encode(text)

    def search_similar(
        self,
        query_emb: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float | None = None
    ) -> list[tuple[str, float]]:
        """Search for similar nodes using cosine similarity.
        
        Args:
            query_emb: Query embedding vector.
            top_k: Number of results to return.
            similarity_threshold: Similarity threshold; results below this value are filtered out (range -1 to 1).
        
        Returns:
            A list of ``(node_id, cosine_similarity)`` tuples, sorted by similarity in descending order.
        """
        if len(self.node_ids) == 0:
            logger.warning("No node IDs loaded, returning empty results")
            return []

        if self.emb_normalized is None:
            raise ValueError("Embeddings not loaded, cannot compute cosine similarity")

        # Ensure query_emb is 1D.
        if query_emb.ndim == 2:
            query_emb = query_emb[0]

        # Normalize query embedding.
        q_norm = np.linalg.norm(query_emb)
        if q_norm == 0:
            logger.warning("Query embedding has zero norm")
            return []
        query_normalized = query_emb / q_norm

        # Compute cosine similarity.
        similarities = self.emb_normalized @ query_normalized.astype("float32")

        # Take top_k results.
        top_k = min(top_k, len(self.node_ids))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if idx < 0 or idx >= len(self.node_ids):
                continue

            sim = float(similarities[idx])

            if similarity_threshold is not None and sim < similarity_threshold:
                logger.debug(
                    f"Filtered out node {self.node_ids[idx]} with similarity {sim:.4f} "
                    f"below threshold {similarity_threshold}"
                )
                continue

            logger.debug(
                f"Selected node {self.node_ids[idx]} with similarity {sim:.4f}"
            )
            results.append((self.node_ids[idx], sim))

        return results

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float | None = None
    ) -> list[tuple[str, float]]:
        """Search with a text query (convenience method).
        
        Args:
            query_text: Query text.
            top_k: Number of results to return.
            similarity_threshold: Similarity threshold (range -1 to 1).
        
        Returns:
            A list of ``(node_id, cosine_similarity)`` tuples, sorted by similarity in descending order.
        """
        query_emb = self.encode(query_text)
        return self.search_similar(query_emb, top_k=top_k, similarity_threshold=similarity_threshold)

    def get_knowledge(self, node_id: str) -> Any:
        """Get the knowledge content of a node.
        
        Args:
            node_id: Node ID.
        
        Returns:
            The content of the node (format depends on the structure of nodes_data).
        
        Note:
            To remain generic, this method does not bind to any specific field (such as ``improve_knowledge``).
            For precise field selection, use ``get_knowledge_by_path()`` instead.
        """
        if not self.nodes_data:
            logger.warning("Nodes data not loaded, cannot retrieve knowledge")
            return None

        node = self.nodes_data.get(str(node_id), {})
        # Fallback: try common fields; if none work, return the whole node.
        for path in self._default_content_candidates():
            val = self._get_by_dotted_path(node, path, default=None)
            if val not in (None, "", [], {}):
                return val
        return node

    def get_knowledge_by_path(self, node_id: str, content_path: str, default: Any = None) -> Any:
        """Extract a node content field via a dotted path (generic)."""
        if not self.nodes_data:
            logger.warning("Nodes data not loaded, cannot retrieve knowledge")
            return default
        node = self.nodes_data.get(str(node_id), {})
        return self._get_by_dotted_path(node, content_path, default=default)

    def get_node_data(self, node_id: str) -> dict | None:
        """Get the full data dictionary for a node.
        
        Args:
            node_id: Node ID.
        
        Returns:
            Full node data dict, or None if not found.
        """
        if not self.nodes_data:
            return None
        return self.nodes_data.get(str(node_id))


def main():
    """Command-line interface example."""
    import argparse

    parser = argparse.ArgumentParser(description="RAG Searcher CLI")
    parser.add_argument("--vec_dir", required=True, help="Vector database directory")
    parser.add_argument(
        "--model",
        default=None,
        help="Embedding model path, HuggingFace model name, or OpenAI model name "
             "(required for local embedding; optional for OpenAI when using defaults)",
    )
    parser.add_argument("--nodes_data", help="Nodes data JSON file")
    parser.add_argument(
        "--node_id_key",
        default="node_id",
        help="ID key name in nodes.jsonl per-line JSON (default: node_id)",
    )
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results")
    parser.add_argument("--threshold", type=float, help="Cosine similarity threshold (range -1 to 1, higher = more similar)")
    parser.add_argument(
        "--content_path",
        default=None,
        help="Dotted path to extract content from nodes_data (e.g. content.text). "
             "If omitted, uses a set of common fallback fields.",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
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
    parser.add_argument(
        "--use_faiss",
        action="store_true",
        help="Load and use faiss.index from vec_dir when available (default: off; use for large-scale search)",
    )

    args = parser.parse_args()

    # Path resolution: ``evomaster/`` is relative to the project root.
    project_root = _find_project_root()
    vec_dir_resolved = str(_resolve_path(args.vec_dir, project_root))
    nodes_data_resolved = str(_resolve_path(args.nodes_data, project_root)) if args.nodes_data else None
    
    # Resolve only local model paths.
    model_resolved = args.model
    if args.embedding_type != "openai" and str(args.model).replace("\\", "/").startswith("evomaster/"):
        model_resolved = str(_resolve_path(args.model, project_root))

    # Initialize searcher.
    searcher = RAGSearcher(
        vec_dir=vec_dir_resolved,
        model_name=model_resolved,
        nodes_data_json=nodes_data_resolved,
        node_id_key=args.node_id_key,
        use_faiss=args.use_faiss,
        embedding_type=args.embedding_type,
        embedding_api_key=args.embedding_api_key,
        embedding_base_url=args.embedding_base_url,
        embedding_dimensions=args.embedding_dimensions,
    )

    # Run search.
    results = searcher.search_by_text(
        query_text=args.query,
        top_k=args.top_k,
        similarity_threshold=args.threshold
    )

    if args.output == "json":
        payload = {
            "query": args.query,
            "results": [
                {
                    "node_id": node_id,
                    "similarity": similarity,
                    "content": (
                        searcher.get_knowledge_by_path(node_id, args.content_path)
                        if (args.nodes_data and args.content_path)
                        else (searcher.get_knowledge(node_id) if args.nodes_data else None)
                    ),
                }
                for (node_id, similarity) in results
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    # Text output.
    print(f"\nSearch results for: '{args.query}'")
    print("=" * 60)
    for i, (node_id, similarity) in enumerate(results, 1):
        print(f"\n{i}. Node ID: {node_id}")
        print(f"   Similarity: {similarity:.4f}")

        if args.nodes_data:
            if args.content_path:
                content = searcher.get_knowledge_by_path(node_id, args.content_path)
            else:
                content = searcher.get_knowledge(node_id)
            if content not in (None, "", [], {}):
                print(f"   Content: {content}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
