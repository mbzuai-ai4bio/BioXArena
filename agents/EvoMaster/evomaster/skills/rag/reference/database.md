## `scripts/database.py`: Vector Database Builder Interface (Placeholder)

`database.py` (with internal class `VectorDatabaseBuilder`) defines an **interface for building and managing a vector database**. In the current version it mainly serves to:

- Standardize an entry point for “how to build/maintain a vector database in code”.
- Reserve a clear API shape for future concrete implementations.

Most methods currently raise `NotImplementedError`. They act as placeholders and documentation of the intended API, rather than a ready-to-use production data builder.

### Role and Responsibilities

`VectorDatabaseBuilder` is intended to provide a business-agnostic “vector store builder service” layer. Typical responsibilities include:

- Building a vector store from raw documents (text, JSON, code, etc.).
- Incrementally adding documents to an existing vector store.
- Maintaining or rebuilding indexes.
- Deleting specified nodes.
- Querying statistics (vector count, dimension, index type, etc.).

### Class and Method Overview

#### `class VectorDatabaseBuilder`

Constructor:

- `output_dir: str`:
  - Root directory where the vector store will be written.
  - The constructor automatically creates this directory.
- `model_name: str = "your-local-embedding-model"`:
  - Name or path of the model used for encoding (for example, a HuggingFace model name such as `sentence-transformers/all-mpnet-base-v2`, or a local directory path).
  - The actual encoding logic is not implemented in this file yet; later you can reuse the embedder logic from `encode.py` / `search.py`.
- `device: str = "cpu"`:
  - Device on which the model runs, can be extended to `"cuda"`, etc.

Main methods (all placeholder for now):

- `build_from_documents(documents, chunk_size=1000, chunk_overlap=200, **kwargs)`:
  - Intended to build a new vector store from a list of raw documents.
  - `documents` is expected to be a list of dicts containing `content` and `metadata`.
- `add_documents(documents, **kwargs)`:
  - Intended to incrementally add documents to an existing vector store.
- `update_index(**kwargs)`:
  - Intended to rebuild or optimize the index structure (e.g., switch to IVF/HNSW, etc.).
- `delete_documents(node_ids, **kwargs)`:
  - Intended to delete nodes by `node_id`.
- `get_stats() -> dict`:
  - Intended to return basic statistics of the vector store.

Currently, each of these methods will:

- Log a `logger.warning` indicating this is a placeholder API.
- Raise `NotImplementedError`.

### Command-Line Interface (Conceptual)

`database.py` also exposes a lightweight CLI entry point that can be used to trigger build flows from the terminal when needed. Conceptually it:

- Accepts an output directory (`--output_dir`).
- Accepts a model name or path for encoding (`--model`).
- Accepts an action parameter (`--action`) to distinguish between building, incremental adding, or viewing stats.

How exactly this entry point is integrated and invoked in your project is up to the host system. This reference only documents parameter semantics and does not provide concrete CLI usage examples.

Common parameters:

- `--output_dir` (required): Output directory for the vector store.
- `--model`: Model name or path used for encoding (defaults to local `all-mpnet-base-v2`).
- `--action`:
  - `build`: Prints “Building database...” along with a not-implemented warning.
  - `add`: Prints “Adding documents...” along with a not-implemented warning.
  - `stats`: Prints “Getting stats...” along with a not-implemented warning.

### How to Extend This in Your Own Project

If you need to build a vector store programmatically in your own RAG project, you can implement this interface concretely. A recommended workflow:

1. **Define data sources and schema**
   - Clearly define the structure of `content` and `metadata` in `documents`.
   - Decide how to name node IDs (e.g., `task_name` or UUID).
2. **Reuse embedding logic**
   - Use `TextEncoder` from `encode.py` or `create_embedder` from `search.py` to generate embeddings.
3. **Write a standard vector store layout**
   - Write embeddings to `embeddings.npy` (e.g. via `encode.py` or your own encoding loop).
   - Generate `nodes.jsonl` to record node IDs and metadata.
   - To enable search with `--use_faiss`, run `build_faiss.py` via run_script with `--vec_dir <output_dir>` (see `reference/build_faiss.md`). This reads `embeddings.npy` and writes `faiss.index`.
   - Optionally write `nodes_data` (JSON file) to be used by `search.py` for content retrieval.
4. **Fill in actual logic inside `VectorDatabaseBuilder`**
   - Implement the full build flow in `build_from_documents`.
   - Implement add/delete logic in `add_documents` / `delete_documents`.
   - Implement index rebuild or parameter tuning in `update_index`.
   - Implement `get_stats` to return information such as vector count, dimension, index type, etc.
