## `scripts/search.py`: Vector Search and Content Retrieval

`search.py` provides a general-purpose **vector search plus optional content retrieval** capability, suitable for any vector store that has embeddings and node IDs (FAISS index optional).

### Feature Overview

- **Vector search**: Perform similarity search based on `embeddings.npy` and `nodes.jsonl` under `vec_dir`. Whether to load `faiss.index` is **explicitly controlled** by the caller via `use_faiss` (CLI: `--use_faiss`; API: `use_faiss=True`). Default is off; when on, the index is loaded only if the FAISS package is installed and the file exists (no error if missing).
- **Optional content retrieval**: If you provide `nodes_data` (JSON/JSONL), the script can use retrieved `node_id`s to look up original content.
- **Multiple embedding backends**:
  - Local Transformer models (HuggingFace or local paths).
  - OpenAI Embedding API (e.g., `text-embedding-3-large`).
- **Generic schema design**:
  - No fixed business fields; everything is configurable via parameters such as `node_id_key`, `content_path`, etc.

The core implementation lives in the `RAGSearcher` class, which handles:

- Loading the vector store; a FAISS index is loaded only when `use_faiss=True` and the package is installed and `faiss.index` exists.
- Text encoding (via the shared `create_embedder`).
- Similarity search (`search_similar` / `search_by_text`).
- Accessing content (`get_knowledge` / `get_knowledge_by_path` / `get_node_data`).

### Required Inputs and File Layout

The minimal requirement is a vector directory `vec_dir` containing at least:

- `embeddings.npy`: Precomputed embedding matrix; **required for search** (used for cosine similarity over normalized vectors). If missing, similarity search will raise.
- `nodes.jsonl`: One JSON per line with fields that identify each node (by default `node_id`).

Optional:

- `faiss.index`: FAISS index file. Loaded only when the caller sets `use_faiss=True` (or `--use_faiss`) and the `faiss` package is installed and this file exists; if absent or not requested, search uses `embeddings.npy` only.
- `nodes_data.json`: Node detail file; its structure is completely defined by your application.

The script does not constrain the schema of `nodes_data` as long as it can be accessed via `node_id` or another agreed-upon ID key.

### Usage Pattern (Conceptual)

In real projects, `search.py` is invoked via run_script (e.g. use_skill). This section lists the parameters the script accepts so that upstream embedding configuration (e.g. OpenAI vs. local models) can be passed through correctly.

#### Basic Search (Vector Results Only)

At minimum, calls should provide:

- A vector-store directory `vec_dir`.
- A query text `query`.
- Optional `top_k` (number of returned results) and `threshold` (cosine similarity cutoff, range -1 to 1; results below this value are filtered out).

When you **already have an external configuration that decides the embedding model**, you should forward that information explicitly via `--model`, `--embedding_type` and (optionally) `--embedding_dimensions`.

Common parameters:

- `--vec_dir`: Vector-store directory (required).
- `--query`: Query text (required).
- `--top_k`: Number of results to return, default `5`.
- `--threshold`: Optional cosine similarity threshold (range -1 to 1); results with similarity below this value are filtered out.
- `--output`: `text` or `json`, default `text`.

#### Search and Retrieve Original Content

If you have a node-details file (e.g., `nodes_data.json` or other JSON/JSONL), you can also provide:

- `nodes_data`: Path to the node-details JSON file.
- `content_path`: A dot path used to extract the specific field to return from each node.

Typical parameters:

- `--nodes_data`: Path to the node-details JSON file.
- `--content_path`: Dot path used to pull a specific field from each node, e.g.:
  - `content.text`
  - `content.code`
  - `meta.task_name`
  - `data_knowledge`
  - `model_knowledge`

If `--content_path` is not provided, the script tries a list of common candidate fields (such as `content.text`, `text`, etc.). If none of those exist, the entire node object is returned.

#### Using OpenAI Embeddings

To use OpenAI embeddings (such as `text-embedding-3-large`), set `embedding_type` to `openai` and provide the model name, dimensions, and credentials via parameters or environment variables.

API key / base URL can be passed via parameters `--embedding_api_key`, `--embedding_base_url`, or environment variables `OPENAI_EMBEDDING_API_KEY` / `OPENAI_API_KEY` and `OPENAI_EMBEDDING_BASE_URL` / `OPENAI_BASE_URL`.

**Parameters to forward from upstream config**  
When your upstream config (e.g. EvoMaster embedding block) exposes `model`, `embedding_type`, and `embedding_dimensions`, pass them through in `script_args` as `--model`, `--embedding_type`, and `--embedding_dimensions` so that:

- When `embedding_type` is `"openai"`, the script uses the exact OpenAI embedding model selected by your higher-level config (and not its own local default such as `all-mpnet-base-v2`).
- When `embedding_type` is `"local"`, the script uses the correct local Transformer model path or name.

### Key Parameters at a Glance

- **Vector-store related**
  - `--vec_dir`: Directory of the vector store (required).
  - `--use_faiss`: If set, load and use `faiss.index` from `vec_dir` when FAISS is installed and the file exists (default: off; use for large-scale search).
  - `--nodes_data`: Node-details JSON file (optional).
  - `--node_id_key`: Field name in `nodes.jsonl` used as the node ID, default `node_id`. If missing, the script tries `task_name` or falls back to the line index.
- **Search-related**
  - `--query`: Query text.
  - `--top_k`: Number of top results to return.
  - `--threshold`: Cosine similarity threshold (range -1 to 1); results below this value are filtered out.
- **Content-retrieval related**
  - `--content_path`: Dot path for extracting content from `nodes_data`.
  - If `content_path` is omitted, the script automatically tries a set of common field names.
- **Output-related**
  - `--output`: `text` or `json`.
- **Embedding-related**
  - `--model`: Local model path, HuggingFace model name, or OpenAI model name.
  - `--embedding_type`: `auto` / `local` / `openai`.
  - `--embedding_api_key` / `--embedding_base_url`: For OpenAI.
  - `--embedding_dimensions`: For `text-embedding-3-*` models, allows configuring the embedding dimension.

### Design Principles for Generality

- No assumptions about business-specific schemas:
  - Field names in `nodes.jsonl` and `nodes_data` are fully defined by the user.
  - Adaptation is done via `node_id_key` and `content_path`.
- No hard dependency on any specific model or provider:
  - Supports arbitrary HuggingFace/local Transformer models.
  - Optionally integrates with the OpenAI Embedding API via parameters.
- Easy to migrate across RAG projects:
  - A vector store needs `embeddings.npy` and `nodes.jsonl`; `faiss.index` is optional and loaded only when `use_faiss` is set by the caller.
  - Node-detail files can be any JSON structure.

### Search Metric: Cosine Similarity

`search.py` performs similarity search using **cosine similarity** over normalized embeddings (loaded from `embeddings.npy`). Results are returned as `(node_id, similarity)` with similarity in the range -1 to 1 (higher means more similar).

- **Threshold**: `--threshold` is a **similarity** threshold. Results with similarity **below** this value are filtered out. Typical values are between 0.5 and 0.9 depending on your data; inspect a sample of similarities to choose a reasonable cutoff.
- The script loads a FAISS index from `vec_dir` only when the caller passes `use_faiss=True` (or `--use_faiss`) and the package is installed and the file exists; the current search path uses the precomputed normalized embedding matrix (`embeddings.npy`) for cosine similarity.

### Performance Tuning and Troubleshooting

- **Performance**:
  - With local models, you can choose smaller/faster models (e.g., `all-MiniLM-L6-v2`) to improve speed.
  - For large-scale vector stores, consider IVF- or HNSW-based index structures to accelerate search.
  - For further speedups, you can move FAISS indexes to GPU in your own code (this script stays neutral and does not enforce GPU usage).
- **Memory**:
  - Load `nodes_data` only when needed to avoid loading excessively large JSON files into memory.
  - If you hit GPU memory limits, force CPU usage or reduce batch size.
- **Quality**:
  - Ensure that indexing and querying use **the same embedding model and dimension**.
  - If retrieval quality is poor:
    - Check whether the raw data is clean and has enough context.
    - Check whether the chosen model (dimension, semantic capability) is appropriate.
    - Adjust `top_k` and `threshold` and inspect the similarity distribution.
- **Paths and files**:
  - Make sure `vec_dir` at minimum contains `embeddings.npy` and `nodes.jsonl`. Add `faiss.index` and pass `--use_faiss` only when you want to use it (e.g. large-scale search).
  - If you provide `nodes_data`, ensure its keys line up with the ID field in `nodes.jsonl` (or with the fallback strategy).

