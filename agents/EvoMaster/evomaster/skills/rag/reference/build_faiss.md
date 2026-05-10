## `scripts/build_faiss.py`: Build FAISS index from embeddings

When you have a vector store that already contains `embeddings.npy` and `nodes.jsonl`, you can generate `faiss.index` so that `search.py` can use it when invoked with `--use_faiss`. This script does that in one step.

### What it does

- Reads `vec_dir/embeddings.npy` (required).
- Normalizes each row (L2 norm) so that inner product equals cosine similarity.
- Builds a FAISS `IndexFlatIP` (exact inner-product search) and writes `vec_dir/faiss.index`.
- Uses the same path resolution as `search.py`: `evomaster/`-prefixed paths are resolved relative to the project root.

### When to use

- You already have a vector store (e.g. produced by your own pipeline or `encode.py` + manual `nodes.jsonl`).
- You want to enable `--use_faiss` in `search.py` for that store (typically for larger-scale search).
- The `faiss` package is installed (`pip install faiss-cpu` or `faiss-gpu`).

If you only have a small number of vectors, you can skip building `faiss.index` and run search without `--use_faiss` (default); search will use `embeddings.npy` only.

### Parameters (for run_script / use_skill)

- **`--vec_dir`** (required): Directory containing `embeddings.npy`. The script writes `faiss.index` into the same directory. Paths starting with `evomaster/` are resolved relative to the project root (same as `search.py`).

### Requirements and errors

- **faiss**: The script requires the `faiss` package. If it is not installed, you get a clear `ImportError` and instructions to install `faiss-cpu` (or `faiss-gpu`).
- **embeddings.npy**: Must exist under `vec_dir`. If missing, the script exits with `FileNotFoundError`.
- **Array shape**: `embeddings.npy` must be a 2D array `(n_vectors, embedding_dim)`. Otherwise a `RuntimeError` is raised.

### Programmatic use

From Python you can call `build_faiss_index(vec_dir, project_root=None)`; it returns the path to the written `faiss.index`. Useful when building the vector store in the same process and generating the FAISS index without a separate script invocation.
