## `scripts/encode.py`: General-Purpose Text Encoding Utility

`encode.py` provides a standalone **text → vector** encoding capability that can be used in any scenario where you need to manually generate or inspect embeddings.

### Feature Overview

- Encode one or more text snippets into vectors (numpy arrays).
- Supports:
  - Local Transformer models (HuggingFace or local paths).
  - OpenAI Embedding API (e.g., `text-embedding-3-large`).
- Supports:
  - Single-text encoding.
  - Batch encoding for multiple texts.
  - Optional vector normalization (suitable for inner-product/cosine-similarity search).
- Output options:
  - Save as `.npy` files for use by downstream scripts or vector-store builders.
  - Print dimensions and vector values directly to the console (for debugging).

The core logic lives in the `TextEncoder` class, which uses `create_embedder` under the hood to share embedding logic with `search.py` and keep encoding behavior consistent.

### Usage Pattern (Conceptual)

In real projects, `encode.py` is typically called through the host system’s “skill script runner” rather than manually from the terminal. This section describes the expected parameters and behavior only; it does not provide concrete CLI examples.

#### Encoding a Single Text

Key parameters:

- `--text`: Provide the text directly on the command line.
- `--output`: Output path (e.g., `./embedding.npy`). If omitted, embeddings are printed to stdout.
- `--normalize`: Whether to apply L2 normalization to the vectors.

#### Batch Encoding (From File)

When encoding many texts, you can use `--file` to read one text per line, with the host system passing these arguments to the script.

In that case:

- `texts.txt`: Each line is a text to encode; empty lines are skipped.
- `--batch_size`: Batch size to balance memory usage and throughput.

#### Reading Text from Standard Input

If neither `--text` nor `--file` is provided, the script reads non-empty lines from standard input. Whether this behavior is used depends on how the host system integrates the script.

### Embedding-Related Parameters

`encode.py` creates embedding models via `create_embedder`, unifying local and OpenAI backends:

- `--model`:
  - Local path, e.g. `/models/all-mpnet-base-v2`.
  - HuggingFace model name, e.g. `sentence-transformers/all-mpnet-base-v2`.
  - OpenAI model name, e.g. `text-embedding-3-large`.
- `--embedding_type`:
  - `auto`: Automatically chooses local vs. OpenAI based on `model` and environment variables.
  - `local`: Force local/HuggingFace model usage.
  - `openai`: Force OpenAI Embedding API usage.
- `--embedding_api_key` / `--embedding_base_url`:
  - Explicitly pass OpenAI API key / base URL.
  - May also be provided via environment variables `OPENAI_EMBEDDING_API_KEY` / `OPENAI_API_KEY` and `OPENAI_EMBEDDING_BASE_URL` / `OPENAI_BASE_URL`.
- `--embedding_dimensions`:
  - For the `text-embedding-3-*` family, you can specify the output dimension (e.g., 512, 1024, 3072, etc.).

### Encoding Behavior and Return Format

- `encode(text, ...)`:
  - Returns a one-dimensional numpy array shaped `(embedding_dim,)`.
  - If the underlying model returns a 2D array, the first vector is selected automatically.
- `encode_batch(texts, ...)`:
  - Returns a two-dimensional numpy array shaped `(n_texts, embedding_dim)`.
- When `--normalize` is set, L2 normalization is applied to each vector:
  - This is appropriate for use with inner-product or cosine-similarity–based retrieval.

### Typical Use Cases

- Manually building or extending a vector database:
  - Use `encode.py` to encode a corpus into vectors and save them as `.npy`.
  - Then use your own builder scripts or tools to create a FAISS index.
- Debugging and alignment:
  - Verify that embedding dimensions and numeric ranges under the current configuration look as expected.
  - Compare outputs from different model configurations (local vs. OpenAI) on the same inputs.
- Integration with other systems:
  - When you just need a simple “text → vector” tool to plug into existing search/recommendation/clustering systems.
