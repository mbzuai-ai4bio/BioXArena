---
name: rag
description: Retrieval-Augmented Generation (RAG) skill for generic semantic search and knowledge retrieval. Use when you need to build or call vector-based search over your own documents using vector indexes (optional FAISS) and embedding models (local transformers or OpenAI embeddings), or when integrating LLMs with external knowledge bases in a project-agnostic way.
license: Proprietary. LICENSE.txt has complete terms
---

# RAG Skill Overview

This skill provides generic Retrieval-Augmented Generation (RAG) capabilities for **building and calling vector search over custom documents in any project**.  

Typical use cases include:

- Powering knowledge-base retrieval for Q&A systems and chatbots;
- Doing semantic search over technical documentation, codebases, logs, etc.;
- Supplying “retrieved relevant snippets” to downstream LLM generation steps.

> Detailed, script-level explanations are split into standalone documents under the `reference/` directory. This file focuses on the overall structure and usage guidelines.

## Directory Structure

The skill directory is organized as follows:

```text
rag/
├── SKILL.md                 # This file: overview and navigation
├── scripts/                 # Executable scripts (core logic)
│   ├── search.py            # Generic vector search + content retrieval
│   ├── encode.py            # Text encoding utilities (embeddings)
│   ├── build_faiss.py       # Build faiss.index from embeddings.npy (for --use_faiss)
│   └── database.py          # Vector database builder interface (placeholder implementation)
└── reference/               # Detailed docs for each script
    ├── search.md            # Parameters, I/O, and examples for search.py
    ├── encode.md            # Usage and scenarios for encode.py
    ├── build_faiss.md       # Building faiss.index from existing embeddings
    └── database.md          # Interface design and extension notes for database.py
```


## When to Use This Skill

Use the `rag` skill when you need any of the following:

- **Semantic search over your own documents/data**, e.g. “find the most relevant snippets in these JSON/Markdown/code files for a given question”;
- **Building or maintaining a vector database**, and querying it efficiently via vector search (e.g. FAISS when available);
- **Combining external knowledge with an LLM**, i.e. implementing a RAG (retrieval-augmented generation) workflow;
- **Reusing an existing vector store** (with `embeddings.npy` and `nodes.jsonl`; `faiss.index` optional) across different tasks or projects.

If you are only doing “single-turn reasoning without external knowledge” (plain Q&A), you do not need this skill.

## Quick Start (Command-Line Perspective)

The primary way to interact with this skill is via the Python scripts in the `scripts/` directory. Below are conceptual flows for the two most common operations, **with explicit embedding parameters** so agents can correctly honor upstream configuration (e.g., EvoMaster playground configs).

### 1. Semantic Search over an Existing Vector Store (`scripts/search.py`)

Prerequisite: You already have a vector store directory `vec_dir` that contains at least:

- `embeddings.npy`: precomputed embedding matrix (used for cosine similarity);
- `nodes.jsonl`: one JSON per line, each containing fields that identify a node (by default `node_id`).

Whether to load `faiss.index` is **explicitly controlled** by the caller: pass `--use_faiss` (CLI) or `use_faiss=True` (API) to load it when FAISS is installed and the file exists; default is off (search uses `embeddings.npy` only).

The minimal call pattern (only returning `node_id` and cosine similarity) is for the host system to run `search.py` with arguments, providing at least:

- `vec_dir`: path to the vector store directory;
- `query`: the query text;
- Optional `top_k`: number of results to return.

When you have an external configuration that already decides **which embedding backend/model to use** (for example, EvoMaster’s `embedding` block with `type`, `openai.model`, `dimensions`, etc.), you should **forward those values explicitly** to this script, instead of relying on its internal defaults.

#### How to invoke

Invoke via run_script (`script_name: search.py`). **For the full parameter list and semantics, load `reference/search.md` first** (use_skill with `action: get_reference`, `reference_name: search.md`); do not rely on this file alone to construct `script_args`. To retrieve original content per hit, you will need `nodes_data` and `content_path`; see `reference/search.md`.

### 2. Building faiss.index (`scripts/build_faiss.py`)

If you already have a vector store with `embeddings.npy` and want to enable `--use_faiss` in search (e.g. for larger-scale retrieval), run `build_faiss.py` via run_script with `script_args: --vec_dir <path_to_vec_dir>`. This reads `vec_dir/embeddings.npy`, builds a normalized inner-product index (cosine similarity), and writes `vec_dir/faiss.index`. The `faiss` package must be installed (`pip install faiss-cpu`). See `reference/build_faiss.md` for details.

### 3. Standalone Text Embedding Generation (`scripts/encode.py`)

When you need to manually build a vector store, inspect embeddings, or reuse the “text → vector” capability in other systems, the host can call `encode.py` and provide at least:

- A model name or path for encoding (local Transformer model or OpenAI embedding model);
- The text(s) to encode (single or multiple);
- Optional output location (e.g. `.npy` file), whether to normalize embeddings, batch size, etc.

`encode.py` supports:

- Single-text encoding / batch encoding of multiple texts;
- Local Transformer/HuggingFace models and the OpenAI Embedding API;
- Optional vector normalization and batch-size control.  

More examples and parameter details are in `reference/encode.md`.

## Reference Documentation Navigation (`reference/`)

This skill follows the principle: **keep `SKILL.md` concise and push details into `reference/`**. When using the skill, selectively load only the docs you actually need:

- `reference/search.md`  
  - **Scope**: Detailed documentation for `scripts/search.py`.  
  - **Content**: Vector store input requirements, CLI parameters, search and content-retrieval modes, configuration for OpenAI vs. local models, generic schema design, and best practices.

- `reference/encode.md`  
  - **Scope**: Documentation for `scripts/encode.py`.  
  - **Content**: Single/batched encoding, output format (`.npy`), embedding normalization, and integration scenarios with other systems.

- `reference/build_faiss.md`  
  - **Scope**: Documentation for `scripts/build_faiss.py`.  
  - **Content**: How to generate `faiss.index` from `embeddings.npy`, when to use it, and dependency (faiss package).

- `reference/database.md`  
  - **Scope**: Interface design for `scripts/database.py`.  
  - **Content**: The responsibilities of `VectorDatabaseBuilder`, expected semantics of each method, current placeholder status, and how to gradually implement build/incremental-update/statistics logic in your own project.

When using this skill, **load only the reference docs that are directly relevant to the current task**, instead of pulling all of them into context at once.

## Design Principles and Notes

- **Project-Agnostic (Generic) Design**
  - Script interfaces do not depend on specific business fields or task definitions.
  - All business-specific fields (such as `data_knowledge`, `model_knowledge`, etc.) are accessed via configurable parameters like `content_path`.

- **Pluggable Embedding Models**
  - Supports local Transformer models, HuggingFace models, and the OpenAI Embedding API.
  - A unified `create_embedder` abstraction wraps these backends so callers do not need to handle SDK-specific details.

- **Extensible Vector Store Structure**
  - Search requires `embeddings.npy` and `nodes.jsonl`. Loading `faiss.index` is controlled by the caller via `use_faiss` (default off). Other files (e.g. `nodes_data.json`) are optional for content retrieval.
  - You are free to design the JSON structure of `nodes_data` according to your own project.

- **Script-First, Minimal Documentation**
  - Logic that can be implemented in scripts is placed under `scripts/` whenever possible.
  - Documentation focuses on “how to call the scripts correctly” and on parameter/I/O descriptions, instead of re-implementing logic in prose.

If you need project-specific “database schema examples” or “custom business field conventions” on top of this skill, document them in your own project, rather than modifying this skill. This keeps `rag` reusable across multiple, unrelated projects.

