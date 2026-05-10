# ML-Master 2.0

**ML-Master 2.0** is a pioneering agentic science framework that tackles the challenge of ultra-long-horizon autonomy through cognitive accumulation, facilitated by a Hierarchical Cognitive Caching (HCC) architecture that dynamically distills transient execution traces into stable long-term knowledge, ensuring that tactical execution and strategic planning remain decoupled yet co-evolve throughout complex, long-horizon scientific explorations.

This repository reimplements and open-sources ML-Master 2.0 on the EvoMaster framework. For quick experimentation, we provide a runnable example task `detecting-insults-in-social-commentary` and an example wisdom database distilled from running this task.

## Architecture

The system implements a pipeline of specialized agents:

1. **Prefetch** - Retrieves relevant knowledge from a wisdom database via RAG (Retrieval-Augmented Generation)
2. **Draft** - Generates an initial ML solution based on task description and retrieved knowledge
3. **Debug** - Fixes code errors if the draft/improvement fails (up to 3 retries)
4. **Research** - Proposes structured improvement directions with specific ideas
5. **Improve** - Implements improvement ideas in parallel, each validated independently
6. **Knowledge Promotion** - Summarizes what worked and what didn't in each research round
7. **Wisdom Promotion** - Extracts reusable insights for future tasks (triggered on timeout)
8. **Metric** - Extracts validation scores from terminal output

### Workflow

```
Prefetch -> Draft -> [Research -> Parallel Improve]* (up to 20 rounds) -> Knowledge Promotion
                                                                              |
                                                          (on timeout) -> Wisdom Promotion
```

### Key Features

- **Parallel Execution**: Multiple improvement ideas run concurrently with workspace isolation
- **Grading Server**: Validates submission format before accepting results
- **Timeout Watchdog**: 24-hour hard limit with graceful shutdown and wisdom extraction
- **RAG Knowledge Retrieval**: Embedding-based retrieval of past experiment insights
- **Score Comparison**: Handles NaN/None values correctly in optimization direction

## Project Structure

```
playground/ml_master_2/
├── __init__.py
├── agent/
│   └── session/
│       └── local.py              # Custom local session with symlink support
├── core/
│   ├── playground.py             # Main orchestrator
│   ├── exp/
│   │   ├── draft_exp.py          # Initial solution generation
│   │   ├── improve_exp.py        # Improvement implementation
│   │   ├── research_exp.py       # Research direction planning
│   │   ├── prefetch_exp.py       # Knowledge retrieval via RAG
│   │   ├── knowledge_promotion_exp.py  # Round summary generation
│   │   └── wisdom_promotion_exp.py     # Reusable wisdom extraction
│   └── utils/
│       ├── code.py               # Code extraction and submission handling
│       ├── data_preview.py       # Dataset preview generation
│       ├── grading.py            # Submission validation client
│       ├── grading_server.py     # Embedded Flask grading server
│       └── watch_dog.py          # Timeout enforcement
├── env/
│   └── local.py                  # Custom local environment
├── prompts/                      # Agent prompt templates
├── example_wisdom/               # Example wisdom database
└── data/                         # Competition datasets
```

## Quick Start

### Prerequisites

- Python 3.12
- EvoMaster framework installed (see project root README.md)
- LLM API endpoint (OpenAI, Anthropic, or local deployment like DeepSeek)

### Installation

1. Install the [MLE-Bench](https://github.com/openai/mle-bench) environment following the official guide.

2. Install ML-Master 2 dependencies (from project root):

```bash
cd playground/ml_master_2
pip install -r requirements.txt
```

3. Download the full MLE-Bench dataset (optional; for quick runs, we include the `detecting-insults-in-social-commentary` task data in `playground/ml_master_2/data`):

   The full MLE-Bench dataset is over **2TB**. We recommend downloading and preparing the dataset using the scripts and instructions provided by **[MLE-Bench](https://github.com/openai/mle-bench)**.

   Once prepared, the expected dataset structure looks like this:

```
/path/to/mle-bench/plant-pathology-2020-fgvc7/
└── prepared
    ├── private
    │   └── test.csv
    └── public
        ├── description.md
        ├── images/
        ├── sample_submission.csv
        ├── test.csv
        └── train.csv
```

   > ML-Master 2 on EvoMaster uses symbolic links to access the dataset. You can download the data to your preferred location and ML-Master will link it accordingly.

### Configuration

1. Create a `.env` file at the project root and set LLM environment variables:

```bash
cp .env.template .env
# Configure the main model API used for coding and iteration (matching your config in configs/ml_master_2/deepseek-v3.2-example.yaml)
DEEPSEEK_API_KEY=""
DEEPSEEK_API_BASE=""
# Configure OpenAI embedding API
OPENAI_API_KEY=""
GPT_CHAT_MODEL=""
GPT_BASE_URL=""
```

2. Complete the configuration in `configs/ml_master_2/deepseek-v3.2-example.yaml`:

```yaml
# competition_id for the task
competition_id: "detecting-insults-in-social-commentary"
# Evaluation metric direction
is_lower_better: false
# Path to prepared data for grading server (see playground/ml_master_2/data for format)
data_root: "./playground/ml_master_2/data"
# See the example config for more options
```

### Running

From the project root, run ML-Master 2 with your configured YAML file and the task description file (e.g., `description.md`):

```bash
python run.py --agent ml_master_2 --config configs/ml_master_2/deepseek-v3.2-example.yaml --task playground/ml_master_2/data/detecting-insults-in-social-commentary/prepared/public/description.md
```

### Viewing Results
All run logs and files are saved in the `runs` directory.

## ✍️ Citation

If you find our work helpful, please use the following citations.

```bibtex
@misc{zhu2026ultralonghorizonagenticsciencecognitive,
      title={Toward Ultra-Long-Horizon Agentic Science: Cognitive Accumulation for Machine Learning Engineering}, 
      author={Xinyu Zhu and Yuzhu Cai and Zexi Liu and Bingyang Zheng and Cheng Wang and Rui Ye and Jiaao Chen and Hanrui Wang and Wei-Chen Wang and Yuzhi Zhang and Linfeng Zhang and Weinan E and Di Jin and Siheng Chen},
      year={2026},
      eprint={2601.10402},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.10402}, 
}
```

```bibtex
@misc{liu2025mlmasteraiforaiintegrationexploration,
      title={ML-Master: Towards AI-for-AI via Integration of Exploration and Reasoning}, 
      author={Zexi Liu and Yuzhu Cai and Xinyu Zhu and Yujie Zheng and Runkun Chen and Ying Wen and Yanfeng Wang and Weinan E and Siheng Chen},
      year={2025},
      eprint={2506.16499},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.16499}, 
}
```