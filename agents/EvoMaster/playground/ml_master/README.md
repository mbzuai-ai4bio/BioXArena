# ML-Master 1.0

**ML-Master** is a novel AI4AI (AI-for-AI) agent that integrates exploration and reasoning into a coherent iterative methodology, facilitated by an adaptive memory mechanism that selectively captures and summarizes relevant insights and outcomes, ensuring each component mutually reinforces the other without compromising either.

Here, we reimplemented the code for ML-Master 1.0 based on the EvoMaster framework. To make it easy to run, we provide a quick-start example task: `detecting-insults-in-social-commentary`.

## Architecture

The system implements a pipeline composed of specialized agents:

1. **Draft** - Generates an initial ML solution based on task description
2. **Debug** - Fixes code errors if the draft/improvement fails
3. **Improve** - Implements improvement ideas in parallel, each validated independently
4. **Metric** - Extracts validation scores from terminal output

### Workflow

```text
Task Input -> Draft -> [Debug / Improve -> Execute Validation -> Memory Refinement]* -> Best Solution Output -> Submit to mle-bench (optional)
```

## Key Features

- **Parallel Execution**: Multiple improvement ideas run concurrently with workspace isolation
- **Grading Server**: Validates submission format before accepting results
- **Tree-search decision making**: Uses MCTS/UCT to balance between “digging deeper into the current path” and “trying new paths”
- **Adaptive memory**: Preserves key experience from parent and sibling nodes and compresses it into reusable context
- **Termination and pruning mechanism**: Stops continuous ineffective improvements and overly deep debugging to avoid wasting compute

## Project Structure

```text
playground/ml_master/
├── __init__.py
├── core/
│   ├── playground.py               # Main orchestrator for ml_master
│   ├── exp/                   
│   │   ├── draft_exp.py            # Initial solution generation
│   │   ├── improve_exp.py          # Improvement implementation
│   │   └── debug_exp.py            # Debugging and error fixing
│   └── utils/      
│   │   ├── artifacts.py            # Result snapshots and persistence
│   │   ├── data_preview.py         # Dataset preview generation
│   │   ├── engine.py               # Execution engine wrapper
│   │   ├── grading_server.py       # Embedded Flask grading server
│   │   ├── grading.py              # Submission validation client
│   │   ├── mlebench_grade.py       # Automatic mle-bench submission
│   │   ├── orchestrator.py         # Parallel search and task scheduling
│   │   └── uct.py                  # UCT tree-search logic         
├── prompts/                        # Agent prompt templates
│   ├── draft/
│   ├── debug/
│   ├── improve/
│   └── metric/
├── scripts/                       
│   └── mlmaster_test.py            # Batch run script for ml_master
└── vis/                            # Visualization for ml_master
```

## Quick Start

### Prerequisites

- Python 3.12
- EvoMaster base dependency framework installed (see the root `README.md` of the project for details)
- An LLM API endpoint (OpenAI, Anthropic, or a locally deployed model such as DeepSeek)

### Installation

1. Install the [MLE-Bench](https://github.com/openai/mle-bench) environment following the official guide.

2. Install ML-Master dependencies (from project root):

```bash
cd playground/ml_master
pip install -r requirements.txt
```

3. Download the full MLE-Bench dataset:

   The full MLE-Bench dataset is over **2TB**. We recommend downloading and preparing the dataset using the scripts and instructions provided by **[MLE-Bench](https://github.com/openai/mle-bench)**.

   Once prepared, the expected dataset structure looks like this:

```text
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

   > ML-Master on EvoMaster uses symbolic links to access the dataset. You can download the data to your preferred location and ML-Master will link it accordingly.

### Configuration

1. Create a `.env` file at the project root and set LLM environment variables:

```bash
cp .env.template .env
# Configure the main model API used for coding and iteration (matching your config in configs/ml_master/config.yaml)
DEEPSEEK_API_KEY=""
DEEPSEEK_API_BASE=""
# Configure OpenAI embedding API
OPENAI_API_KEY=""
GPT_CHAT_MODEL=""
GPT_BASE_URL=""
```

2. Complete the configuration in `configs/ml_master/config.yaml`:

```yaml
# Competition exp_id for the task
exp_id: "detecting-insults-in-social-commentary"
# Path to prepared data for grading server
data_root: "/data/exp_data"   
# Symlink configuration
symlinks:"/data/exp_data/detecting-insults-in-social-commentary/prepared/public": "input"
# See the example config for more options
```

### Running a Single Task

From the project root, run ML-Master with your configured YAML file and the task description file (e.g., `description.md`):

```bash
python run.py --agent ml_master --config configs/ml_master/config.yaml --task /data/exp_data/detecting-insults-in-social-commentary/prepared/public/description.md
```

### Running Batch Tasks

1. `mlmaster/scripts/mlmaster_test.py` provides batch execution. Complete the parameters in the script:

```bash
PROJECT_DIR = Path("Path-to-EvoMaster")
BASE_CONFIG = Path("configs/ml_master/config.yaml")
EXP_ROOT = Path("/data/exp_data")
COMPETITIONS = ["detecting-insults-in-social-commentary", "plant-pathology-2020-fgvc7"]
```

2. Then modify the configuration file as follows:

```yaml
exp_id: "X"  
symlinks:"/data/exp_data/X/prepared/public": "input"
```

3. Run the script:

```bash
python playground/ml_master/scripts/mlmaster_test.py
```

4. The script will automatically create competition-specific config files in the `.tmp_configs` directory and create working directories under `runs` based on the competition names.

### Viewing Results

- All run logs and files are saved in the `runs` directory.
- Visualization: From the project root, specify the corresponding working directory (e.g., `runs/ml_master_xxx`) and run the following command, then visit the corresponding port (default: http://127.0.0.1:8765):

```bash
python -m playground.ml_master.vis.app --run_dir runs/ml_master_xxx
```

## Development and Extension

- Adjust or add prompt templates in `prompts/` to change agent generation strategies
- Extend or modify the experiment workflow under `core/exp/` (draft, research, improve, etc.)
- Use the embedded grading server to validate submissions during local development

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

