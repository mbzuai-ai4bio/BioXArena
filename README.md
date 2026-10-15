# BioXArena: Benchmarking LLM Agents on Multi-Modal Biomedical Machine Learning Tasks

<p align="center">
  <a href="https://arxiv.org/abs/2605.15766"><img src="https://img.shields.io/badge/arXiv-2605.15766-b31b1b.svg" alt="arXiv"></a>
  <a href="https://mbzuai-ai4bio.github.io/BioXArena-ProjectPage/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/mbzuai-ai4bio/BioXArena-Data-Public"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow.svg" alt="Hugging Face Dataset"></a>
  <img src="https://img.shields.io/badge/NeurIPS%202026-E%26D%20Track-purple.svg" alt="NeurIPS 2026 E&D Track">
</p>

<p align="center">
  <img src="figs/Figure1_benchmark_pipeline.png" alt="BioXArena Overview" width="95%">
</p>

This repository contains the evaluation code for **BioXArena**, a biomedical machine learning (BioML) coding benchmark that tests whether LLM agents can build task-specific models on heterogeneous, often multi-modal biomedical data. BioXArena contains **76 end-to-end tasks across 9 domains** (sequence, single-cell, structure, network biology, chemical biology, perturbation dynamics, phenotype–disease, imaging, and text-integrated). Each task ships public training data and a task description, while the test labels and graders stay private. Agents must write runnable code, train models, and submit predictions for the held-out test samples. We use this codebase to evaluate **18 agent configurations** under the same controlled resource configuration.

## 📰 News

- **[2026.10.15]** 🔥 We updated our [project page](https://mbzuai-ai4bio.github.io/BioXArena-ProjectPage/) and this repository. The **camera-ready paper** is available!
- **[2026.10.14]** [CodaBench](https://www.codabench.org/competitions/18373/) is available for result evaluation.
- **[2026.10.07]** We updated most of the benchmark data on [Hugging Face](https://huggingface.co/datasets/mbzuai-ai4bio/BioXArena-Data-Public). We fixed data leakage, improved the task descriptions, and made further data corrections. All 18 configurations were re-evaluated on the corrected tasks.
- **[2026.09.24]** 🎉 BioXArena was accepted to the **NeurIPS 2026 Evaluations & Datasets Track**!
- **[2026.05.15]** The first version of BioXArena went online, including the [arXiv paper](https://arxiv.org/abs/2605.15766) and the [Hugging Face dataset](https://huggingface.co/datasets/mbzuai-ai4bio/BioXArena-Data-Public).
- **[2026.05.10]** We open-sourced the **BioXArena** codebase and released the [project page](https://mbzuai-ai4bio.github.io/BioXArena-ProjectPage/).

## 🏆 Leaderboard (18 agent configurations)

<p align="center">
  <img src="figs/Figure3_main_leaderboard_failures.png" alt="BioXArena main results" width="90%">
</p>

Results of the 18 agent configurations on the 76 tasks (Figure 3 in the paper). **(a)** Successful and failed runs per configuration, with failures split into no submission, timeout, ID mismatch, NaN/Inf predictions, constant predictions and format errors. **(b)** Normalized scores (0–100, higher is better) per domain; failed runs score 0. **Task Avg** averages over all 76 tasks and **Domain Avg** gives each of the 9 domains equal weight. Rows are grouped by agent framework: Minimal (our minimal code-execution runner, which generates, executes and repairs a single solution script), OpenCode, Codex and Claude Code (generic coding agents), Biomni and STELLA (biomedical agents), and MLEvolve and MLMaster2 (ML coding agents).

The subscript of each row gives the backbone LLM:

| Subscript | Backbone LLM | | Subscript | Backbone LLM |
|:---|:---|---|:---|:---|
| g56s | GPT-5.6-sol | | ge | Gemini-3.1-Pro |
| g6a | GPT-6 Astra | | glm | GLM-5.1 |
| o46 | Claude Opus 4.6 | | qw | Qwen3.6-Plus |
| o5 | Claude Opus 5 | | gm | Gemma-4-31B |
| o55 | Claude Opus 5.5 | | dv3 | DeepSeek-V3.2 |
| s5 | Claude Sonnet 5 | | dv4 | DeepSeek-V4-Pro |

## 💬 Join Our Community

<div align="center">
  <img src="figs/wechat_group.jpg" alt="WeChat Group" width="200"/>
  <p><strong>Join our WeChat group for discussions and updates!</strong></p>
</div>


## 📧 Contact

For any questions, suggestions, submission evaluations, or collaboration opportunities, please feel free to reach:

| **Loka Li** | **Duzhen Zhang** | **Xingbo Du** |
|:---:|:---:|:---:|
| 📧 [longkang.li@mbzuai.ac.ae](mailto:loka.li@mbzuai.ac.ae) | 📧 [duzhen.zhang@mbzuai.ac.ae](mailto:duzhen.zhang@mbzuai.ac.ae) | 📧 [xingbo.du@mbzuai.ac.ae](mailto:xingbo.du@mbzuai.ac.ae) |

## How to Use

### 1. Download the BioXArena data

Before running any evaluation, download **BioXArena-Data-Public** from Hugging Face:

https://huggingface.co/datasets/mbzuai-ai4bio/BioXArena-Data-Public

Use the following commands:

```bash
wget "https://huggingface.co/datasets/mbzuai-ai4bio/BioXArena-Data-Public/resolve/main/BioXArena-Data-Public.tar.gz" -O BioXArena-Data-Public.tar.gz
tar -xzf BioXArena-Data-Public.tar.gz
```

---

### 2. Install the environment

The recommended environment is defined in [environment.yaml](environment.yaml).

Create and activate the environment with:

```bash
conda env create -n bioxbench -f environment.yaml
conda activate bioxbench
```

If you prefer to use the environment name declared inside `environment.yaml`, you can also run:

```bash
conda env create -f environment.yaml
conda activate bioxbench
```

---

### 3. Run evaluation

All agents in BioXArena are evaluated through the paid **[OpenRouter](https://openrouter.ai) API**: every backbone LLM in the paper is called with its OpenRouter model id (e.g. `anthropic/claude-opus-4.6`, `openai/gpt-5.6-sol`), and every agent framework, including Claude Code, Codex CLI and OpenCode, is pointed at OpenRouter.

Clone the repository and provide your OpenRouter key, either as an environment variable or in `.env`:

```bash
git clone git@github.com:mbzuai-ai4bio/BioXArena.git
cd BioXArena
export OPENROUTER_API_KEY=sk-or-...          # or: cp .env.example .env and fill in OPENROUTER_API_KEY
conda activate bioxbench
cd training
```

#### Unified entry point

[`training/run_agent.py`](training/run_agent.py) runs any agent with any backbone. Its inputs are the **agent name**, the **backbone model** (an OpenRouter model id) and the **API key**:

```bash
python run_agent.py --agent claudecode --model anthropic/claude-opus-4.6 --api-key "$OPENROUTER_API_KEY" \
    --data-root /path/to/BioXArena-Data-Public --task chemical-biology/bace1-binding-affinity
```

| Agent (`--agent`) | Framework | Runner |
|:---|:---|:---|
| `minimal` | Minimal runner (generate → execute → repair, 3 attempts) | `run_general_llm_agents.py` |
| `opencode` | [OpenCode](https://opencode.ai) CLI | `run_opencode_agent.py` |
| `codex` | [Codex CLI](https://github.com/openai/codex) | `run_codexcli_agent.py` |
| `claudecode` | [Claude Code](https://docs.claude.com/en/docs/claude-code) CLI | `run_claudecode_agent.py` |
| `biomni` | [Biomni](https://github.com/snap-stanford/Biomni) A1 | `run_biomni_agent.py` |
| `stella` | [STELLA](https://github.com/zaixizhang/STELLA) | `run_stella_agent.py` |
| `mlevolve` | [MLEvolve](https://github.com/InternScience/MLEvolve) | `run_mlevolve_agent.py` |
| `mlmaster2` | MLMaster 2.0 (via [EvoMaster](https://github.com/sjtu-sai-agents/EvoMaster)) | `run_mlmaster2.0_agent.py` |

Task selection: `--task <domain>/<task>` (repeatable), `--domain <domain>`, `--part {1,2,3,4}` (19 tasks each, sorted by task key) or `--all-tasks`. By default each task gets a 2-hour wall clock (`--time-limit 7200`) and 1 GPU, as in the paper. Useful extras:

```bash
python run_agent.py --list-configs                                  # agents + the 18 paper configurations
python run_agent.py --agent minimal --model z-ai/glm-5.1 --check    # one short request: is the backbone reachable?
python run_agent.py --agent codex --model openai/gpt-5.6-sol --part 1 --dry-run   # print the runner command only
```

#### Reproducing the 18 paper configurations

`--config` fills in the agent, backbone and paper-specific settings (e.g. the $30 per-task budget cap for Claude Code with Opus 5 / 5.5):

| `--config` | Agent | OpenRouter model | | `--config` | Agent | OpenRouter model |
|:---|:---|:---|---|:---|:---|:---|
| `minimal_g56s` | Minimal | `openai/gpt-5.6-sol` | | `codex_g56s` | Codex | `openai/gpt-5.6-sol` |
| `minimal_o46` | Minimal | `anthropic/claude-opus-4.6` | | `codex_g6a` | Codex | `openai/gpt-6-astra` |
| `minimal_ge` | Minimal | `google/gemini-3.1-pro-preview` | | `claudecode_o46` | Claude Code | `anthropic/claude-opus-4.6` |
| `minimal_glm` | Minimal | `z-ai/glm-5.1` | | `claudecode_o5` | Claude Code | `anthropic/claude-opus-5` |
| `minimal_qw` | Minimal | `qwen/qwen3.6-plus` | | `claudecode_o55` | Claude Code | `anthropic/claude-opus-5.5` |
| `minimal_gm` | Minimal | `google/gemma-4-31b-it` | | `biomni_s5` | Biomni | `anthropic/claude-sonnet-5` |
| `minimal_dv3` | Minimal | `deepseek/deepseek-v3.2` | | `stella_s5` | STELLA | `anthropic/claude-sonnet-5` |
| `opencode_g56s` | OpenCode | `openai/gpt-5.6-sol` | | `mlevolve_ge` | MLEvolve | `google/gemini-3.1-pro-preview` |
| `opencode_o46` | OpenCode | `anthropic/claude-opus-4.6` | | `mlmaster2_dv4` | MLMaster2 | `deepseek/deepseek-v4-pro` |

```bash
python run_agent.py --config codex_g6a --data-root /path/to/BioXArena-Data-Public --part 1
```

On a Slurm cluster, [`training/Slurm_run_agent.sh`](training/Slurm_run_agent.sh) runs one part (19 tasks) on 1 GPU; the four parts can run in parallel. Set `PREFIX_DIR` (the directory containing `BioXArena-Data-Public/`) in the script, then:

```bash
for part in 1 2 3 4; do sbatch Slurm_run_agent.sh --config claudecode_o55 $part; done
for part in 1 2 3 4; do sbatch Slurm_run_agent.sh opencode qwen/qwen3.6-plus $part; done   # any agent/backbone
```

The API key is passed to the runners through the `OPENROUTER_API_KEY` environment variable only; it is not written to command lines, configs or output files. CLI state (Claude Code / Codex / OpenCode configs and sessions) is kept in `BioXArena/.runtime/`, outside the results tree.

#### Special notes

The **Claude Code**, **Codex** and **OpenCode** agents need the corresponding CLIs on `PATH` (we used Claude Code 2.1, Codex CLI 0.153 and OpenCode 1.18):

```bash
curl -fsSL https://claude.ai/install.sh | bash      # claude
npm install -g @openai/codex opencode-ai            # codex, opencode (Node.js >= 20)
```

For the **Biomni** agent, download the Biomni data lake in advance and place `biomni_data/` under `PREFIX_DIR` (or pass `--biomni-data-dir`).
You may either use the shared `bioxbench` environment or create its dedicated environment (`biomni_e1`) as follows.
```bash
cd BioXArena/agents/Biomni/biomni_env
conda env create -f environment.yml
```

For **MLEvolve** or **mlmaster2.0**, you may either use the shared `bioxbench` environment or create their dedicated environments as follows.

##### MLEvolve

```bash
conda create -n MLEvolve python=3.12
conda activate MLEvolve
cd BioXArena/agents/MLEvolve
pip install --no-deps -r requirements_base.txt
pip install --no-deps -r requirements_ml.txt
pip install --no-deps -r requirements_domain.txt
```

##### mlmaster2.0

```bash
conda create -n mlmaster2.0 python=3.12
conda activate mlmaster2.0
cd BioXArena/agents/EvoMaster
pip install -r playground/ml_master_2/requirements.txt
```

---

### 4. Evaluation outputs

After evaluation finishes, each agent produces outputs following a unified directory structure:

```text
BioXArena-Output/<agent>__<model>/<round>/<domain>/<task>/
```

For example, `BioXArena-Output/claudecode__anthropic__claude-opus-4.6/round1/chemical-biology/bace1-binding-affinity/`. By default `BioXArena-Output/` is created next to the repository (`--output-root` to change it).

#### Required files

For **each task**, the corresponding task **root** directory must contain at least the following files:

- `submission.csv` (**required**) — the primary file used for evaluation
- `solution.py` — the generated solution code
- `metrics.json` — metadata and runtime statistics

Among these files, `submission.csv` is **mandatory** and must be correctly formatted for downstream evaluation.

#### Search-based coding agents

For agents that employ **search-based strategies** such as Monte Carlo Tree Search (MCTS), including:

- **MLEvolve**
- **MLMaster2.0**

the execution process may generate intermediate results within a nested workspace directory:

```text
BioXArena-Output/<agent>__<model>/<round>/<domain>/<task>/<workspace>/
```

During execution, multiple candidate solutions may be explored. After completion, the best-performing artifacts are selected and promoted to the task **root** directory:

```text
BioXArena-Output/<agent>__<model>/<round>/<domain>/<task>/
```

Specifically:

- The best `submission.csv` is copied or moved to the task root directory
- The corresponding `solution.py` is copied or moved to the task root directory
- A `metrics.json` file is generated in the task root directory

This post-processing behavior is implemented in:

- `training/run_mlevolve_agent.py`
- `training/run_mlmaster2.0_agent.py`

As a result, after post-processing, each task root directory should contain at least:

```text
submission.csv
solution.py
metrics.json
```

The `metrics.json` file typically records information such as:

- total runtime
- generated model identifier
- token usage
- other execution-related statistics

#### Standard agents

For all other agents, outputs are written directly to the task root directory:

```text
BioXArena-Output/<agent>__<model>/<round>/<domain>/<task>/
```

These agents must also produce the same required files:

```text
submission.csv
solution.py
metrics.json
```

#### Extending to new agents

To integrate a new agent into the BioXArena evaluation pipeline, ensure the following (you may refer to the existing agent runner scripts under `training/`):

1. Outputs follow the standardized directory structure:

   ```text
   BioXArena-Output/<agent>__<model>/<round>/<domain>/<task>/
   ```

2. Each task produces the required files:

   - `submission.csv`
   - `solution.py`
   - `metrics.json`

3. If the agent involves intermediate exploration, such as multiple candidates or search procedures, implement a post-processing step to:

   - select the best result
   - copy or move the best `submission.csv` to the task root directory
   - copy or move the corresponding `solution.py` to the task root directory
   - generate `metrics.json`

Adhering to this format ensures compatibility with the evaluation and scoring pipeline.

---

### 5. Scoring and leaderboard

At the moment, we have **not yet publicly released** the ground-truth `answers.csv` file for the 76 benchmark tasks.

If you run evaluations for your agents, you can package the contents under `BioXArena-Output/` and send them to us. If the outputs are large, you do **not** need to include all intermediate files—please ensure that for each task, the root directory contains at least the required files (`submission.csv`, `solution.py`, and `metrics.json`), with `submission.csv` being the minimum requirement. You can package only these files and send them to us.

We will compute the final scores using your outputs together with the private answers, return the results to you, and optionally include your submission on our leaderboard.

## 📝 Citation

If you find BioXArena useful in your research, please cite our paper:

```bibtex
@article{li2026bioxarena,
  title={{BioXArena: Benchmarking LLM Agents on Multi-Modal Biomedical Machine Learning Tasks}},
  author={Li, Loka and Zhang, Duzhen and Du, Xingbo and Song, Leonard and Wang, Zixiao and Aukenov, Assanali and Thomas, Noel and Sailaukan, Shakhnazar and Yang, Yonghan and Chen, Feilong and Dong, Jiahua and Zhang, Kun and Zhang, Bin and Song, Le},
  journal={arXiv preprint arXiv:2605.15766},
  year={2026},
  url={https://arxiv.org/abs/2605.15766}
}
```
