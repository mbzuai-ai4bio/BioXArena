# BioXArena: Benchmarking LLM Agents on Multi-Modal Biomedical Machine Learning Tasks (NeurIPS 2026)

<p align="center">
  <a href="https://arxiv.org/abs/2605.15766"><img src="https://img.shields.io/badge/arXiv-2605.15766-b31b1b.svg" alt="arXiv"></a>
  <a href="https://mbzuai-ai4bio.github.io/BioXArena-ProjectPage/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/mbzuai-ai4bio/BioXArena-Data-Public"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow.svg" alt="Hugging Face Dataset"></a>
  <img src="https://img.shields.io/badge/NeurIPS%202026-E%26D%20Track-purple.svg" alt="NeurIPS 2026 E&D Track">
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code%20License-MIT-green.svg" alt="Code License: MIT"></a>
  <a href="https://creativecommons.org/licenses/by-nc/4.0/"><img src="https://img.shields.io/badge/Data%20License-CC%20BY--NC%204.0-lightgrey.svg" alt="Data License: CC BY-NC 4.0"></a>
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

## 📄 License

- **Code** in this repository is released under the [MIT License](LICENSE). The third-party agent frameworks in `agents/` (Biomni, STELLA, MLEvolve, EvoMaster) keep their own licenses (Apache License 2.0).
- **Data** ([BioXArena-Data-Public](https://huggingface.co/datasets/mbzuai-ai4bio/BioXArena-Data-Public)) is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), for academic and other non-commercial use. Each task is derived from public sources (cited in its `description.md`), and use of a task's data is also subject to the license and terms of use of its original source.

## 💬 Join Our Community

<div align="center">
  <img src="figs/wechat_group.jpg" alt="WeChat Group" width="200"/>
  <p><strong>Join our WeChat group for discussions and updates!</strong></p>
</div>


## 📧 Contact

For any questions, suggestions, submission evaluations, or collaboration opportunities, please feel free to reach:

| **Duzhen Zhang** |
|:---:|
| 📧 [duzhen.zhang@mbzuai.ac.ae](mailto:duzhen.zhang@mbzuai.ac.ae) |

## 🚀 How to Use

1. [Download the data](#1-download-the-data)
2. [Install the environment](#2-install-the-environment)
3. [Run an agent](#3-run-an-agent)
4. [Submit to CodaBench for scoring](#4-submit-to-codabench-for-scoring)
5. [Output format and adding your own agent](#5-output-format-and-adding-your-own-agent)

All agents in BioXArena are evaluated through the **[OpenRouter](https://openrouter.ai) API**. Test labels are private: scoring is done on **[CodaBench](https://www.codabench.org/competitions/18373/)**.

### 1. Download the data

Download **BioXArena-Data-Public** from [Hugging Face](https://huggingface.co/datasets/mbzuai-ai4bio/BioXArena-Data-Public):

```bash
wget "https://huggingface.co/datasets/mbzuai-ai4bio/BioXArena-Data-Public/resolve/main/BioXArena-Data-Public.tar.gz" -O BioXArena-Data-Public.tar.gz
tar -xzf BioXArena-Data-Public.tar.gz
```

Each task folder `BioXArena-Data-Public/<domain>/<task>/public/` contains the task description, the training/test inputs and a sample submission.

### 2. Install the environment

```bash
git clone git@github.com:mbzuai-ai4bio/BioXArena.git
cd BioXArena
conda env create -n bioxbench -f environment.yaml
conda activate bioxbench
export OPENROUTER_API_KEY=sk-or-...    # or: cp .env.example .env and fill in OPENROUTER_API_KEY
```

<details>
<summary><b>Agent-specific setup</b> (Claude Code / Codex / OpenCode CLIs, Biomni, MLEvolve, MLMaster2)</summary>

<br>

**Claude Code, Codex and OpenCode** need their CLIs on `PATH` (we used Claude Code 2.1, Codex CLI 0.153 and OpenCode 1.18):

```bash
curl -fsSL https://claude.ai/install.sh | bash      # claude
npm install -g @openai/codex opencode-ai            # codex, opencode (Node.js >= 20)
```

**Biomni**: download the Biomni data lake in advance and place `biomni_data/` next to `BioXArena-Data-Public/` (or pass `--biomni-data-dir`). You may use the shared `bioxbench` environment or Biomni's own environment:

```bash
cd agents/Biomni/biomni_env
conda env create -f environment.yml
```

**MLEvolve** (optional dedicated environment):

```bash
conda create -n MLEvolve python=3.12
conda activate MLEvolve
cd agents/MLEvolve
pip install --no-deps -r requirements_base.txt
pip install --no-deps -r requirements_ml.txt
pip install --no-deps -r requirements_domain.txt
```

**MLMaster2.0** (optional dedicated environment):

```bash
conda create -n mlmaster2.0 python=3.12
conda activate mlmaster2.0
cd agents/EvoMaster
pip install -r playground/ml_master_2/requirements.txt
```

</details>

### 3. Run an agent

[`training/run_agent.py`](training/run_agent.py) runs any agent with any backbone. Its inputs are the **agent name**, the **backbone model** (an OpenRouter model id) and the **API key**:

```bash
cd training
python run_agent.py --agent claudecode --model anthropic/claude-opus-4.6 --api-key "$OPENROUTER_API_KEY" \
    --data-root /path/to/BioXArena-Data-Public --part 1
```

Select tasks with `--task <domain>/<task>` (repeatable), `--domain <domain>`, `--part {1,2,3,4}` (19 tasks each) or `--all-tasks`. Each task gets a 2-hour wall clock and 1 GPU by default, as in the paper. Results are written to `BioXArena-Output/<agent>__<model>/<round>/<domain>/<task>/`.

<details>
<summary><b>Available agents</b></summary>

<br>

| `--agent` | Framework | Runner |
|:---|:---|:---|
| `minimal` | Minimal runner (generate → execute → repair, 3 attempts) | `run_general_llm_agents.py` |
| `opencode` | [OpenCode](https://opencode.ai) CLI | `run_opencode_agent.py` |
| `codex` | [Codex CLI](https://github.com/openai/codex) | `run_codexcli_agent.py` |
| `claudecode` | [Claude Code](https://docs.claude.com/en/docs/claude-code) CLI | `run_claudecode_agent.py` |
| `biomni` | [Biomni](https://github.com/snap-stanford/Biomni) A1 | `run_biomni_agent.py` |
| `stella` | [STELLA](https://github.com/zaixizhang/STELLA) | `run_stella_agent.py` |
| `mlevolve` | [MLEvolve](https://github.com/InternScience/MLEvolve) | `run_mlevolve_agent.py` |
| `mlmaster2` | MLMaster 2.0 (via [EvoMaster](https://github.com/sjtu-sai-agents/EvoMaster)) | `run_mlmaster2.0_agent.py` |

Any model available on OpenRouter can be used as `--model`.

</details>

<details>
<summary><b>Reproducing the 18 paper configurations</b></summary>

<br>

`--config` fills in the agent, backbone and paper-specific settings (e.g. the $30 per-task budget cap for Claude Code with Opus 5 / 5.5):

```bash
python run_agent.py --config codex_g6a --data-root /path/to/BioXArena-Data-Public --part 1
```

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

</details>

<details>
<summary><b>Running on a Slurm cluster</b></summary>

<br>

[`training/Slurm_run_agent.sh`](training/Slurm_run_agent.sh) runs one part (19 tasks) on 1 GPU; the four parts can run in parallel. Set `PREFIX_DIR` (the directory containing `BioXArena-Data-Public/`) in the script, then:

```bash
for part in 1 2 3 4; do sbatch Slurm_run_agent.sh --config claudecode_o55 $part; done
for part in 1 2 3 4; do sbatch Slurm_run_agent.sh opencode qwen/qwen3.6-plus $part; done   # any agent/backbone
```

</details>

<details>
<summary><b>More options</b></summary>

<br>

```bash
python run_agent.py --list-configs                                  # agents + the 18 paper configurations
python run_agent.py --agent minimal --model z-ai/glm-5.1 --check    # one short request: is the backbone reachable?
python run_agent.py --agent codex --model openai/gpt-5.6-sol --part 1 --dry-run   # print the runner command only
```

Other flags: `--output-root` (default: `BioXArena-Output/` next to the repository), `--round-name` (default `round1`), `--time-limit` (seconds per task, default 7200), `--max-workers` / `--num-gpus` (tasks run in parallel). Arguments that `run_agent.py` does not know are passed on to the agent runner.

The API key reaches the runners only through the `OPENROUTER_API_KEY` environment variable; it is not written to command lines, configs or output files. CLI state (Claude Code / Codex / OpenCode configs and sessions) is kept in `BioXArena/.runtime/`, outside the results tree.

</details>

### 4. Submit to CodaBench for scoring

The private test labels are not released. All scoring is done on our **[CodaBench competition](https://www.codabench.org/competitions/18373/)**, with the same graders as in the paper, and results appear on the [CodaBench leaderboard](https://www.codabench.org/competitions/18373/#/results-tab).

Zip the output of **one model and one round** and upload it on the CodaBench *My Submissions* page:

```bash
cd BioXArena-Output
zip -r my_submission.zip claudecode__anthropic__claude-opus-4.6/round1
```

Only `submission.csv` (plus the `predictions/` folder for the two file-based tasks) is needed for each task. Missing tasks and tasks with `FAILED.json` score 0. The exact layout, the two file-based tasks (`imaging/amos-organ-segmentation`, `structure/protein-structure-prediction`) and the score normalization are described on the competition's *Submission format* and *Evaluation* pages.

### 5. Output format and adding your own agent

<details>
<summary><b>Output files of each task</b></summary>

<br>

Each task folder `BioXArena-Output/<agent>__<model>/<round>/<domain>/<task>/` contains:

- `submission.csv` (**required**): predictions for the test set; the first column is `id`, in the same order as the test input.
- `solution.py`: the final training and inference code.
- `metrics.json`: runtime, token usage, model used and other run statistics.
- `FAILED.json`: written by the runner when the task failed (e.g. timeout or missing outputs); such tasks score 0.

Search-based agents (**MLEvolve**, **MLMaster2.0**) explore many candidate solutions in a nested `workspaces/` directory. After the run, `training/run_mlevolve_agent.py` and `training/run_mlmaster2.0_agent.py` copy the best `submission.csv` and `solution.py` to the task folder and write `metrics.json`.

</details>

<details>
<summary><b>Adding your own agent</b></summary>

<br>

To evaluate a new agent, follow the existing runners in `training/`:

1. Write outputs to `BioXArena-Output/<agent>__<model>/<round>/<domain>/<task>/`.
2. Produce `submission.csv`, `solution.py` and `metrics.json` for each task.
3. If the agent explores several candidates (e.g. tree search), copy the best `submission.csv` and its `solution.py` to the task folder and write `metrics.json`.
4. Optionally register the runner in `AGENTS` in `training/run_agent.py` to use the unified entry point.

Then zip the output and submit it to CodaBench as in [step 4](#4-submit-to-codabench-for-scoring).

</details>

## 📝 Citation

If you find BioXArena useful in your research, please cite our NeurIPS 2026 paper:

```bibtex
@inproceedings{li2026bioxarena,
  title={{BioXArena: Benchmarking LLM Agents on Multi-Modal Biomedical Machine Learning Tasks}},
  author={Li, Loka and Zhang, Duzhen and Du, Xingbo and Song, Leonard and Wang, Zixiao and Aukenov, Assanali and Thomas, Noel and Sailaukan, Shakhnazar and Yang, Yonghan and Chen, Feilong and Dong, Jiahua and Zhang, Kun and Zhang, Bin and Song, Le},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year={2026},
  url={https://arxiv.org/abs/2605.15766}
}
```
