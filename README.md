# BioXArena

<p align="center">
  <img src="figs/Figure1_benchmark_pipeline.pdf" alt="BioXArena Overview" width="80%">
</p>

This repository contains the evaluation code for **BioXArena**, our benchmark for assessing State-Of-The-Art (SOTA) LLM agents on biomedical tasks. We have used this codebase to evaluate more than a dozen SOTA LLM agents on the **BioXArena** benchmark.

## 💬 Join Our Community

<div align="center">
  <img src="figs/wechat_group.png" alt="WeChat Group" width="200"/>
  <p><strong>Join our WeChat group for discussions and updates!</strong></p>
</div>

## 📰 News

- **[2026.05.10]** We open-sourced the **BioXArena** codebase and released the [project page](https://leagein.github.io/BioXArena-ProjectPage/).


---

## 1. Download the BioXArena data

Before running any evaluation, download **BioXArena-Data-Public** from Hugging Face:

https://huggingface.co/datasets/Leagein/BioXArena-Data-Public

Use the following commands:

```bash
wget "https://huggingface.co/datasets/Leagein/BioXArena-Data-Public/resolve/main/BioXArena-Data-Public.tar.gz" -O BioXArena-Data-Public.tar.gz
tar -xzf BioXArena-Data-Public.tar.gz
```

---

## 2. Install the environment

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

## 3. Run evaluation

First, clone the repository and move into it:

```bash
git clone git@github.com:Leagein/BioXArena.git
cd BioXArena
```

We evaluate LLM agents through the **OpenRouter** LLM API. Add your API key to `.env` before running:

```bash
api_key=xxx
```

Then move into the training directory:

```bash
cd training
```

The `training/` directory contains shell entrypoints for evaluating different LLM agents.

For each agent, the benchmark is split into **four parts**, and **each part contains 19 tasks**. The four parts can be run in parallel, typically with **one GPU per part**.

Before launching any script, make sure the correct environment is activated:

```bash
conda activate bioxbench
```

### Example: General LLM agents

For general LLM agents, run the following four evaluation entry shell scripts:

```bash
Slurm_run_general_llm_part1.sh
Slurm_run_general_llm_part2.sh
Slurm_run_general_llm_part3.sh
Slurm_run_general_llm_part4.sh
```

These shell scripts internally call the corresponding Python runner. For example:

```text
Slurm_run_general_llm_part*.sh -> run_general_llm_agents.py
```

Before running the scripts, update the following fields in each shell script:

- `PREFIX_DIR="xxx"` on line 7
- `MODEL="z-ai/glm-5.1"` on line 8

`PREFIX_DIR` should be the path that contains the extracted `BioXArena-Data-Public/` directory, but it should **not** include the `BioXArena-Data-Public` folder name itself.

For example, if your data is located at:

```text
/path/to/data/BioXArena-Data-Public
```

then set:

```bash
PREFIX_DIR="/path/to/data"
```

`MODEL` can be set to any LLM available on OpenRouter. In the example scripts, we use:

```bash
MODEL="z-ai/glm-5.1"
```

### Other agents

Evaluation for other LLM agents follows the same pattern. Their evaluation entry shell scripts and corresponding Python runner files are also located under `training/`.

You can also add more agents, including your own LLM agents, by following the same script structure.

### Special notes

For the **Biomni** agent, you must download the Biomni data lake in advance and place `biomni_data/` under `PREFIX_DIR`.

For **MLEvolve** or **mlmaster2.0**, you may either use the shared `bioxbench` environment or create their dedicated environments as follows.

#### MLEvolve

```bash
conda create -n MLEvolve python=3.12
conda activate MLEvolve
cd BioXArena/agents/MLEvolve
pip install --no-deps -r requirements_base.txt
pip install --no-deps -r requirements_ml.txt
pip install --no-deps -r requirements_domain.txt
```

#### mlmaster2.0

```bash
conda create -n mlmaster2.0 python=3.12
conda activate mlmaster2.0
cd BioXArena/agents/EvoMaster
pip install -r playground/ml_master_2/requirements.txt
```

---

## 4. Evaluation outputs

After evaluation finishes, each agent produces outputs following a unified directory structure:

```text
BioXArena-Output/<AgentName>/<round>/<domain>/<task>/
```

All outputs are stored under `BioXArena-Output/`, which is located inside `PREFIX_DIR/`.

### Required files

For **each task**, the corresponding task **root** directory must contain at least the following files:

- `submission.csv` (**required**) — the primary file used for evaluation
- `solution.py` — the generated solution code
- `metrics.json` — metadata and runtime statistics

Among these files, `submission.csv` is **mandatory** and must be correctly formatted for downstream evaluation.

### Search-based coding agents

For agents that employ **search-based strategies** such as Monte Carlo Tree Search (MCTS), including:

- **MLEvolve**
- **MLMaster2.0**

the execution process may generate intermediate results within a nested workspace directory:

```text
BioXArena-Output/<AgentName>/<round>/<domain>/<task>/<workspace>/
```

During execution, multiple candidate solutions may be explored. After completion, the best-performing artifacts are selected and promoted to the task **root** directory:

```text
BioXArena-Output/<AgentName>/<round>/<domain>/<task>/
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

### Standard agents

For all other agents, outputs are written directly to the task root directory:

```text
BioXArena-Output/<AgentName>/<round>/<domain>/<task>/
```

These agents must also produce the same required files:

```text
submission.csv
solution.py
metrics.json
```

### Extending to new agents

To integrate a new agent into the BioXArena evaluation pipeline, ensure that:

1. Outputs follow the standardized directory structure:

   ```text
   BioXArena-Output/<AgentName>/<round>/<domain>/<task>/
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

## 5. Scoring and leaderboard

At the moment, we have **not yet publicly released** the ground-truth `answers.csv` file for the 76 benchmark tasks.

If you run evaluations for your agents, you can package the contents under `BioXArena-Output/` and send them to us. We can compute the final scores using your outputs together with the private answers, send the results back to you, and include your submission on our leaderboard.