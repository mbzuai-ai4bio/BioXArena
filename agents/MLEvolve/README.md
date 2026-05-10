<p align="center">
  <img src="assets/logo.svg" alt="MLEvolve" width="400"/>
</p>

🌐 **Project Page**: https://internscience.github.io/MLEvolve/

An agentic MLE (Machine Learning Engineering) system that automatically solves Kaggle-style ML competitions through Monte Carlo Graph Search (MCGS) with multi-agent collaboration. This is an advanced version based on [AutoMLGen](https://arxiv.org/abs/2510.08511). MLEvolve achieves **#1 on the [MLE-bench](https://github.com/openai/mle-bench) leaderboard** with **only 12 hours** of runtime.

## Timeline

- **2026-03-23** — Now supports OpenAI-compatible APIs (GPT, Qwen, DeepSeek, etc.). Models with function calling support are recommended for best performance.
- **2026-02-14** — MLEvolve codebase is now open-source.
- **2026-02-14** — MLEvolve achieves **#1 on MLE-bench** (12-hour budget).


## MLE-bench Results

Performance on the [MLE-bench](https://github.com/openai/mle-bench) leaderboard (Any Medal %, mean ± SEM):

| Rank | Agent | LLM | Low (%) | Medium (%) | High (%) | All (%) | Time (h) |
|------|-------|-----|---------|------------|----------|---------|----------|
| 1 | **MLEvolve (Ours)** | Gemini-3-Pro-Preview | **80.30 ± 1.52** | 57.89 ± 1.52 | **42.22 ± 2.22** | **61.33 ± 1.33** | 12 |
| 2 | PiEvolve | Gemini-3-Pro-Preview | **80.30 ± 1.52** | **58.77 ± 0.88** | 40.00 ± 0.00 | **61.33 ± 0.77** | 24 |
| 3 | Famou-Agent 2.0 | Gemini-2.5-Pro | 75.76 ± 1.52 | 57.89 ± 1.52 | 40.00 ± 0.00 | 59.56 ± 0.89 | 24 |
| 4 | ML-Master 2.0 | Deepseek-V3.2-Speciale | 75.76 ± 1.51 | 50.88 ± 3.51 | **42.22 ± 2.22** | 56.44 ± 2.47 | 24 |
| 5 | PiEvolve | Gemini-3-Pro-Preview | 74.24 ± 3.03 | 45.61 ± 0.88 | 35.55 ± 2.22 | 52.00 ± 0.77 | 12 |


## Coding Module in AI-Scientist

MLEvolve powers the **coding and algorithm optimization** module within the [InternAgent](https://github.com/InternScience/InternAgent) system. Built on MLEvolve's refinement engine, [InternAgent 1.5](https://arxiv.org/abs/2602.08990) **further enables autonomous algorithm design and end-to-end scientific discovery**.


## Key Technical Contributions

**Multi-Mode Planning & Code Generation** — Supports base (single-shot) and memory-enhanced (two-stage retrieval-augmented) planning, paired with three code generation strategies: single-pass, stepwise multi-agent pipeline, and incremental SEARCH/REPLACE diff patching. Different modes are dispatched adaptively based on search state.

**Experience-Driven Memory** — A global memory layer records plan, code, metrics, and success/failure labels for every node. Retrieval combines BM25 + FAISS allowing the planner to reinforce proven strategies and avoid known pitfalls from its own search history. Different agents query memory in different ways to encourage novel approaches.

**Progressive MCGS with Cross-Branch Fusion** — The search graph extends vanilla UCT with piecewise exploration decay, time-aware explore-exploit switching, and automatic stagnation detection. Multiple solution branches evolve in parallel; when progress stalls, the system performs cross-branch fusion — merging insights from top-performing nodes across different branches into new solution candidates — and trajectory-aware evolution that leverages each branch's full improvement history to propose informed next steps.



## Setup

**1. Prepare mle-bench** — Install [mle-bench](https://github.com/openai/mle-bench) and download the dataset following its instructions.

**2. Install MLEvolve dependencies**

```bash
pip install --no-deps -r requirements_base.txt
pip install --no-deps -r requirements_ml.txt
pip install --no-deps -r requirements_domain.txt  
```

**3. Configure** — Edit `config/config.yaml`, fields you **must** fill in:

```yaml
dataset_dir: "/path/to/mle-bench/data"

agent:
  code:
    base_url: "https://your-gemini-endpoint"
    api_key: "your-api-key"
  feedback:
    base_url: "https://your-gemini-endpoint"
    api_key: "your-api-key"
```

Other tunable fields (`agent.steps`, `agent.time_limit`, etc.) have sensible defaults — see comments in the yaml file.

### Cold-Start Models (optional)

Cold-start recommends pretrained models per task category based on `engine/coldstart/models_guidance_classified.json`. Most models auto-download from HuggingFace; for models requiring local weights, set `torch_hub_dir` in `config.yaml`. To disable cold-start entirely, set `coldstart.use_coldstart: False`.

## Quick Start

```bash
bash run_single_task.sh <EXP_ID> <DATASET_DIR> [SERVER_ID]

# Example
bash run_single_task.sh denoising-dirty-documents /mle-bench/data 1
```

Results are written to `./runs/<timestamp>_<exp_id>/` including search tree logs, best solution code, and top-K candidate submissions.

## Acknowledgments

We thank [AIDE](https://github.com/WecoAI/aideml) and [ML-Master](https://github.com/sjtu-sai-agents/ML-Master) for their contributions to the development of the MCTS in MLE, and [InternAgent 1.5](https://github.com/InternScience/InternAgent) for its contributions to the development of the agentic memory mechanism. We sincerely thank all teams for their open-source contributions to the community.

## Citation

If you find this repo useful, you can also cite our earlier work.

```bibtex
@article{du2025automlgen,
  title={AutoMLGen: Navigating Fine-Grained Optimization for Coding Agents},
  author={Du, Shangheng and Yan, Xiangchao and Jiang, Dengyang and Yuan, Jiakang and Hu, Yusong and Li, Xin and He, Liang and Zhang, Bo and Bai, Lei},
  journal={arXiv preprint arXiv:2510.08511},
  year={2025}
}

@article{feng2026internagent,
  title={InternAgent-1.5: A Unified Agentic Framework for Long-Horizon Autonomous Scientific Discovery},
  author={Shiyang Feng and Runmin Ma and Xiangchao Yan and Yue Fan and Yusong Hu and Songtao Huang and Shuaiyu Zhang and Zongsheng Cao and Tianshuo Peng and Jiakang Yuan and Zijie Guo and Zhijie Zhong and Shangheng Du and Weida Wang and Jinxin Shi and Yuhao Zhou and Xiaohan He and Zhiyin Yu and Fangchen Yu and Bihao Zhan and Qihao Zheng and Jiamin Wu and Mianxin Liu and Chi Zhang and Shaowei Hou and Shuya Li and Yankai Jiang and Wenjie Lou and Lilong Wang and Zifu Wang and Jiong Wang and Wanghan Xu and Yue Deng and Dongrui Liu and Yiheng Wang and Wenlong Zhang and Fenghua Ling and Shufei Zhang and Xiaosong Wang and Shuangjia Zheng and Xun Huang and Siqi Sun and Shuyue Hu and Peng Ye and Chunfeng Song and Bin Wang and Conghui He and Yihao Liu and Xin Li and Qibin Hou and Tao Chen and Xiangyu Yue and Bin Wang and Liang He and Dahua Lin and Bowen Zhou and Bo Zhang and Lei Bai},
  journal={arXiv preprint arXiv:2602.08990},
  year={2026}
}
```
