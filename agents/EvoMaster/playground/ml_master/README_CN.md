# ML-Master 1.0

**ML-Master** 是一个创新的 AI4AI（AI-for-AI）智能体，将探索与推理融合为一种连贯的迭代方法论，借助自适应记忆机制，选择性地捕获和总结相关洞察与结果，确保各组件相互强化而不互相妥协。

在这里，我们基于EvoMaster框架重新实现了ML-Master 1.0的代码。为了方便运行，我们提供了一个可以快速运行的示例任务`detecting-insults-in-social-commentary`。

## 架构

系统实现了一个由专用智能体组成的流水线：

1. **Draft（起草）** - 根据任务描述生成初始 ML 解决方案
2. **Debug（调试）** - 当代码运行失败时修复错误
3. **Improve（改进）** - 并行实现多个改进想法，每个独立验证
4. **Metric（评估）** - 从终端输出中提取验证分数

### 工作流程

```
任务输入 -> Draft -> [Debug / Improve -> 执行验证 -> 记忆提炼]* -> 最优解输出 -> 提交mle-bench（可选）                             
```

## 核心特性

- **并行执行**：多个改进想法并发运行，工作空间隔离
- **评分服务器**：在接受结果前验证提交格式
- **树搜索决策**：用 MCTS/UCT 在“深挖当前路径”和“尝试新路径”之间做平衡
- **自适应记忆**：保留父节点和兄弟节点中的关键经验，压缩成可复用上下文
- **终止剪枝机制**：对连续无效改进和过深调试设置停止条件，避免浪费算力

## 项目结构

```
playground/ml_master/
├── __init__.py
├── core/
│   ├── playground.py               # ml_master 的主编排器
│   ├── exp/                   
│   │   ├── draft_exp.py            # 初始方案生成
│   │   ├── improve_exp.py          # 改进实现
│   │   └── debug_exp.py            # 调试并修复报错
│   └── utils/      
│   │   ├── artifacts.py            # 结果快照与持久化
│   │   ├── data_preview.py         # 数据集预览生成 
│   │   ├── engine.py               # 执行引擎封装 
│   │   ├── grading_server.py       # 内嵌 Flask 评分服务器
│   │   ├── grading.py              # 提交验证客户端 
│   │   ├── mlebench_grade.py       # mle-bench自动提交
│   │   ├── orchestrator.py         # 并行搜索与任务调度
│   │   └── uct.py                  # UCT树搜索逻辑         
├── prompts/                        # 智能体提示模板
│   ├── draft/
│   ├── debug/
│   ├── improve/
│   └── metric/
├── scripts/                       
│   └── mlmaster_test.py            #ml_master批量运行脚本
└── vis/                            # ml_master 可视化
```

## 快速开始

### 前置条件

- Python 3.12
- 已安装 EvoMaster 基础需求框架（详见项目根目录的 README.md）
- LLM API 端点（OpenAI、Anthropic 或本地部署如 DeepSeek）


### 安装

1. 根据 [MLE-Bench](https://github.com/openai/mle-bench) 官方指引安装其环境

2. 安装 ML-Master 依赖（在项目根目录下执行）：

```bash
cd playground/ml_master
pip install -r requirements.txt
```

3. 下载 MLE-Bench 完整数据集：

   MLE-Bench 完整数据集超过 **2TB**，建议按 **[MLE-Bench](https://github.com/openai/mle-bench)** 官方脚本和说明下载与准备。

   准备完成后，数据集目录结构示例如下：

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

   > ML-Master 在 EvoMaster 中通过符号链接访问数据集，可将数据下载到任意位置，ML-Master 会自动建立链接。

### 配置

1. 在项目根目录创建 `.env` 文件，配置 LLM 环境变量：

```bash
cp .env.template .env
# 根据 configs/ml_master/config.yaml 中使用的模型，配置用于写代码和迭代的主模型 API
DEEPSEEK_API_KEY=""
DEEPSEEK_API_BASE=""
# 配置 OpenAI 的 embedding API
OPENAI_API_KEY=""
GPT_CHAT_MODEL=""
GPT_BASE_URL=""
```

2. 在 `configs/ml_master/config.yaml` 中完成相关配置：

```yaml
# 比赛的 exp_id
exp_id: "detecting-insults-in-social-commentary"
# 准备好的原始数据位置，供 grading server 使用
data_root: "/data/exp_data"   
# 软链接配置
symlinks:"/data/exp_data/detecting-insults-in-social-commentary/prepared/public": "input"
# 更多配置说明请参见示例配置文件
```

### 单个任务运行

在项目根目录下，指定配置的 yaml 文件和任务描述文件（如 `description.md`）运行 ML-Master：

```bash
python run.py --agent ml_master --config configs/ml_master/config.yaml --task /data/exp_data/detecting-insults-in-social-commentary/prepared/public/description.md
```

### 批量任务运行

1. `mlmaster/scripts/mlmaster_test.py`提供批量运行功能，在脚本中完善参数：
```bash
PROJECT_DIR = Path("Path-to-EvoMaster")
BASE_CONFIG = Path("configs/ml_master/config.yaml")
EXP_ROOT = Path("/data/exp_data")
COMPETITIONS = ["detecting-insults-in-social-commentary","plant-pathology-2020-fgvc7"]
```
2. 然后在配置文件中做如下修改：
```yaml
exp_id: "X"  
symlinks:"/data/exp_data/X/prepared/public": "input"
```
3. 运行脚本即可：
```bash
python playground/ml_master/scripts/mlmaster_test.py
```
4. 脚本会在`.tmp_configs`目录下自动创建适配每个比赛的配置，并在`runs`目录下根据比赛名称自动创建工作目录

### 查看结果
- 所有运行日志和文件会被保存在`runs`目录下
- 可视化：在根目录下指定对应的工作目录（示例：runs/ml_master_xxx）运行命令，然后访问对应端口（默认http://127.0.0.1:8765）：
```bash
python -m playground.ml_master.vis.app --run_dir runs/ml_master_xxx
```

## 开发与扩展

- 在 `prompts/` 中调整或新增提示模板以更改智能体生成策略
- 在 `core/exp/` 下扩展或修改实验流程（draft、research、improve 等）
- 使用内嵌评分服务器进行本地开发时的提交验证

## ✍️ Citation

如果认为这对你有帮助，欢迎引用以下文章：

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


