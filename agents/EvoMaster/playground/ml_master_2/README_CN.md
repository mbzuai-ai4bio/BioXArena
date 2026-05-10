# ML-Master 2.0

**ML-Master 2.0** 是一个前沿的的智能体科学框架，通过认知积累应对超长程自主任务的挑战。其采用分层认知缓存（HCC）架构，将瞬态执行轨迹动态提炼为稳定的长期知识，确保战术执行与战略规划在复杂、长时域科学探索中保持解耦却又协同演化。
在这里，我们基于EvoMaster框架重新实现和开源了ML-Master 2.0的代码。为了方便运行，我们提供了一个可以快速运行的示例任务`detecting-insults-in-social-commentary`和这个任务在运行后提炼得到的示例数据库。

## 架构

系统实现了一个由专用智能体组成的流水线：

1. **Prefetch（预取）** - 通过 RAG（检索增强生成）从知识库中检索相关知识
2. **Draft（起草）** - 根据任务描述和检索到的知识生成初始 ML 解决方案
3. **Debug（调试）** - 当代码运行失败时修复错误（最多重试 3 次）
4. **Research（研究）** - 提出结构化的改进方向和具体想法
5. **Improve（改进）** - 并行实现多个改进想法，每个独立验证
6. **Knowledge Promotion（知识提炼）** - 总结每轮研究中哪些有效、哪些无效
7. **Wisdom Promotion（智慧提炼）** - 提取可复用的经验用于未来任务（超时时触发）
8. **Metric（评估）** - 从终端输出中提取验证分数

### 工作流程

```
预取 -> 起草 -> [研究 -> 并行改进]* (最多 20 轮) -> 知识提炼
                                                        |
                                        (超时时) -> 智慧提炼
```

### 核心特性

- **并行执行**：多个改进想法并发运行，工作空间隔离
- **评分服务器**：在接受结果前验证提交格式
- **超时看门狗**：24 小时硬性时间限制，优雅关闭并提取智慧
- **RAG 知识检索**：基于向量嵌入检索历史实验经验
- **分数比较**：正确处理 NaN/None 值和优化方向

## 项目结构

```
playground/ml_master_2/
├── __init__.py
├── agent/
│   └── session/
│       └── local.py              # 自定义本地 Session（支持软链接）
├── core/
│   ├── playground.py             # 主编排器
│   ├── exp/
│   │   ├── draft_exp.py          # 初始方案生成
│   │   ├── improve_exp.py        # 改进实现
│   │   ├── research_exp.py       # 研究方向规划
│   │   ├── prefetch_exp.py       # RAG 知识检索
│   │   ├── knowledge_promotion_exp.py  # 轮次总结生成
│   │   └── wisdom_promotion_exp.py     # 可复用智慧提取
│   └── utils/
│       ├── code.py               # 代码提取和提交文件处理
│       ├── data_preview.py       # 数据集预览生成
│       ├── grading.py            # 提交验证客户端
│       ├── grading_server.py     # 内嵌 Flask 评分服务器
│       └── watch_dog.py          # 超时控制
├── env/
│   └── local.py                  # 自定义本地环境
├── prompts/                      # 智能体提示词模板
├── example_wisdom/               # 示例知识库
└── data/                         # 竞赛数据集
```

## 快速开始

### 前置条件

- Python 3.12
- 已安装 EvoMaster 基础需求框架（详见项目根目录的 README.md）
- LLM API 端点（OpenAI、Anthropic 或本地部署如 DeepSeek）


### 安装

1. 根据 [MLE-Bench](https://github.com/openai/mle-bench) 官方指引安装其环境

2. 安装 ML-Master 2 依赖（在项目根目录下执行）：

```bash
cd playground/ml_master_2
pip install -r requirements.txt
```

3. 下载 MLE-Bench 完整数据集（可选。若想快速运行，项目已在 `playground/ml_master_2/data` 中附带 `detecting-insults-in-social-commentary` 任务的数据集）：

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

   > ML-Master 2 在 EvoMaster 中通过符号链接访问数据集，可将数据下载到任意位置，ML-Master 会自动建立链接。


### 配置

1. 在项目根目录创建 `.env` 文件，配置 LLM 环境变量：

```bash
cp .env.template .env
# 根据 configs/ml_master_2/deepseek-v3.2-example.yaml 中使用的模型，配置用于写代码和迭代的主模型 API
DEEPSEEK_API_KEY=""
DEEPSEEK_API_BASE=""
# 配置 OpenAI 的 embedding API
OPENAI_API_KEY=""
GPT_CHAT_MODEL=""
GPT_BASE_URL=""
```

2. 在 `configs/ml_master_2/deepseek-v3.2-example.yaml` 中完成相关配置：

```yaml
# 比赛的 competition_id
competition_id: "detecting-insults-in-social-commentary"
# 评价指标方向
is_lower_better: false
# 准备好的原始数据位置，供 grading server 使用（格式可参考 playground/ml_master_2/data）
data_root: "./playground/ml_master_2/data"
# 更多配置说明请参见示例配置文件
```

### 运行

在项目根目录下，指定配置的 yaml 文件和任务描述文件（如 `description.md`）运行 ML-Master 2：

```bash
python run.py --agent ml_master_2 --config configs/ml_master_2/deepseek-v3.2-example.yaml --task playground/ml_master_2/data/detecting-insults-in-social-commentary/prepared/public/description.md
```

### 查看结果
所有运行日志和文件会被保存在`runs`目录下

## ✍️ Citation

如果认为这对你有帮助，欢迎引用以下文章

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