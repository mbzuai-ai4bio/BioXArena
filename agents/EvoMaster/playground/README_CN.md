# Playground

Playground 是开发者构建自己科研智能体的工作区。每个 playground 定义了一个完整的实验工作流，通过继承 EvoMaster 的基础组件（`BasePlayground`、`BaseExp`）来实现特定的科学实验自动化。

**开发者应该在此目录下创建自己的 playground，实现自己的科研智能体。**

## 现有示例

| Playground | 描述 | 文档 |
|------------|------|------|
| `minimal` | 基础单智能体 | [README](./playground/minimal/README_CN.md) |
| `minimal_bohrium` | 玻尔平台科学计算 | [README](./playground/minimal_bohrium/README_CN.md) |
| `minimal_kaggle` | 简易Kaggle 竞赛自动化 | [README](./playground/minimal_kaggle/README_CN.md) |
| `minimal_multi_agent` | 简易多智能体 | [README](./playground/minimal_multi_agent/README_CN.md) |
| `minimal_multi_agent_parallel` | 并行多智能体实验 | [README](./playground/minimal_multi_agent_parallel/README_CN.md) |
| `minimal_openclaw_skill` | TypeScript 技能接入 | [README](./playground/minimal_openclaw_skill/README_CN.md) |
| `minimal_skill_task` | Anthropic原生技能接入 | [README](./playground/minimal_skill_task/README_CN.md) |
| `ml_master` | ML-Master 1.0 自主机器学习 | [README](./playground/ml_master/README_CN.md) |
| `ml_master_2` | ML-Master 2.0 认知积累框架 | [README](./playground/ml_master_2/README_CN.md) |
| `x_master` | X-Master科学智能体 | [README](./playground/x_master/README_CN.md) |
| `browse_master` | Browse-Master网页搜索智能体 | [README](./playground/browse_master/README_CN.md) |


## 快速开始：创建你的 Playground

### 1. 创建目录结构

```bash
mkdir -p playground/my_agent/core
mkdir -p playground/my_agent/prompts
mkdir -p configs/my_agent
```

### 2. 实现 Playground 类

`playground/my_agent/core/playground.py`:

```python
import logging
from pathlib import Path
from evomaster.core import BasePlayground, register_playground

@register_playground("my_agent")
class MyPlayground(BasePlayground):
    def __init__(self, config_dir=None, config_path=None):
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "my_agent"
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
```

这是最小实现。如果需要多智能体、自定义工具，自定义实验流程，可以参考 `minimal_multi_agent` ，`chat_aegnt`和`minimal_kaggle`。

### 3. 编写提示词

`playground/my_agent/prompts/system_prompt.txt`:

```
你是一个科研智能体。请根据任务描述进行分析、实验和总结。
```

`playground/my_agent/prompts/user_prompt.txt`:

```
任务 ID：{task_id}
描述：{description}
{input_data}
```

### 4. 配置

`configs/my_agent/config.yaml`:


### 5. 运行

```bash
python run.py --agent my_agent --task "你的任务描述"
```

更多细节请参考 [开发文档](../docs/architecture.md)。
