# Skills 模块

Skills 模块为 EvoMaster 提供技能系统，支持可按需加载的可复用能力。

## 概述

```
evomaster/skills/
├── base.py           # BaseSkill, Skill, SkillRegistry
└── {skill_name}/
    ├── SKILL.md         # 技能定义（frontmatter + body）
    ├── scripts/         # 可选的可执行脚本
    │   ├── main.py
    │   └── helper.sh
    └── references/      # 可选的参考文档
        └── api.md
```

## 统一技能模型

在 v0.0.2 中，之前的 `KnowledgeSkill` 和 `OperatorSkill` 子类已被移除。所有技能现在由统一的 `Skill` 类表示，支持所有三个层级的信息：

- **Level 1 (meta_info)**：~100 tokens，始终在上下文中
- **Level 2 (full_info)**：500-2000 tokens，按需加载
- **Level 3 (scripts)**：可选的可执行代码，通过工具调用运行

技能是否包含脚本取决于技能目录中是否存在 `scripts/` 目录，而非类型字段。

## SkillMetaInfo

从 SKILL.md frontmatter 解析的元数据。

```python
class SkillMetaInfo(BaseModel):
    """技能元信息（Level 1）

    从 SKILL.md 的 YAML frontmatter 解析得到。
    始终在上下文中，帮助 Agent 决定是否使用该技能。
    """
    name: str = Field(description="技能名称")
    description: str = Field(description="技能描述，包含使用场景")
    license: str | None = Field(default=None, description="许可证信息")
    type: str | None = Field(default=None, description="技能类型，如 'openclaw' 表示 Openclaw 插件技能")
    tool_name: str | None = Field(default=None, description="Openclaw 工具名称，如 'feishu_doc'")
```

## BaseSkill

所有技能的抽象基类。

```python
class BaseSkill(ABC):
    """技能基类

    Skills 是 EvoMaster 的技能组件，包含：
    - Level 1 (meta_info)：技能元信息（~100 tokens），始终在上下文
    - Level 2 (full_info)：完整信息（500-2000 tokens），按需加载
    - Level 3 (scripts)：可选的可执行脚本
    """

    def __init__(self, skill_path: Path):
        """初始化 Skill

        Args:
            skill_path: 技能目录路径
        """

    def get_full_info(self) -> str:
        """获取完整信息（Level 2）

        若存在 job_submit.md 则返回其内容；否则从 SKILL.md 的 body 提取。

        Returns:
            完整的技能信息文本
        """

    def get_reference(self, reference_name: str) -> str:
        """获取参考文档内容

        Args:
            reference_name: 参考文档名称（如 "forms.md", "reference/api.md"）

        Returns:
            参考文档内容
        """

    @abstractmethod
    def to_context_string(self) -> str:
        """转换为上下文字符串

        返回应该添加到 Agent 上下文中的字符串。
        """
```

## Skill

统一的 `BaseSkill` 具体实现。替代之前的 `KnowledgeSkill` 和 `OperatorSkill` 类。

```python
class Skill(BaseSkill):
    """Skill 具体实现

    包含可选的可执行脚本：
    - Level 1：meta_info（始终在上下文）
    - Level 2：full_info（按需加载）
    - Level 3：scripts（可选的可执行脚本）
    """

    def __init__(self, skill_path: Path):
        super().__init__(skill_path)
        self.scripts_dir = self.skill_path / "scripts"
        self.available_scripts = self._scan_scripts()

    def _scan_scripts(self) -> list[Path]:
        """扫描 scripts 目录获取可执行脚本

        Returns:
            脚本路径列表（.py, .sh, .js）
        """

    def get_script_path(self, script_name: str) -> Path | None:
        """按名称获取脚本路径

        Args:
            script_name: 脚本名称

        Returns:
            脚本路径，如果不存在则返回 None
        """

    def to_context_string(self) -> str:
        """返回 meta_info 和可用脚本列表"""
        scripts_info = ", ".join([s.name for s in self.available_scripts]) if self.available_scripts else "No scripts"
        return f"[Skill: {self.meta_info.name}] {self.meta_info.description} (Scripts: {scripts_info})"
```

## SkillRegistry

管理所有可用技能的注册表。

```python
class SkillRegistry:
    """技能注册中心

    管理所有可用的 Skills，支持：
    - 自动发现和加载
    - 按名称过滤加载
    - 按需检索
    - 创建子集用于 per-agent 技能视图
    - 提供 meta_info 供 Agent 选择
    """

    def __init__(self, skills_root: Path, skills: list[str] | None = None):
        """初始化 SkillRegistry

        Args:
            skills_root: skills 根目录
            skills: 可选的 skill 目录名列表；None 表示加载全部
        """

    def load_from_directory(self, directory: Path, skills: list[str] | None = None) -> None:
        """从额外目录加载 skills

        可多次调用以从多个目录加载 skills。

        Args:
            directory: 额外的 skills 目录
            skills: 可选的 skill 目录名列表；None 表示加载全部
        """

    def get_skill(self, name: str) -> Skill | None:
        """按名称获取技能

        Args:
            name: 技能名称

        Returns:
            Skill 对象，如果不存在则返回 None
        """

    def get_all_skills(self) -> list[Skill]:
        """获取所有技能"""

    def get_meta_info_context(self) -> str:
        """获取所有技能的 meta_info，用于添加到 Agent 上下文

        Returns:
            包含所有技能 meta_info 的字符串
        """

    def create_subset(self, skill_names: list[str]) -> SkillRegistry:
        """创建仅包含指定技能的子集 SkillRegistry

        用于为每个 Agent 创建独立的、过滤后的技能视图。

        Args:
            skill_names: 要保留的技能名称列表

        Returns:
            新的 SkillRegistry 实例，仅包含指定的技能
        """

    def search_skills(self, query: str) -> list[Skill]:
        """按关键词搜索技能

        Args:
            query: 搜索关键词

        Returns:
            匹配的技能列表
        """
```

## SKILL.md 格式

### Frontmatter（YAML）

```yaml
---
name: skill-name
description: 简要描述，包含使用场景和触发条件
license: MIT
---
```

注意：`skill_type` 字段在 v0.0.2 中已被移除。技能是否包含脚本由 `scripts/` 目录是否存在决定。

### Body（Markdown）

body 部分包含 full_info（Level 2）：

```markdown
# 技能名称

## 概述

详细描述此技能的功能。

## 使用场景

何时使用此技能：
- 场景 1
- 场景 2

## 详情

技术细节、参数、示例等。

## 参考

- [参考 1](./references/ref1.md)
- [参考 2](./references/ref2.md)
```

## 目录结构

所有技能遵循相同的目录结构：

```
evomaster/skills/
└── my_skill/
    ├── SKILL.md           # 技能定义
    ├── scripts/           # 可选的可执行脚本
    │   ├── main.py
    │   └── helper.sh
    └── references/        # 可选的参考文档
        └── api.md
```

如果 `scripts/` 目录存在且包含文件，则技能具有可执行脚本（Level 3）。否则，技能仅提供知识（Level 1 和 Level 2）。

## 使用示例

### 在 Playground 中加载 Skills（v0.0.2）

Skills 现在在配置文件中按 agent 配置：

```yaml
# config.yaml
agents:
  search_agent:
    llm: "openai"
    skills:            # 为此 agent 加载所有 skills
      - "*"
  summarize_agent:
    llm: "openai"
    skills:            # 仅加载指定 skills
      - "rag"
      - "pdf"
  plan_agent:
    llm: "openai"
    # 无 skills 键 -> 不为此 agent 加载任何 skill
```

### 编程方式使用

```python
from evomaster.skills import SkillRegistry, Skill
from pathlib import Path

# 从根目录加载所有 skills
registry = SkillRegistry(Path("evomaster/skills"))

# 按名称过滤加载
registry = SkillRegistry(Path("evomaster/skills"), skills=["rag", "pdf"])

# 从额外目录加载 skills
registry.load_from_directory(Path("extra_skills"))

# 获取所有技能
all_skills = registry.get_all_skills()

# 获取 Agent 上下文的 meta_info
context = registry.get_meta_info_context()

# 为特定 agent 创建子集
subset = registry.create_subset(["rag", "pdf"])

# 搜索技能
results = registry.search_skills("rag")
```

### 通过 SkillTool 使用 Skills

Agent 可以通过 `use_skill` 工具使用技能：

```python
# 获取技能信息
{"action": "get_info", "skill_name": "rag"}

# 获取参考文档
{"action": "get_reference", "skill_name": "rag", "reference_name": "api.md"}

# 运行脚本
{"action": "run_script", "skill_name": "rag", "script_name": "search.py", "script_args": "--query 'search term'"}
```

### 创建新技能

1. 创建技能目录：
```bash
mkdir -p evomaster/skills/my_skill
```

2. 创建 SKILL.md：
```markdown
---
name: my-skill
description: 一个帮助完成 XYZ 任务的技能。当需要做 ABC 时使用。
license: MIT
---

# 我的技能

## 概述

此技能提供关于 XYZ 的知识...

## 使用场景

- 当需要理解 ABC 时
- 当处理 DEF 概念时

## 详情

详细信息在这里...
```

3. 添加脚本（可选）：
```bash
mkdir -p evomaster/skills/my_skill/scripts
cat > evomaster/skills/my_skill/scripts/run.py << 'EOF'
# 你的可执行脚本
EOF
```

4. 添加参考文档（可选）：
```bash
mkdir -p evomaster/skills/my_skill/references
echo "# 参考文档" > evomaster/skills/my_skill/references/guide.md
```

## 相关文档

- [架构概述](./architecture.md)
- [Tools 模块](./tools.md)
- [Agent 模块](./agent.md)
