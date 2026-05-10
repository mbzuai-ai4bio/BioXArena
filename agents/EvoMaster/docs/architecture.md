# Architecture Overview

EvoMaster is an agent system designed for iteratively completing scientific experiment tasks, focusing on MLE, Physics, Embodied AI, and other scientific domains.

## System Architecture

```
EvoMaster/
├── evomaster/              # Core library
│   ├── agent/              # Agent components
│   │   ├── agent.py        # BaseAgent, Agent classes
│   │   ├── context.py      # Context management
│   │   ├── session/        # Session implementations
│   │   └── tools/          # Tool system
│   ├── core/               # Workflow components
│   │   ├── exp.py          # BaseExp class
│   │   └── playground.py   # BasePlayground class
│   ├── env/                # Environment management
│   ├── skills/             # Skill system
│   └── utils/              # Utilities (LLM, Types)
├── playground/             # Playground implementations
├── configs/                # Configuration files
└── docs/                   # Documentation
```

## Three-Layer Architecture

EvoMaster follows a three-layer architecture:

```
┌─────────────────────────────────────────────────┐
│                  Playground                      │
│  (Workflow orchestration, parallel execution)   │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│                     Exp                          │
│  (Single experiment execution, task instance)   │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│                    Agent                         │
│  (LLM + Tools + Memory, intelligent decision)   │
└─────────────────────────────────────────────────┘
```

### Layer 1: Playground

**BasePlayground** is the workflow orchestrator that manages:
- Configuration loading and component initialization
- Session lifecycle management
- Multi-agent coordination and parallel execution
- MCP server connections
- Result aggregation

### Layer 2: Exp

**BaseExp** represents a single experiment execution:
- Receives task description
- Creates TaskInstance
- Runs Agent and collects trajectory
- Saves results

### Layer 3: Agent

**BaseAgent** is the intelligent core:
- Manages dialog (conversation history)
- Executes tool calls
- Handles context truncation
- Records execution trajectory

## Component Interaction

```
┌────────────────────────────────────────────────────────────┐
│                        Playground                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   LLM    │  │ Session  │  │  Tools   │  │  Skills  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       └─────────────┴─────────────┴─────────────┘         │
│                           │                                │
│                    ┌──────▼──────┐                        │
│                    │    Agent    │                        │
│                    └──────┬──────┘                        │
│                           │                                │
│                    ┌──────▼──────┐                        │
│                    │     Exp     │                        │
│                    └─────────────┘                        │
└────────────────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. Separation of Concerns
- **Playground**: Orchestration and lifecycle
- **Exp**: Single task execution
- **Agent**: Intelligent decision making

### 2. Extensibility
- Custom Playgrounds for different workflows
- Custom Agents for different behaviors
- Custom Tools via ToolRegistry
- MCP protocol for external tools

### 3. Thread Safety
- Each Exp has independent Agent instance
- Parallel execution at Playground level
- Thread-safe trajectory recording

### 4. Context Management
- Automatic context truncation
- Multiple truncation strategies
- Token counting support

## Data Flow

```
Task Description
       │
       ▼
┌─────────────────┐
│   Playground    │──────────────────────┐
│    setup()      │                      │
└───────┬─────────┘                      │
        │                                │
        ▼                                │
┌─────────────────┐                      │
│  Create Agent   │◄── LLM, Session,     │
│                 │    Tools, Skills     │
└───────┬─────────┘                      │
        │                                │
        ▼                                │
┌─────────────────┐                      │
│   Create Exp    │                      │
└───────┬─────────┘                      │
        │                                │
        ▼                                │
┌─────────────────┐                      │
│   Exp.run()     │                      │
│  ┌───────────┐  │                      │
│  │Agent.run()│  │                      │
│  │  └─step() │  │                      │
│  │    └─LLM  │  │                      │
│  │    └─Tool │  │                      │
│  └───────────┘  │                      │
└───────┬─────────┘                      │
        │                                │
        ▼                                │
┌─────────────────┐                      │
│   Trajectory    │◄─────────────────────┘
│    (Result)     │
└─────────────────┘
```

## Related Documentation

- [Agent Module](./agent.md)
- [Core Module](./core.md)
- [Tools Module](./tools.md)
- [Skills Module](./skills.md)
- [LLM Module](./llm.md)
