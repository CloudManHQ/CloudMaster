# Agent Harness 完整指南：生产级 Agent 评估框架

> 全面解析 Agent 评估体系：从 Agent Harness 到 GAIA、OSWorld、SWE-bench，构建可靠的 Agent 能力评估标准
> 
> 更新时间: 2026-04 | 覆盖: Agent Harness, GAIA, OSWorld, SWE-bench, ToolBench, MLAgentBench

---

## 📋 目录

1. [Agent 评估概述](#一agent-评估概述)
2. [Agent Harness 架构](#二agent-harness-架构)
3. [主流 Agent 基准测试](#三主流-agent-基准测试)
4. [评估维度与指标](#四评估维度与指标)
5. [构建自定义 Agent Harness](#五构建自定义-agent-harness)
6. [生产环境评估实践](#六生产环境评估实践)
7. [2026 评估技术趋势](#七2026-评估技术趋势)

---

## 一、Agent 评估概述

### 1.1 为什么需要 Agent Harness？

传统 LLM 评估（如 MMLU、HumanEval）仅测试**静态知识和代码生成能力**，无法评估 Agent 的**动态决策、工具使用、长程规划**等核心能力。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LLM 评估 vs Agent 评估                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LLM 评估 (静态)                                                        │
│  ────────────────                                                       │
│  Input: "What is the capital of France?"                                │
│  Output: "Paris"                                                        │
│  Metric: Exact Match / Perplexity                                       │
│                                                                         │
│  局限:                                                                   │
│  • 单次交互，无状态                                                     │
│  • 无法评估工具使用                                                     │
│  • 无法评估错误恢复                                                     │
│  • 无法评估长程规划                                                     │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Agent 评估 (动态) ★                                                    │
│  ─────────────────                                                      │
│  Task: "Book a flight from NYC to Paris next Monday"                    │
│                                                                         │
│  Step 1: Agent ──► Search API ──► Flight options                        │
│  Step 2: Agent ──► Calendar API ──► Check availability                  │
│  Step 3: Agent ──► Booking API ──► Make reservation                     │
│  Step 4: Agent ──► Email API ──► Send confirmation                      │
│                                                                         │
│  Metrics:                                                               │
│  • Task Success Rate (任务成功率)                                       │
│  • Steps to Success (成功步数)                                          │
│  • Tool Use Accuracy (工具使用准确率)                                    │
│  • Error Recovery (错误恢复能力)                                         │
│  • Cost Efficiency (成本效率)                                           │
│  • Safety Score (安全评分)                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Agent 评估的独特挑战

| 挑战 | 描述 | 解决方案 |
|------|------|----------|
| **非确定性** | Agent 可能有多种正确路径 | 多维度评估，非单一标准 |
| **环境依赖** | 需要真实或仿真环境 | 容器化、沙箱化环境 |
| **状态管理** | 需要维护环境状态 | 状态快照与回滚 |
| **成本问题** | LLM API 调用成本高 | 缓存、采样评估 |
| **安全性** | Agent 可能执行危险操作 | 沙箱、权限控制 |
| **可重复性** | 外部 API 可能变化 | Mock 服务、版本锁定 |

---

## 二、Agent Harness 架构

### 2.1 核心组件

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Agent Harness 架构                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      Task Definition                              │  │
│  │  • 任务描述 (自然语言)                                             │  │
│  │  • 初始状态配置                                                    │  │
│  │  • 成功判定条件                                                    │  │
│  │  • 评估指标定义                                                    │  │
│  └───────────────────────────┬──────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      Environment (Sandbox)                        │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │  │
│  │  │   OS Env     │  │   Web Env    │  │   Code Env   │          │  │
│  │  │  (Ubuntu/    │  │  (Browser    │  │  (Python/    │          │  │
│  │  │   Windows)   │  │   Container) │  │   Jupyter)   │          │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │  │
│  │                                                                   │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │  │
│  │  │   DB Env     │  │   API Env    │  │   Game Env   │          │  │
│  │  │  (Postgres/  │  │  (Mock       │  │  (Minecraft/ │          │  │
│  │  │   MongoDB)   │  │   Services)  │  │   Crafting)  │          │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │  │
│  └───────────────────────────┬──────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      Agent Execution Loop                         │  │
│  │                                                                   │  │
│  │   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │  │
│  │   │  Agent  │───►│  Action │───►│   Env   │───►│Observation│    │  │
│  │   │         │◄───│         │◄───│         │◄───│         │    │  │
│  │   └─────────┘    └─────────┘    └─────────┘    └─────────┘     │  │
│  │        ▲                                            │           │  │
│  │        └────────────────────────────────────────────┘           │  │
│  │                      (Loop until done or max steps)              │  │
│  └───────────────────────────┬──────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      Evaluation Engine                            │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │  │
│  │  │   Success    │  │   Metrics    │  │   Traces     │          │  │
│  │  │   Checker    │  │   Computer   │  │   Analyzer   │          │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 环境类型详解

#### OS Environment (操作系统环境)

```python
class OSEnvironment:
    """
    OSWorld, Ubuntu Arena 等使用的操作系统环境
    """
    
    def __init__(self):
        self.container = DockerContainer(
            image="ubuntu:22.04",
            resources={
                "cpu": 4,
                "memory": "8GB",
                "disk": "20GB"
            }
        )
        
        # 桌面环境 (用于 GUI 任务)
        self.desktop = VNCDesktop(
            resolution=(1920, 1080),
            fps=30
        )
        
        # 预装软件
        self.preinstalled_apps = [
            "chrome", "firefox", "vscode",
            "libreoffice", "terminal"
        ]
    
    def execute(self, action: str) -> Observation:
        """
        执行操作，返回观察
        """
        if action.startswith("click"):
            x, y = parse_click(action)
            self.desktop.click(x, y)
        elif action.startswith("type"):
            text = parse_type(action)
            self.desktop.type(text)
        elif action.startswith("bash"):
            command = parse_bash(action)
            result = self.container.execute(command)
            return Observation(type="terminal", content=result)
        
        # 返回屏幕截图
        screenshot = self.desktop.capture()
        return Observation(type="image", content=screenshot)
```

#### Web Environment (浏览器环境)

```python
class WebEnvironment:
    """
    浏览器环境，用于 Web Agent 评估
    """
    
    def __init__(self):
        self.browser = PlaywrightBrowser(
            headless=True,
            viewport={"width": 1280, "height": 720}
        )
        
        # 网络拦截与 Mock
        self.mock_server = MockServer()
        
        # 访问记录
        self.history = []
    
    def navigate(self, url: str):
        """导航到 URL"""
        if url in self.mock_server.routes:
            page = self.mock_server.get_mock_page(url)
        else:
            page = self.browser.goto(url)
        
        self.history.append({"action": "navigate", "url": url})
        return self.get_observation(page)
    
    def execute_action(self, action: WebAction):
        """执行网页操作"""
        if action.type == "click":
            self.browser.click(action.selector)
        elif action.type == "fill":
            self.browser.fill(action.selector, action.value)
        elif action.type == "select":
            self.browser.select(action.selector, action.value)
        
        return self.get_observation()
```

#### Code Environment (代码环境)

```python
class CodeEnvironment:
    """
    SWE-bench, HumanEval 等使用的代码执行环境
    """
    
    def __init__(self, repo: str, commit: str):
        self.sandbox = Sandbox()
        
        # 克隆代码仓库
        self.sandbox.execute(f"git clone {repo} /workspace/repo")
        self.sandbox.execute(f"cd /workspace/repo && git checkout {commit}")
        
        # 安装依赖
        self.sandbox.execute("pip install -e /workspace/repo")
        
        # 测试框架
        self.test_runner = PyTestRunner()
    
    def apply_patch(self, patch: str):
        """应用代码修改"""
        self.sandbox.write_file("/workspace/patch.diff", patch)
        self.sandbox.execute("cd /workspace/repo && git apply /workspace/patch.diff")
    
    def run_tests(self, test_file: str) -> TestResult:
        """运行测试"""
        return self.test_runner.run(f"/workspace/repo/{test_file}")
```

---

## 三、主流 Agent 基准测试

### 3.1 基准测试全景

| 基准 | 环境 | 任务类型 | 难度 | 2026 SOTA |
|------|------|----------|------|-----------|
| **Agent Harness** | 通用 | 综合 | ⭐⭐⭐⭐⭐ | 68.2% |
| **GAIA** | OS/Web | 真实世界问答 | ⭐⭐⭐⭐⭐ | 65.4% |
| **OSWorld** | Ubuntu | OS 操作 | ⭐⭐⭐⭐⭐ | 38.5% |
| **SWE-bench** | Code | 真实代码修复 | ⭐⭐⭐⭐⭐ | 56.7% |
| **WebArena** | Web | 网站导航任务 | ⭐⭐⭐⭐ | 45.2% |
| **Mind2Web** | Web | 网页操作 | ⭐⭐⭐⭐ | 52.8% |
| **ToolBench** | API | 工具调用 | ⭐⭐⭐⭐ | 78.3% |
| **MLAgentBench** | ML | 机器学习研究 | ⭐⭐⭐⭐⭐ | 42.1% |
| **TravelPlanner** | Web | 旅行规划 | ⭐⭐⭐⭐ | 71.5% |
| **ScienceWorld** | Text Game | 科学实验 | ⭐⭐⭐ | 85.2% |

### 3.2 GAIA (General AI Assistants)

GAIA 是评估通用 AI 助手能力的基准，任务需要**多步骤推理、工具使用、真实世界知识**。

#### 任务示例

```
Level 1 (简单):
"2023 年诺贝尔物理学奖的获得者是谁？"
→ 需要：搜索能力

Level 2 (中等):
"找出 2023 年全球票房最高的三部电影，并计算它们的平均评分"
→ 需要：搜索、数据提取、计算

Level 3 (困难):
"找出过去 5 年在 arXiv 上发表的引用次数最多的关于 Transformer 的论文，
 分析其引用趋势，并生成一个图表"
→ 需要：学术搜索、数据分析、绘图工具
```

#### 评估指标

```python
class GAIAEvaluator:
    """
    GAIA 评估器
    """
    
    def evaluate(self, prediction: str, ground_truth: str, level: int):
        """
        GAIA 使用不同类型的答案和评估方法
        """
        if level == 1:
            # 短答案，精确匹配
            score = exact_match(prediction, ground_truth)
        
        elif level == 2:
            # 可能包含多个部分
            score = fuzzy_match(prediction, ground_truth)
        
        elif level == 3:
            # 复杂答案，需要多维度评估
            scores = {
                "completeness": check_completeness(prediction, ground_truth),
                "accuracy": check_accuracy(prediction, ground_truth),
                "source_verification": verify_sources(prediction)
            }
            score = weighted_average(scores)
        
        return {
            "score": score,
            "level": level,
            "steps_taken": self.get_step_count(),
            "tools_used": self.get_tool_usage(),
            "time_elapsed": self.get_time()
        }
```

### 3.3 OSWorld

OSWorld 是在真实 Ubuntu 操作系统环境中评估 Agent 的基准。

#### 任务类型

| 类别 | 示例任务 | 难度 |
|------|----------|------|
| **文件管理** | "将下载文件夹中所有 PDF 移动到文档文件夹" | ⭐⭐ |
| **软件安装** | "安装 VLC 播放器并设置为默认视频播放器" | ⭐⭐⭐ |
| **系统设置** | "设置系统自动登录并更改桌面壁纸" | ⭐⭐⭐ |
| **办公任务** | "在 LibreOffice Writer 中创建表格并填充数据" | ⭐⭐⭐⭐ |
| **编程任务** | "配置 Python 环境并运行一个 Flask 应用" | ⭐⭐⭐⭐ |
| **故障排查** | "诊断为什么无法连接到 WiFi 并修复" | ⭐⭐⭐⭐⭐ |

#### 观察空间

```python
class OSWorldObservation:
    """
    OSWorld 观察空间
    """
    
    def __init__(self):
        self.screenshot = None  # 屏幕截图
        self.accessibility_tree = None  # 无障碍树 (UI 元素层级)
        self.terminal_output = None  # 终端输出
        self.system_info = {  # 系统状态
            "active_window": "",
            "clipboard": "",
            "running_processes": []
        }
```

### 3.4 SWE-bench

SWE-bench 是评估 Agent 修复真实 GitHub Issue 能力的基准。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SWE-bench 流程                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Issue 提取                                                            │
│     从 GitHub 提取真实 Issue (描述 + 评论)                                 │
│                                                                         │
│  2. 环境搭建                                                              │
│     克隆仓库到特定 commit                                                 │
│     安装依赖                                                              │
│                                                                         │
│  3. Agent 执行                                                            │
│     Agent 阅读 Issue                                                      │
│     探索代码库                                                            │
│     生成修复补丁 (diff)                                                   │
│                                                                         │
│  4. 评估                                                                   │
│     应用补丁                                                               │
│     运行回归测试 (确保没破坏其他功能)                                        │
│     运行针对性测试 (验证 Issue 已修复)                                       │
│                                                                         │
│  成功标准: 所有测试通过                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.5 WebArena

WebArena 提供真实网站（购物、社交、开发工具等）的独立副本供 Agent 测试。

#### 支持的网站

- **电商**: One Stop Market (购物网站)
- **开发**: GitLab (代码托管)
- **协作**: Rocket.Chat (聊天)
- **办公**: Odoo (ERP)
- **地图**: OpenStreetMap

---

## 四、评估维度与指标

### 4.1 多维度评估框架

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Agent 评估维度                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     1. 任务完成度 (Task Completion)                │  │
│  │  ┌────────────────────────────────────────────────────────────┐   │  │
│  │  │ Success Rate        │ 任务成功率 (最重要指标)                │   │  │
│  │  │ Partial Credit      │ 部分完成得分                          │   │  │
│  │  │ Quality Score       │ 完成质量评分                          │   │  │
│  │  └────────────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     2. 效率指标 (Efficiency)                       │  │
│  │  ┌────────────────────────────────────────────────────────────┐   │  │
│  │  │ Steps to Success    │ 成功所需步数 (越少越好)                │   │  │
│  │  │ Token Usage         │ LLM Token 消耗                        │   │  │
│  │  │ API Calls           │ API 调用次数                          │   │  │
│  │  │ Time to Complete    │ 完成时间                              │   │  │
│  │  │ Cost per Task       │ 单任务成本                            │   │  │
│  │  └────────────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     3. 能力指标 (Capabilities)                     │  │
│  │  ┌────────────────────────────────────────────────────────────┐   │  │
│  │  │ Tool Use Accuracy   │ 工具使用准确率                        │   │  │
│  │  │ Error Recovery      │ 错误恢复能力                          │   │  │
│  │  │ Planning Quality    │ 规划质量                              │   │  │
│  │  │ Context Utilization │ 上下文利用                            │   │  │
│  │  │ Multi-step Reasoning│ 多步推理能力                          │   │  │
│  │  └────────────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     4. 安全指标 (Safety)                           │  │
│  │  ┌────────────────────────────────────────────────────────────┐   │  │
│  │  │ Harmful Action Prevention │ 有害动作预防                  │   │  │
│  │  │ Data Privacy Protection   │ 数据隐私保护                  │   │  │
│  │  │ Sandbox Escape Attempts   │ 沙箱逃逸尝试                  │   │  │
│  │  │ Resource Limits           │ 资源限制遵守                  │   │  │
│  │  └────────────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     5. 用户体验 (UX)                               │  │
│  │  ┌────────────────────────────────────────────────────────────┐   │  │
│  │  │ Explanation Quality │ 解释质量                              │   │  │
│  │  │ Transparency        │ 透明度                                │   │  │
│  │  │ Interruptibility    │ 可中断性                              │   │  │
│  │  │ Human Alignment     │ 人类对齐                              │   │  │
│  │  └────────────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 指标计算公式

```python
class AgentMetrics:
    """
    Agent 评估指标计算
    """
    
    @staticmethod
    def success_rate(results: List[TaskResult]) -> float:
        """任务成功率"""
        successes = sum(1 for r in results if r.success)
        return successes / len(results)
    
    @staticmethod
    def normalized_steps(result: TaskResult, optimal_steps: int) -> float:
        """
        归一化步数
        1.0 = 最优步数，越低越好
        """
        return optimal_steps / max(result.steps, optimal_steps)
    
    @staticmethod
    def tool_efficiency(result: TaskResult) -> float:
        """
        工具使用效率
        正确工具调用 / 总工具调用
        """
        correct = sum(1 for call in result.tool_calls if call.correct)
        return correct / len(result.tool_calls)
    
    @staticmethod
    def cost_efficiency(result: TaskResult) -> float:
        """
        成本效率得分
        考虑任务复杂度
        """
        base_cost = result.token_cost * 0.001 + result.api_cost
        complexity_factor = result.task_complexity
        return complexity_factor / (base_cost + 0.01)
    
    @staticmethod
    def composite_score(result: TaskResult, weights: Dict[str, float]) -> float:
        """
        综合得分
        """
        scores = {
            'success': 1.0 if result.success else 0.0,
            'efficiency': AgentMetrics.normalized_steps(result, result.optimal_steps),
            'tool_accuracy': AgentMetrics.tool_efficiency(result),
            'cost': AgentMetrics.cost_efficiency(result),
            'safety': result.safety_score
        }
        
        total = sum(scores[k] * weights[k] for k in weights)
        return total / sum(weights.values())
```

---

## 五、构建自定义 Agent Harness

### 5.1 快速开始

```python
# agent_harness_example.py

from agent_harness import Harness, Task, Environment, Evaluator

# 1. 定义任务
class MyTask(Task):
    def __init__(self):
        super().__init__(
            task_id="data_analysis_001",
            description="分析 CSV 文件并生成报告",
            initial_state={
                "files": ["/data/sales.csv"],
                "tools": ["python", "pandas", "matplotlib"]
            },
            success_criteria={
                "file_exists": "/output/report.pdf",
                "contains_charts": True,
                "analysis_correct": True
            }
        )
    
    def check_success(self, environment: Environment) -> bool:
        """检查任务是否成功完成"""
        # 检查输出文件
        if not environment.file_exists("/output/report.pdf"):
            return False
        
        # 检查内容
        content = environment.read_file("/output/analysis.json")
        return self.verify_analysis(content)

# 2. 配置环境
env = Environment(
    type="jupyter",
    docker_image="jupyter/datascience-notebook:latest",
    resources={"cpu": 2, "memory": "4GB"}
)

# 3. 创建 Harness
harness = Harness(
    tasks=[MyTask()],
    environment=env,
    max_steps=50,
    timeout=600  # 10分钟
)

# 4. 运行评估
results = harness.evaluate(agent=my_agent)

# 5. 输出报告
print(results.summary())
print(results.detailed_report())
```

### 5.2 与现有框架集成

```python
# 与 LangChain 集成
from langchain.agents import AgentExecutor
from agent_harness.adapters import LangChainAdapter

agent = AgentExecutor(...)
harness_adapter = LangChainAdapter(agent)

results = harness.evaluate(agent=harness_adapter)

# 与 AutoGen 集成
from autogen import ConversableAgent
from agent_harness.adapters import AutoGenAdapter

agent = ConversableAgent(...)
harness_adapter = AutoGenAdapter(agent)

results = harness.evaluate(agent=harness_adapter)
```

---

## 六、生产环境评估实践

### 6.1 CI/CD 集成

```yaml
# .github/workflows/agent-eval.yml
name: Agent Evaluation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Environment
        run: |
          pip install agent-harness
          docker pull agent-harness/ubuntu-env
      
      - name: Run Evaluation Suite
        run: |
          agent-harness evaluate \
            --agent-config agent.yaml \
            --test-suite production \
            --output results.json
      
      - name: Check Regression
        run: |
          agent-harness compare \
            --baseline baseline.json \
            --current results.json \
            --threshold 0.05  # 5% 回归阈值
      
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: eval-results
          path: results.json
```

### 6.2 A/B 测试框架

```python
class AgentABTest:
    """
    Agent A/B 测试框架
    """
    
    def __init__(self, agent_a, agent_b):
        self.agent_a = agent_a  # 对照组
        self.agent_b = agent_b  # 实验组
        self.harness = Harness()
    
    def run_test(self, tasks, split=0.5):
        """运行 A/B 测试"""
        import random
        
        results_a = []
        results_b = []
        
        for task in tasks:
            if random.random() < split:
                result = self.harness.evaluate(task, self.agent_a)
                results_a.append(result)
            else:
                result = self.harness.evaluate(task, self.agent_b)
                results_b.append(result)
        
        # 统计显著性检验
        from scipy import stats
        
        success_a = [r.success for r in results_a]
        success_b = [r.success for r in results_b]
        
        t_stat, p_value = stats.ttest_ind(success_a, success_b)
        
        return {
            "agent_a_success_rate": sum(success_a) / len(success_a),
            "agent_b_success_rate": sum(success_b) / len(success_b),
            "p_value": p_value,
            "significant": p_value < 0.05
        }
```

---

## 七、2026 评估技术趋势

### 7.1 趋势预测

| 趋势 | 描述 | 预计时间 |
|------|------|----------|
| **LLM-as-Judge 标准化** | 使用 LLM 自动评估 Agent | 2026 |
| **多 Agent 评估** | 评估 Agent 协作能力 | 2026-2027 |
| **持续评估** | 在线监控 Agent 性能 | 2026 |
| **安全性红队** | 专门评估 Agent 安全风险 | 2026 |
| **人类偏好对齐** | 基于人类反馈的评估 | 2026-2027 |

### 7.2 LLM-as-Judge

```python
class LLMJudge:
    """
    使用 LLM 作为评估裁判
    """
    
    def __init__(self, model="gpt-4.5"):
        self.llm = OpenAI(model=model)
    
    def evaluate_response(self, task: str, agent_response: str, ground_truth: str) -> dict:
        """使用 LLM 评估 Agent 回答"""
        
        prompt = f"""
        任务: {task}
        
        Agent 回答:
        {agent_response}
        
        参考答案:
        {ground_truth}
        
        请评估 Agent 回答的以下维度 (1-5分):
        1. 准确性: 信息是否正确
        2. 完整性: 是否回答了所有部分
        3. 有用性: 对用户是否有帮助
        4. 安全性: 是否包含有害内容
        
        以 JSON 格式输出评分和理由。
        """
        
        response = self.llm.generate(prompt)
        evaluation = json.loads(response)
        
        return evaluation
```

### 7.3 关键资源

| 资源 | 链接 | 说明 |
|------|------|------|
| **Agent Harness** | https://github.com/ai-harness/agent-harness | 开源评估框架 |
| **GAIA** | https://gaia-benchmark.github.io | 通用 AI 助手基准 |
| **OSWorld** | https://osworld.github.io | OS 操作基准 |
| **SWE-bench** | https://www.swebench.com | 代码修复基准 |
| **WebArena** | https://webarena.dev | Web Agent 基准 |
| **ToolBench** | https://github.com/OpenBMB/ToolBench | 工具使用基准 |

---

*Last updated: 2026-04-03 | Version: 2026 Edition*
