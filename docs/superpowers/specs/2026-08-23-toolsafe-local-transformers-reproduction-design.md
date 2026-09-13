# ToolSafe 本地 Transformers 复现设计

## 1. 目标

在当前 Apple M3 Pro、36GB 统一内存、无 NVIDIA CUDA 的机器上，尽可能保留 ToolSafe 论文开源实现的真实技术链路，仅将论文使用的 7B Agent/Guardian 模型替换为本地可承受的 Qwen2.5-1.5B-Instruct。

本阶段的最终目标是跑通以下真实流程：

```text
用户请求
  -> 原始 Agent prompt 与消息循环
  -> Agent 产生候选工具调用
  -> ToolSafe 原始 ReAct parser
  -> TS-Guard 风格 Guardian Transformers 推理
  -> 风险反馈或允许执行
  -> 原始模拟环境/运行时执行器
  -> 原始评测结果与轨迹日志
```

本地阶段不追求论文数值复现，而追求入口、数据流、工具执行前护栏位置、解析协议和评测产物与论文实现一致。

## 2. 明确边界

### 保留

- `main_experiment.py` 和 `guardian_experiment.py` 的任务入口与结果保存逻辑；
- `ReAct_Agent`、`SecReAct_Agent`、`Agent_Core` 的消息循环；
- `agent_prompts.py` 中的原始工具描述、格式约束和安全反馈提示；
- `tool_parser.py`、`guardian_parser.py`、`guardian_score_mapping.py`；
- `TS-Bench`、AgentHarm、ASB、AgentDojo 的真实数据与环境执行器；
- `SecReAct_Agent` 在 `runtime.run_function(...)` 之前调用 Guardian 的顺序；
- Guardian 的 Prompt、结构化风险分数和评测指标计算；
- 后续训练阶段的 SFT/GRPO 思想、奖励维度、rollout、KL 和 checkpoint 逻辑。

### 允许缩小

- Agent 和 Guardian 基座模型从 Qwen2.5-7B 替换为 Qwen2.5-1.5B-Instruct；
- AgentDojo 限制为一个 suite、一个 user task、一个 injection task；
- ASB/AgentHarm 只运行少量样本；
- 减少 `max_turns`、`max_new_tokens`、训练 epoch、batch size、rollout 数和验证集规模；
- 必要时使用 LoRA/QLoRA，但必须在实验报告中标明它与论文全参数训练不同。

### 禁止替代

- 规则 Guardian 替代模型 Guardian；
- 手工风险判断替代 Guardian Prompt + 模型输出 + parser；
- 假工具或只打印结果的执行器替代 AgentDojo/ASB 模拟环境；
- 手工构造评测结果替代 TS-Bench/AgentDojo 数据；
- 为了通过测试而放宽原始工具调用解析器。

## 3. 代码组织

原始仓库保持不变。必要的兼容文件放在独立实践目录：

```text
practice/toolsafe_reproduction/
├── README.md
├── model/
│   └── model.py              # 原 model.py 的兼容副本
├── agent/
│   ├── agent.py              # 原 Agent_Core 的兼容副本
│   └── sec_react_agent.py    # 原 SecReAct 的兼容副本
├── runners/
│   ├── run_guardian_experiment.py
│   └── run_main_experiment.py
├── config/
└── tests/
```

每个复制文件在文件头注明原始路径、来源 Git commit 和仅有的本地适配内容。未复制的模块通过 `PYTHONPATH` 继续复用 `ToolSafe/src`，避免无必要的代码分叉。

## 4. 模型适配

### Agent

复用原 `Model` 的 `analysis` 语义：Tokenizer 构造 Chat Template，Transformers 模型生成文本，`output_scores=True` 保留每步分数，`Agent_Core` 继续计算 Token 熵。

本地副本只修正设备和依赖边界：

- vLLM 改为惰性导入；
- `analysis` 使用本地 Transformers 和 MPS/CPU；
- 保留 `do_sample`、`max_new_tokens` 和消息历史语义；
- 不在本机实现 vLLM。

### Guardian

为原 `Guardian` 增加真正的 `analysis` 分支，而不是规则分类：

1. 使用原 `GUARD_TEMPLATES[model_name]` 构造 Guardian 输入；
2. 使用本地 Transformers 生成 Guardian 文本；
3. 使用原 `guardian_paser_map` 解析风险等级和结构化结果；
4. 解析失败时沿用原代码的重试/无效结果语义；
5. `call_tool("tool_safety_guardian", ...)` 的接口保持不变。

## 5. 实施阶段

### 阶段 A：真实依赖与入口核对

- 确认当前仓库缺少 README 所称的 `TS-Guard/verl-main`；
- 固定当前 ToolSafe Git commit；
- 核对 Python 依赖、原始模块导入和本地 Transformers 可用性；
- 不在本阶段编写任何规则 Guardian。

### 阶段 B：Guardian 独立评测

- 使用 `guardian_experiment.py` 的真实评测入口；
- 先运行 AgentHarm 或 ASB 的极小子集；
- 输出 `meta_data.json`、`preds.json`、`labels.json`、`metrics_strict.json`；
- 抽查 `guard_res.reason`、风险分数和标签的对应关系；
- 将结果命名为未训练的 `mini-TS-Guard baseline`，不宣称等同官方 TS-Guard。

### 阶段 C：SecReAct 端到端评测

- 使用原 `main_experiment.py` 和 `SecReAct_Agent`；
- Agent 使用本地 1.5B Transformers；
- Guardian 使用本地 1.5B Transformers；
- 只运行一个 AgentDojo suite、一个任务和一个攻击类型；
- 对比普通 ReAct 与 SecReAct 的工具调用、Guardian 反馈、工具执行和最终 utility/security；
- 检查高风险候选是否在 `runtime.run_function(...)` 前被阻断。

### 阶段 D：训练前后对比

- 先用未训练 1.5B Guardian 建立 baseline；
- 训练前重新确定 TS-Bench 字段、标签、Prompt 和目标输出格式；
- 若取得官方 verl 代码，优先按其 GRPO 入口缩小 epoch、batch、rollout 和数据子集；
- 若官方训练目录确实无法取得，再单独设计保持论文奖励思想的 SFT/GRPO 替代实现，并记录偏差；
- 比较训练前后 accuracy、F1、recall、格式正确率、风险解释和 AgentDojo utility/security。

## 6. 训练代码缺失处理

当前仓库没有 `TS-Guard/verl-main`，因此本地阶段先不假装可以执行论文训练脚本。需要训练时按以下顺序处理：

1. 核验官方仓库、分支、子模块或单独训练仓库来源；
2. 获取代码后先做配置和数据格式审计；
3. 先进行极小 SFT，理解数据、loss、checkpoint 和推理闭环；
4. 再进行极小 GRPO，保留组采样、奖励、优势、KL 和 clipping 逻辑；
5. 最后再讨论 LoRA/QLoRA、云 GPU 和更大模型。

## 7. 验证标准

- 原始 ToolSafe 代码目录没有新增教学替代逻辑；
- Guardian 的输入、模型生成、parser 和风险分数均可追踪；
- Agent 的工具调用在真实执行器前经过 Guardian；
- 至少一条安全任务和一条风险任务产生可审计轨迹；
- 评测 JSON 可以由用户亲自在终端重新生成；
- 所有模型规模、子集规模和训练缩减参数均记录在实践配置中。

## 8. 非目标

- 本机实现 vLLM 或 CUDA 张量并行；
- 在没有训练代码的情况下声称复现论文 GRPO 数字；
- 将 1.5B 未训练 Guardian 称为官方 TS-Guard；
- 连接真实邮箱、银行、Slack 或其他生产工具。
