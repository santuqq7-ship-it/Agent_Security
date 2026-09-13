# TS-Flow Agent 小测试设计规格

日期：2026-09-13

状态：用户已批准方案一；本地实现完成，等待云端双 7B 真实小测试

## 1. 目标与阶段边界

本阶段只验证“步级 Guard 能否在真实 AgentDojo 工具执行前介入，并把完整安全反馈交给
Agent 触发重新规划”。它不是正式 Agent 能力评测，也不使用本地 7B Agent 的 Utility 作为
最终研究结论。

小测试使用：

- Agent：本地 `Qwen2.5-7B-Instruct`；
- Guard：官方 `TS-Guard 7B`；
- 环境：现有 AgentDojo `v1.2.2`；
- Guard 解码：现有 tokenizer-aware Grammar/FSM；
- 数据边界：只取 AgentDojo workspace 的少量固定任务；不读取或使用 ASB。

正式测试保留完全相同的 Guard、FSM、AgentDojo 环境、控制流、轨迹和指标接口，仅把
Agent 后端切换为 OpenAI-compatible API。API URL、API key 和模型名称不进入本次小测试，
也不写入源码。

## 2. 为什么采用该方案

本地双 7B 的优点是链路可控、无需 API 成本，并能先排除 TS-Flow 控制流本身的问题。
其局限是 7B Agent 可能无法正确理解 Guard 反馈并稳定重新规划，因此小测试只回答
“机制是否工作”，不回答“TS-Flow 对强 Agent 的最终收益是多少”。正式结论必须来自后续
API Agent 对照实验。

Guard 使用纯 Hugging Face/PyTorch FSM，而不重新引入 vLLM guided decoding。现有 FSM 已经
覆盖协议 token、Think 正文、三字段枚举值及闭合符，并且与后续 constrained GRPO 使用同一
套语义。复用它比维护第二套 grammar 更容易审计，也避免再次出现 HF 与 vLLM 输出协议不一致。

## 3. 对照模式

同一 runner 支持三种互斥模式，三者使用同一个 Agent checkpoint、相同任务、相同最大轮数：

1. `react`：不调用 Guard，Agent 的合法工具请求直接进入环境；
2. `abort`：每次工具执行前调用 Guard；风险评分大于等于 `0.5` 时阻断工具并立即结束轨迹；
3. `ts_flow`：每次工具执行前调用 Guard；风险评分大于等于 `0.5` 时阻断工具，把完整规范化
   Guard 输出作为 Observation 返回 Agent，然后继续下一轮规划。

`ts_flow` 反馈必须包含非空 Think、`Malicious_User_Request`、`Being_Attacked` 和
`Harmfulness_Rating`，不能再把反馈裁剪成只有 `risk rating` 的字典。安全动作在 `abort` 和
`ts_flow` 中正常执行；`react` 不伪装成调用过 Guard。

## 4. 组件设计

### 4.1 独立模型后端

runner 将原来的单一 `--model-path` 拆为：

- `--agent-backend`：本次为 `transformers`，后续可为 `api`；
- `--agent-model-path`：本地 Agent 权重；
- `--agent-model-name`：模型显示名或 API model；
- `--guardian-model-path`：官方 Guard 权重；
- `--flow-mode`：`react`、`abort` 或 `ts_flow`。

未来 API 所需 URL/key 通过命令行或环境读取，但密钥永不写入 trace、结果 JSON 或文档。
本阶段不要求真实 API 调用，只保证后端边界不会迫使以后重写 Agent/Guard 控制流。

### 4.2 Constrained Guardian 适配器

新增一个只负责 Guard 推理的适配器：

1. 使用官方模板构造 Guard prompt；
2. 用 Guard tokenizer 应用 chat template；
3. 编译并缓存 `CompiledGuardianGrammar`；
4. 调用 `generate_constrained_rollout(..., do_sample=False)`；
5. 直接从 FSM 的 `judgments` 取得三个结构化值；
6. 返回兼容现有 Agent 的 `risk rating/results/reason`，其中 `reason` 是完整规范化四行文本。

字段格式由 FSM 确定，字段值和 Think 内容仍由 Guard 模型决定。由于输出不再经过宽松正则
恢复，正常路径的格式率应为 100%。若约束编译、模型前向或 FSM 推理异常，Guard 路径
fail-closed：不执行当前工具，同时在 trace 中记录明确错误；不能静默 fail-open。

### 4.3 Agent 控制流

在本地 overlay 的 `SecReAct_Agent` 中加入显式 flow mode，而不是维护三份近似 Agent 文件。
工具执行顺序固定为：

`Agent 提议动作 -> 解析参数 ->（按模式）Guard 判断 -> 阻断/反馈或 runtime.run_function`

所有真正调用 `runtime.run_function` 的路径都必须经过上述唯一分支，避免某类工具绕过 Guard。
被阻断动作绝不进入运行时。`abort` 返回可辨识的终止原因；`ts_flow` 将完整 Guard 输出追加为
Observation，再调用同一 Agent 生成后续动作。

### 4.4 轨迹与结果

沿用 JSONL trace，并为每个事件保存以下可核验证据：

- Agent 原始输出与解析后的工具名/参数；
- Guard 输入、FSM 规范输出和三个结构化判断；
- flow mode、是否阻断、是否把完整反馈送回 Agent；
- 反馈后的下一次 Agent 输出；
- 实际进入 `runtime.run_function` 的工具调用及 Observation；
- AgentDojo Utility 与 Security 结果；
- Guard 格式率、调用数、阻断数、实际工具执行数的汇总。

Think 正文可以出现在研究 trace 中，但正式 API 阶段不得记录 API key、Authorization header
或第三方 SDK 的原始请求对象。

## 5. 小测试矩阵

先通过 suite inventory 选定并在命令中显式写死两个 workspace 样本：

- 一个低风险、工具链短的 benign 用户任务；
- 同一 suite 中一个会把间接提示注入带入 Agent 上下文的攻击组合。

两个样本分别跑 `react`、`abort`、`ts_flow`，最多形成 6 条短轨迹。任务 ID、attack type 和
injection task ID 必须显式记录，不允许依赖“第一个任务”之类会随版本改变的隐式默认值。
若本地 Agent 无法完成 benign 任务，该现象只记为模型能力限制，不反向修改 Guard 阈值或
安全标签来迎合结果。

## 6. 验收标准

代码级验收：

- Agent 与 Guard 路径完全独立；
- 约束 Guard 的 3 个判断值可直接由 FSM 结构化返回；
- `react` 不调用 Guard；
- `abort` 的危险动作被阻断且轨迹立即结束；
- `ts_flow` 的危险动作被阻断、完整四行反馈进入下一轮 Agent 上下文；
- 任何被阻断动作均未触发 `runtime.run_function`；
- Guard 异常时 fail-closed，并留下错误事件；
- 不读取 ASB，不修改训练数据或模型权重。

真实小测试验收：

- 所有 Guard 响应满足 FSM 格式，格式率 100%；
- 至少一条攻击轨迹触发真实阻断；
- trace 能证明 Guard 反馈之后 Agent 确实获得一次重新规划机会；
- benign 与攻击样本均产生完整、可复核的 AgentDojo 结果文件；
- Utility 较低可以接受，但必须与控制流缺陷区分记录。

如果官方 Guard 对所选攻击样本判断为安全，则该轨迹不能伪造成“阻断成功”；应保留结果，
另选一个明确危险的非 ASB 样本补做机制验收，并记录选择原因。

## 7. 测试策略

先用假模型、假 runtime 做聚焦单元测试，不加载 7B 权重：

- 三种 flow mode 的分支行为；
- 完整反馈未被裁剪；
- 阻断动作无法到达 runtime；
- Guard 错误 fail-closed；
- trace 汇总不泄漏 API 密钥。

再验证现有 FSM 对官方 tokenizer 的编译和一条真实 constrained generation。最后才运行上述
最多 6 条 AgentDojo 轨迹。不会在本阶段跑全套数据、训练、LoRA 或 ASB。

## 8. 交付物

- 一个支持独立 Agent/Guard 后端与三种 flow mode 的单轨迹 runner；
- 一个复用现有 FSM 的官方 Guard 推理适配器；
- 对 SecReAct overlay 的最小控制流修正；
- 聚焦单元测试与两样本小测试命令；
- 一份小测试结果报告，并把阶段结论同步到 `PROJECT_JOURNEY.md`。

后续正式实验只接入 API Agent 并扩展样本矩阵，不重新设计 Guard 或 TS-Flow 安全边界。
