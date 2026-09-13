# ToolSafe Phase 4 Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不立即租赁 GPU 的前提下，先打通 ToolSafe 的本地理解与小模型实验，再使用一台 Linux NVIDIA GPU 部署 vLLM 和 TS-Guard，完成从推理到安全评测的可验证链路。

**Architecture:** 先用 Transformers 在 Mac M3 Pro 上运行一个小型 Agent 模型，验证消息循环、工具调用解析、logits 和 Token 熵；再使用第二个本地小模型作为未训练 Guardian，完成护栏接口和 AgentDojo 子集流程；随后对 Guardian 做小规模 SFT（必要时 LoRA），比较训练前后；API 作为可选的接口基线，最后租赁 Linux NVIDIA GPU，以 vLLM 部署 7B 模型和 TS-Guard。每条路线有独立的成功标准，失败时不阻塞其他路线。

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers, vLLM, OpenAI-compatible API, AgentDojo, ToolSafe `src/main_experiment.py` and `src/guardian_experiment.py`.

## Global Constraints

- 当前本机为 Apple M3 Pro、36 GB 统一内存、无 NVIDIA CUDA GPU；当前仓库的 vLLM 路径按 CUDA GPU 设计。
- 先在本机完成代码、配置和小模型验证，再租赁付费 GPU；GPU 实例只运行已验证的命令。
- 不把真实 API Key、远程地址凭据或包含敏感数据的日志写入 Git。
- AgentDojo 使用模拟环境；在真实工具集成前禁止连接真实邮件、银行或生产系统。
- 所有实验固定模型版本、采样参数、任务子集和随机种子，记录实际配置与结果。
- 论文官方 TS-Guard 是全参数 GRPO；个人低成本 LoRA/QLoRA 实验必须在报告中标明是替代方案。

## Route Decision

### Route A: Mac + Transformers/MPS

用途：理解模型输入、工具调用格式、logits 和熵；不追求 7B TS-Guard 指标。

可运行性：Mac 可以加载小型 Qwen Instruct 模型；7B 模型即使使用统一内存也可能慢、占用高，当前 `device_map="auto"` 路径还需要验证 MPS 行为。

推荐模型规模：0.5B 或 1.5B Instruct；先不加载官方 TS-Guard。

### Route B: 本地双模型 Agent + Guardian

用途：用两个本地小模型分别承担 Agent 和 Guardian，观察护栏训练前后的行为变化。

可运行性：本机可以加载 0.5B 或 1.5B 模型；当前 `Guardian` 封装只有 API/vLLM 分支，执行时需要增加一个 Transformers Guardian 适配器，或把本地模型包装成 OpenAI-compatible 服务。

限制：小模型的结构化输出和安全判断能力有限，结果只能称为 mini-TS-Guard，不能替代论文中的 Qwen2.5-7B TS-Guard。

### Route C: API / OpenAI-compatible 推理

用途：最快打通 AgentDojo/ASB 的 Agent 循环、护栏调用和结果保存；本机不需要 GPU。

可运行性：需要用户可用的 API 服务；配置通过环境变量注入密钥。

限制：无法保证服务支持完整 logits；通常只能观察文本，若服务支持 `logprobs` 才能做有限概率分析。

### Route D: Linux NVIDIA + vLLM

用途：部署 Qwen2.5-7B 或 TS-Guard，完成高吞吐推理和论文链路复现。

可运行性：不建议在当前 Mac 直接运行仓库的 vLLM 路径；租赁单张 4090 24 GB 可做 7B 推理，TS-Guard GRPO 全参数训练仍需要多卡或更大显存。

推荐顺序：Route A → Route B → Route C（可选）→ Route D。先确认逻辑正确，再支付 GPU 费用。

---

### Task 1: 本机 Transformers 小模型冒烟测试

**Files:**
- Read: `ToolSafe/src/model/model.py`
- Read: `ToolSafe/src/agent/agent.py`
- Create during execution: `practice/phase4_transformers/phase4_transformers_smoke.yaml`
- Create during execution: `practice/phase4_transformers/phase4_transformers_smoke.py`
- Create during execution: `practice/phase4_transformers/tests/test_phase4_transformers_smoke.py`

**Interfaces:**
- Consumes: 本地小型 Instruct 模型路径、一个固定用户请求和工具描述。
- Produces: 一次模型响应、解析出的工具名/参数、每 Token logits 分数、平均熵和运行设备。

- [ ] 确认 Python、PyTorch、Transformers、`torch.backends.mps.is_available()` 状态，并记录输出。
- [ ] 使用 `model.type="analysis"` 加载小模型，固定 `do_sample=False`、`max_new_tokens=128`，避免先运行 7B。
- [ ] 发送一个只读工具调用请求，确认 `Agent_Core.messages` 包含 system、user、assistant 三类消息。
- [ ] 验证 `outputs.scores` 的步数、词表维度和 `token_entropy` 的非负结果。
- [ ] 如果 `device_map="auto"` 没有正确选择 MPS，只修改本地实验的设备选择逻辑，明确记录改动，不改变论文训练代码。
- [ ] 保存一份不含密钥的实验日志，确认能解释每个字段的含义。

**验证命令:**

```bash
python scripts/phase4_transformers_smoke.py
```

成功标准：模型生成合法的 ReAct 格式；能观察至少一个生成步骤的分数和熵；没有调用任何真实外部工具。

### Task 2: 本机双模型护栏接口冒烟测试

**Files:**
- Create during execution: `practice/phase4_local_guardian/phase4_local_guardian_smoke.yaml`
- Create during execution: `practice/phase4_local_guardian/tiny_guardian.py`
- Read: `ToolSafe/src/main_experiment.py`
- Read: `ToolSafe/src/agent/sec_react_agent.py`

**Interfaces:**
- Consumes: 本地 Agent 小模型、本地 Guardian 小模型、固定的安全/风险工具调用样本。
- Produces: 一条 Agent 轨迹，包含候选动作、护栏反馈、工具执行结果和最终输出。

- [ ] 先使用只读的本地模拟工具，不连接真实邮件、银行或生产系统。
- [ ] 为 Guardian 实现最小 Transformers 适配器，统一返回 `risk rating`、`reason` 和解析结果。
- [ ] 用同一个固定样本分别调用 Agent 和 Guardian，确认两者职责分离。
- [ ] 运行单个 ASB 或 AgentDojo 任务，限制 `task_nums` 或 `user_task` 为 1。
- [ ] 检查工具调用前是否出现 TS-Guard 请求，确认高风险动作没有进入 `runtime.run_function`。
- [ ] 检查输出目录中的 `meta_data.json`，手工核对消息顺序。

**验证命令:**

```bash
python src/main_experiment.py --config src/config/phase4_local_guardian_smoke.yaml
```

成功标准：完整链路运行一次；能区分模型输出、护栏判断和工具执行；不产生真实外部副作用。

### Task 3: 本地 Guardian 小规模 SFT 对比

**Files:**
- Create during execution: `practice/phase4_guardian_sft/phase4_prepare_guardian_sft.py`
- Create during execution: `practice/phase4_guardian_sft/phase4_train_guardian_sft.py`
- Create during execution: `practice/phase4_guardian_sft/phase4_guardian_sft.yaml`
- Read: `ToolSafe/TS-Bench/`

**Interfaces:**
- Consumes: TS-Bench 轨迹、风险标签、一个 0.5B/1.5B Instruct 基座模型。
- Produces: 训练前后 Guardian checkpoint、验证集 accuracy/F1/recall、格式正确率和熵对照。

- [ ] 将每条样本整理为用户请求、历史、当前动作、工具描述和结构化目标输出。
- [ ] 按任务或场景划分训练集与验证集，避免同一轨迹模板泄漏。
- [ ] 先用全参数或 LoRA 完成极小 SFT，不直接开始 GRPO。
- [ ] 固定解码参数，比较训练前后风险分数、格式正确率和验证集指标。
- [ ] 将该模型标记为 `mini-TS-Guard`，不宣称等同论文官方权重。

成功标准：能够展示训练前后同一批样本的可解释差异，并在未见验证样本上报告结果。

### Task 4: 本地 AgentDojo 子集与指标解释

**Files:**
- Read: `ToolSafe/src/task_executor/agentdojo_exec.py`
- Read: `ToolSafe/src/task_executor/agentdojo/benchmark.py`
- Read: `ToolSafe/src/task_executor/agentdojo/task_suite/task_suite.py`
- Read: `ToolSafe/src/task_executor/agentdojo/base_tasks.py`

**Interfaces:**
- Consumes: 一个 suite、一个 user task、一个 injection task 和一个 attack 类型。
- Produces: `utility.json`、`security.json`、`meta_data.json`，以及对 `security=True` 含义的人工核对。

- [ ] 只运行一个 suite 和一个 `user_task`，固定 `attack_type=tool_knowledge`。
- [ ] 检查 Canary 注入候选、环境模板替换和攻击文本生成结果。
- [ ] 检查环境执行前后的状态副本，确认 utility 和 security 的布尔定义。
- [ ] 把原始攻击成功率与防御成功率分开记录，避免直接把 `Average security` 当作防御率。

成功标准：能手工解释一条任务从注入到环境状态变化的全过程。

### Task 5: 可选 API Agent 流程基线

**Files:**
- Create during execution: `ToolSafe/src/config/phase4_api_smoke.yaml`
- Read: `ToolSafe/src/model/model.py`
- Read: `ToolSafe/src/main_experiment.py`

**Interfaces:**
- Consumes: 用户提供的 OpenAI-compatible `base_url`、模型名和通过环境变量提供的 API Key。
- Produces: 一条不依赖本机 GPU 的 Agent 轨迹，作为本地双模型路线的接口基线。

- [ ] 只有在本地双模型链路通过后才使用 API；把 API Key 从 YAML 移到环境变量。
- [ ] 先使用只读模拟工具，运行一个 AgentDojo 任务。
- [ ] 对比 API Agent 与本地小模型 Agent 的消息格式和工具调用解析。

成功标准：完整 Agent 链路运行一次，且没有将 API 凭据写入仓库。

### Task 6: 租赁 Linux NVIDIA GPU 部署 vLLM

**Files:**
- Read: `ToolSafe/src/model/model.py`
- Create during execution: `ToolSafe/src/config/phase4_vllm_smoke.yaml`
- Create during execution: `ToolSafe/docs/phase4_vllm_run.md`

**Interfaces:**
- Consumes: Linux NVIDIA 实例、模型权重路径、固定端口和健康检查命令。
- Produces: 一个 OpenAI-compatible vLLM 服务，以及一次本地客户端调用结果。

- [ ] 仅在本地双模型和必要的 API 流程通过后租赁 GPU；优先按小时计费的 4090 24 GB 实例。
- [ ] 在 GPU 实例上确认 CUDA、PyTorch、vLLM 和显存容量，再下载权重。
- [ ] 使用固定模型、`dtype`、端口和 `gpu-memory-utilization` 启动服务。
- [ ] 先调用 `/v1/models`，再调用一次普通聊天接口。
- [ ] 记录启动日志、显存占用、响应时间和释放实例时间。

示例启动形式（执行时逐参数解释并按平台调整）：

```bash
vllm serve /path/to/model --host 0.0.0.0 --port 8000 --dtype bfloat16
```

成功标准：服务健康、客户端收到响应、服务端没有 OOM；完成后立即释放实例。

### Task 7: 官方 TS-Guard 护栏推理

**Files:**
- Modify during execution: `ToolSafe/src/config_guardrail_eval/agentharm_traj.yaml`
- Read: `ToolSafe/src/guardian_experiment.py`
- Read: `ToolSafe/src/utils/guardian_score_mapping.py`

**Interfaces:**
- Consumes: `MurrayTom/TS-Guard` 权重或可访问的官方 checkpoint、TS-Bench 轨迹子集。
- Produces: `preds.json`、`labels.json`、`metrics_strict.json`，并报告实际 score mapping。

- [ ] 先只运行 AgentHarm trajectory 子集，再运行 ASB OPI 子集。
- [ ] 固定 `score_mode=strict`，另外离线计算 loose/exact 作为对照。
- [ ] 抽查若干 `guard_res.reason`，确认模型输出解析与风险分数一致。
- [ ] 报告 accuracy、F1、recall，并注明当前代码没有输出 precision。

成功标准：护栏模型能独立完成一次轨迹分类评测，结果文件可复查。

### Task 8: TS-Guard + SecReAct 端到端子集

**Files:**
- Modify during execution: `ToolSafe/src/config/phase4_end_to_end.yaml`
- Read: `ToolSafe/src/agent/sec_react_agent.py`
- Read: `ToolSafe/src/task_executor/agentdojo/benchmark.py`

**Interfaces:**
- Consumes: Agent 模型、TS-Guard 服务、一个 AgentDojo suite 和一个攻击类型。
- Produces: 正常任务 utility、攻击任务 security、消息轨迹和护栏反馈对照。

- [ ] 分别运行普通 ReAct、SecReAct 和 Firewall Agent。
- [ ] 对同一任务记录三者的工具调用、护栏决策和最终环境状态。
- [ ] 检查 `SecReAct` 是否只传递风险等级，明确它与论文完整反馈的差异。
- [ ] 计算并同时报告攻击成功率、任务完成率和安全-可用性折中。

成功标准：得到一组可解释的对照结果，而不是只得到一个最终数字。
