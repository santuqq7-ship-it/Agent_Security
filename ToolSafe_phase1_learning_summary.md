# ToolSafe 阶段一学习总结清单

> 更新时间：2026-08-08  
> 当前范围：论文第一遍速览、系统框架、TS-Guard/TS-Flow 方法、仓库可复现性核验、GRPO 基础。  
> 主要材料：[ToolSafe 论文](./2026.findings-acl.1850.pdf) 与 [官方代码仓库](./ToolSafe/)。

---

## 0. 当前学习进度

- [x] 明确项目的可复现性、算力风险和校招价值。
- [x] 完成论文第一遍速览：Abstract、Introduction、主要图表、Conclusion。
- [x] 理解 TS-Bench、TS-Guard、TS-Flow 的关系。
- [x] 理解 step-level guardrail 为什么必须在工具执行前介入。
- [x] 核验本地 Git 仓库、官方 GitHub 和 Hugging Face 中的 TS-Guard 资源。
- [x] 初步理解 GRPO、rollout、reward、advantage、value network 和 KL loss。
- [ ] 完成实验指标、baseline 和消融实验的逐表精读。
- [ ] 完成阶段一的 3 分钟论文电梯演讲和面试演练。

## 1. 项目可复现性与学习目标

- [x] ToolSafe 适合作为校招项目，覆盖 Agent、工具调用安全、Prompt Injection、Guardrail、SFT、GRPO 和 vLLM。
- [x] 不追求完整复现论文的每一个数字，优先理解系统逻辑并跑通代表性子集。
- [x] 论文训练 TS-Guard 使用 Qwen2.5-7B-Instruct 和 GRPO，完整训练成本较高。
- [x] 论文实验环境为 8 张 NVIDIA 96GB H20，个人低成本环境不应直接照搬。
- [x] 实践策略：本地完成论文与源码分析；使用官方权重做推理和评测；用小模型、小数据完成迷你 GRPO。

核心原则：**不要只复制粘贴，要理解每一行配置的含义。算力不够就跑子集。**

## 2. 论文的一句话总结

> ToolSafe 用 TS-Bench 定义步级工具调用安全检测任务，用 GRPO 训练能够解释风险来源的 TS-Guard，再通过 TS-Flow 将护栏反馈注入 Agent 的推理过程，在减少危险工具调用的同时尽量保留正常任务能力。

## 3. 论文要解决的问题

- [x] Agent 可以调用外部工具并对真实环境产生副作用，安全风险不再局限于文本输出。
- [x] 最终输出检测通常发生得太晚：危险工具调用可能已经执行，事后文本拒绝无法撤销真实副作用。
- [x] 静态输入/输出护栏缺少完整交互历史，难以判断当前动作是否偏离用户目标。
- [x] 完整轨迹级检测属于事后判断，无法在每个候选工具调用执行前及时干预。

论文的三个研究问题：

1. Q1：哪些步级信号能够在执行前表明工具调用可能不安全？
2. Q2：如何训练一个可泛化的步级工具调用护栏？
3. Q3：如何集成护栏，在提升安全性的同时不显著损害良性任务完成率？

严谨表述：**单独做最终输出检测不充分，而不是最终输出检测完全没有价值。**

## 4. ToolSafe 关注的四类风险模式

论文从两个维度分析风险：

- 触发原因：恶意用户请求，或第三方 Prompt Injection。
- 风险表现：调用有害工具，或使用正常工具但传入危险参数。

四个缩写：

- **MUR（Malicious User Request）**：恶意用户请求直接诱导危险动作。
- **PI（Prompt Injection）**：第三方内容注入指令，使 Agent 偏离用户目标。
- **HT（Harmful Tools）**：工具名称、描述或能力本身具有明显危害性。
- **BTRA（Benign Tools with Risky Arguments）**：工具本身正常，但调用参数或调用目的危险。

## 5. ToolSafe 三个核心组件

### 5.1 TS-Bench：数据与评测标准

- [x] TS-Bench 是 benchmark，不是运行时护栏。
- [x] 每个样本包含可用工具、用户请求、历史交互、候选动作和安全标签。
- [x] 安全标签为 `safe / controversial / unsafe`，工程表示可映射为 `0.0 / 0.5 / 1.0`。
- [x] 数据来源包括 AgentAlign、AgentHarm、ASB 和 AgentDojo。
- [x] 训练集主要使用 AgentAlign 和 ASB 的部分领域，评测集使用 AgentHarm、AgentDojo 和 ASB 的其余领域。
- [x] 论文通过 embedding 相似度分析降低训练集与评测集泄漏风险。

形式化输入：

```text
s = (T, u, H_t, a_t, y_t)

T   = 可用工具及描述
u   = 用户原始请求
H_t = 第 t 步之前的交互历史
a_t = 当前候选工具调用
y_t = 当前动作的安全标签
```

### 5.2 TS-Guard：安全判断模型

- [x] TS-Guard 的初始底座是 Qwen2.5-7B-Instruct。
- [x] 经过 TS-Bench 数据和 GRPO 训练后，得到新的 TS-Guard checkpoint。
- [x] TS-Guard 与原始 Qwen 架构相同，但权重已经改变，不能仅理解为模型改名。
- [x] TS-Guard 在一次推理中输出分析和三个结构化判断。

```text
输入：用户请求 + 历史交互 + 当前候选动作 + 工具描述

输出：
1. <Think> 简短风险分析
2. <Malicious_User_Request> 用户请求是否恶意
3. <Being_Attacked> 当前动作是否与第三方攻击相关
4. <Harmfulness_Rating> 当前动作安全等级
```

多任务设计的意义：先判断风险来源，再判断动作结果，减少仅凭关键词或“出现注入内容”就过度拦截的情况。

### 5.3 TS-Flow：运行时反馈框架

- [x] TS-Flow 不是模型，而是 Agent 与 Guard 的运行时编排方式。
- [x] 每个候选工具调用在真实执行前都要经过 TS-Guard。
- [x] 安全动作正常执行并返回真实 observation。
- [x] 危险动作被阻止，TS-Guard 的反馈替代真实工具 observation，交给 Agent 重新推理。
- [x] TS-Flow 不直接替 Agent 生成最终替代方案，最终重新规划仍由 Agent 完成。

```text
Agent 生成候选动作
        |
        v
TS-Guard 执行前检查
        |
        +-- 安全 --> 执行工具 --> 返回真实 observation
        |
        +-- 危险 --> 阻止工具 --> 返回 guard feedback --> Agent 重新规划
```

与 detect-and-abort 的区别：

- LlamaFirewall 式策略检测到风险后可能终止整个任务，安全性提高但 Utility 下降。
- TS-Flow 阻止当前危险动作并继续任务，目标是同时保留安全性和可用性。

## 6. Step-level Guardrail 概念清单

- **是什么**：每次工具调用执行前的安全检查点。
- **为什么需要**：工具调用可能产生不可逆副作用，最终结果检测太晚。
- **原理直觉**：在动作进入真实环境之前拦截，而不是等完整轨迹结束后复盘。
- **代码位置**：[sec_react_agent.py](./ToolSafe/src/agent/sec_react_agent.py) 在提取候选动作后调用 guard，再决定是否执行真实工具。
- **输出提示词**：[agent_prompts.py](./ToolSafe/src/agent/agent_prompts.py) 定义 TS-Guard 的多任务输入和输出格式。
- **面试问题**：为什么 step-level 比 trajectory-level 更适合工具调用安全？
- **面试答案**：前者能在副作用发生前逐步干预，后者通常只能做事后检测。

## 7. Agent、Guard 和 Judge 的模型角色

```text
用户请求
  |
  v
Agent Backbone：GPT-4o 或 Qwen2.5-14B-Instruct
  | 生成候选工具调用
  v
Guard：TS-Guard（GRPO 训练后的 Qwen2.5-7B-Instruct）
  | 安全判断与反馈
  v
工具执行或阻止
  |
  v
Judge / Rule Evaluator：实验结束后计算安全和任务指标
```

- [x] Agent 负责推理、规划和工具调用。
- [x] Guard 负责执行前安全判断，不负责完成整个用户任务。
- [x] Judge 负责实验评测，不应参与运行时防御决策。
- [x] 论文训练的是 TS-Guard，并未联合训练 Agent 与 Guard。
- [x] Agent 与 Guard 独立训练可能导致二者的判断和推理不一致，这是论文承认的限制。

## 8. 主要实验结论与限制

- [x] 摘要报告 TS-Flow 平均减少约 65% 的有害工具调用。
- [x] Prompt Injection 场景下，良性任务完成率平均提高约 10%。
- [x] TS-Guard 在 strict mode 下跨 AgentHarm、ASB、AgentDojo 取得较强综合表现。
- [x] 完整 TS-Guard 反馈优于只给安全等级，说明“为什么危险”对 Agent 重规划有帮助。
- [x] TS-Flow 会增加输入上下文长度和额外护栏推理延迟，并非零成本方案。

论文明确的两个限制：

1. Guard feedback 只是追加到 Agent 上下文，Agent 可能无法完全吸收或遵循。
2. Agent 与 Guard 独立训练，可能出现安全判断与推理目标不一致。

## 9. 本地仓库与 Hugging Face 核验结果

- [x] 当前本地工作树没有 `ToolSafe/TS-Guard/verl-main`。
- [x] 当前官方 GitHub 主分支同样没有该目录。
- [x] 本地 Git 历史提交 `c8e12b1` 中曾存在完整训练目录，约 1211 个文件。
- [x] 提交 `fafcd8c` 明确删除了整个 `TS-Guard/verl-main`。
- [x] 历史训练目录包含 `run_TSGuard_train.sh`、verl、reward 函数、训练数据和处理脚本。
- [x] Hugging Face `MurrayTom/TS-Guard` 提供约 15.2 GB BF16 权重、4 个 safetensors 分片、配置、Tokenizer 和 chat template。
- [x] Hugging Face 仓库不提供 verl 或 GRPO 训练源码。
- [x] 当前 `config_guardrail_eval/*.yaml` 仍指向作者内部 `/mnt/shared-storage-user/...` checkpoint，不能直接复用。
- [x] 当前 `sec_react_agent.py` 在高风险时只保留 `risk rating`，与论文强调的“完整反馈”可能存在实现差异。

后续可复现策略：

```text
训练代码学习：从 Git 历史读取或导出删除前版本
官方推理模型：从 Hugging Face 下载 TS-Guard 权重
评测与 Agent：使用当前 ToolSafe 工作树
```

## 10. GRPO 基础知识清单

### 10.1 关键术语

- **RL（Reinforcement Learning）**：根据奖励而不是唯一标准文本更新策略。
- **Policy / Actor**：正在被训练、负责生成回答的模型。
- **Prompt**：输入模型的用户请求、历史交互、候选动作和工具描述。
- **Response / Completion**：模型针对 prompt 生成的完整回答。
- **Rollout**：当前策略针对一个 prompt 实际采样生成一次回答的过程和结果。
- **Reward**：对一条 rollout 的最终标量评分。
- **Group**：同一个 prompt 生成的多条 rollout。
- **Advantage**：某条 rollout 相对于比较基准好多少或差多少。
- **PPO**：通常使用 actor 和 value network，并限制每次策略更新幅度。
- **GRPO**：用同一 prompt 的组内奖励统计量估计 advantage，不再单独训练 value network。
- **Value Network / Critic**：预测当前状态预期回报的辅助模型。
- **Old Policy**：本轮更新前的策略快照，用于限制单次更新。
- **Reference Model**：冻结的初始模型，用于 KL 约束整个训练过程的长期偏移。
- **Batch Size**：一次收集或处理的 prompt/样本规模，不是 token 数量。
- **Mini-batch**：从训练 batch 中拆出的优化批次。
- **Micro-batch**：单张 GPU 一次实际放入显存计算的更小批次。
- **Epoch**：完整遍历一轮训练数据；在线 RL 中通常还伴随重新生成 rollout。
- **KL Divergence / KL Loss**：衡量当前策略与参考策略概率分布差异的约束项。

### 10.2 TS-Guard 多任务奖励

设三个预测分别为：

- `h`：用户请求是否恶意；
- `v`：是否存在第三方攻击；
- `y`：当前动作的安全等级。

奖励函数：

```text
r = 1
    - 1/3 * I[h 预测错误]
    - 1/3 * I[v 预测错误]
    - 1/3 * I[y 预测错误]
```

- 三项全对：奖励为 1。
- 错一项：奖励约为 0.67。
- 错两项：奖励约为 0.33。
- 三项全错：奖励为 0。
- 历史 reward parser 中，结构化标签缺失或格式无法解析时也会得到 0 分。

### 10.3 组内相对优势

对于同一个 prompt 生成的 `G` 条 rollout：

```text
group mean = 同组奖励平均值
group std  = 同组奖励标准差
A_i        = (r_i - group mean) / (group std + epsilon)
```

示例：

```text
奖励：       1.00    0.67    0.33    0.00
平均奖励：   0.50
advantage： +1.34   +0.45   -0.45   -1.34（约值）
```

- 正 advantage：提高该回答中已选 token 的生成概率。
- 负 advantage：降低该回答中已选 token 的生成概率。
- 必须在同一个 prompt 内比较，因为不同 prompt 的固有难度不同。

### 10.4 强化学习如何产生梯度

强化学习不是随机修改模型参数。它优化的是期望奖励：

```text
J(theta) = E[R(y)]
```

策略梯度将奖励转换为可求导的目标：

```text
gradient J ≈ A(y) * gradient log pi_theta(y | x)
```

简化的最小化损失：

```text
L_RL = -A(y) * log pi_theta(y | x)
```

因此：

- `A > 0` 时，梯度下降提高该回答的概率。
- `A < 0` 时，梯度下降降低该回答的概率。
- reward 本身可以是不可导的规则函数；训练只需对模型的 token log-probability 求导。
- 采样负责探索，advantage 与策略梯度负责有方向地更新参数。

TS-Guard 使用整条输出的 reward，并将同一个 output-level advantage 作用于该回答的所有 token。这种方式简单，但无法精确指出哪一个推理 token 导致了最终奖励。

### 10.5 PPO 与 GRPO 的 value network 差异

```text
PPO advantage  ≈ 实际回报 - value network 预测回报
GRPO advantage ≈ 当前奖励 - 同组平均奖励（再除以标准差）
```

GRPO 的好处：

- 不需要单独训练 critic/value network。
- 减少模型参数、显存和额外计算。
- 对于能通过规则直接评分的多个候选回答比较方便。

需要注意：去掉 value network 不代表去掉 reference model。

### 10.6 KL loss 如何约束模型

KL 不依赖输出文本中的特殊标记，而是比较两个模型在同一 token 位置上的概率分布：

```text
当前策略：   pi_theta(token | context)
参考策略：   pi_ref(token | context)
```

```text
KL(pi_theta || pi_ref)
    = sum_v pi_theta(v) * log(pi_theta(v) / pi_ref(v))
```

- 两个概率分布相同：KL 为 0。
- 当前模型偏离参考模型越远：KL 越大。
- 总损失可以直观理解为 `GRPO 策略损失 + beta * KL 惩罚`。
- GRPO 梯度推动模型获得更高奖励，KL 梯度将模型拉回初始分布附近。
- KL 用于防止过度拒绝、语言能力退化、固定模板和 reward hacking。

需要区分：

- **Old Policy**：通过 probability ratio 和 clipping 约束单次更新。
- **Reference Model**：通过 KL 约束整个训练过程相对初始模型的偏移。

### 10.7 历史训练配置的含义

```text
algorithm.adv_estimator=grpo
```

- 选择 GRPO 的组内相对 advantage 计算器。

```text
actor_rollout_ref.rollout.n=16
```

- 同一个 prompt 生成 16 条 rollout。

```text
data.train_batch_size=256
```

- 一次训练批次包含约 256 个 prompt；结合 16 条 rollout，概念上可产生约 4096 条候选输出。

```text
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.001
algorithm.use_kl_in_reward=False
```

- KL 作为 actor loss 的额外项，而不是直接修改任务 reward。

```text
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.tensor_model_parallel_size=2
```

- 使用 vLLM 生成 rollout；单个模型权重横跨 2 张 GPU 做张量并行。

```text
data.max_prompt_length=4096
data.max_response_length=1024
trainer.total_epochs=10
```

- 输入最多 4096 token，输出最多 1024 token，训练共 10 个 epoch。

可复现性差异：历史脚本设置 4 GPU/节点、1 节点，而论文实验环境写 8 张 H20，不能把历史脚本直接等同于论文最终训练配置。

## 11. 学员疑问与答案归档

### Q1：为什么最终输出检测对 Agent 不够？

**学员理解：** Agent 能与外部世界真实交互，最终输出检测发生在工具调用之后，恶意调用可能已经造成破坏。

**补充答案：** 理解正确。更严谨的说法是“单独依赖最终输出检测不充分”。最终输出检测仍能处理文本风险，但必须增加执行前的 step-level 检查。

### Q2：TS-Guard 和 TS-Flow 分别负责什么？

**学员理解：** TS-Guard 根据请求、历史和候选调用给出风险判断；TS-Flow 把反馈交给 Agent，引导下一步调用。

**补充答案：** TS-Guard 是训练出的判断模型；TS-Flow 是运行时框架。TS-Flow 会先阻止当前危险调用，再把 guard feedback 作为 observation 交给 Agent 重规划。

### Q3：为什么不发现风险就直接终止？

**学员理解：** 直接终止会降低可用性，影响良性任务。

**补充答案：** 理解正确。TS-Flow 的目标是只阻止危险步骤，并尽可能让 Agent 继续完成原始良性任务，从而优化 Safety-Utility trade-off。

### Q4：TS-Guard 是否就是 Qwen2.5-7B？Agent 的基座是什么？

**答案：** TS-Guard 以 Qwen2.5-7B-Instruct 初始化，经 GRPO 更新参数后成为新的 checkpoint。论文中的 Agent backbone 是 GPT-4o 或 Qwen2.5-14B-Instruct。Agent 与 Guard 是两个独立模型。

### Q5：强化学习没有标准答案，模型怎样知道参数更新方向？

**答案：** 奖励先被转换为 advantage，再构造 `-A * log probability` 形式的策略损失。正 advantage 提高采样回答的概率，负 advantage 降低它的概率；梯度由 token log-probability 反向传播得到，不是随机修改参数。

### Q6：GRPO 是不是随机尝试参数来碰运气？

**答案：** 不是。随机性只用于从当前策略采样不同 rollout 进行探索，参数更新由 reward、advantage 和策略梯度决定，并受到 clipping 与 KL 的约束。

### Q7：KL 如何知道模型偏离了多少？输出里有指示信息吗？

**答案：** 不需要文本指示。当前模型和冻结参考模型对相同上下文分别输出 token 概率分布，训练代码直接计算两组分布的 KL divergence，并将其作为损失惩罚。

### Q8：为什么 GRPO 不需要 value network，却还需要 reference model？

**答案：** GRPO 用同组奖励统计量代替 value network 来估计 advantage；reference model 不负责估计 advantage，而是用于计算 KL，限制当前策略偏离初始模型。

## 12. 当前可用于面试的简短表达

### ToolSafe 项目故事

> 传统护栏主要检查静态输入和输出，但 Agent 的危险副作用可能发生在中间工具调用。ToolSafe 通过 TS-Bench 定义步级安全检测任务，用 GRPO 训练 TS-Guard 分析用户恶意性、攻击关联和当前动作风险，再由 TS-Flow 在工具执行前拦截危险动作并把反馈交给 Agent 重规划，从而降低攻击成功率并尽量保留任务完成能力。

### GRPO 训练故事

> 对每个包含 Agent 交互历史的 prompt，TS-Guard 生成多条结构化安全判断，并通过三个分类任务计算 reward。GRPO 在同一 prompt 的 rollout 组内计算相对 advantage，用它提高高奖励输出的 token 概率、降低低奖励输出的概率，从而不再需要 PPO 的 value network；同时使用 clipping 限制单次更新，并通过 KL loss 限制模型长期偏离初始 Qwen。

## 13. 待继续核验的问题

- [ ] 逐表解释 strict、loose、exact 三种评测模式。
- [ ] 理解 Accuracy、Precision、Recall、F1、ASR、Utility、Refusal Rate 和 Task Completion Score。
- [ ] 精读 SFT、SFT+RL、RL-only 消融实验，判断“RL-only 更好”的证据边界。
- [ ] 核验论文完整 TS-Flow feedback 与当前 `sec_react_agent.py` 只传递 risk rating 的差异。
- [ ] 从 Git 历史读取 TS-Guard 训练数据格式、reward parser 和训练脚本的完整调用链。
- [ ] 决定后续是否将删除前的训练代码导出到独立目录，避免污染当前主工作树。

