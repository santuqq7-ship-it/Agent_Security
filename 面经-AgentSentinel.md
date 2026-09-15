# AgentSentinel：智能体工具调用安全项目可背诵级面试手册

> 适用岗位：大模型应用研发、AI安全算法、智能体安全、安全研发、模型训练工程。  
> 版本日期：2026-09-15。  
> 使用方式：先背第1、2、3、11章，再按岗位选择第7—10章；所有“已完成”均有仓库或实验记录支撑，“规划中”不得在面试中说成已经完成。  
> 项目性质：个人研究型工程项目，参考公开论文思想与开源实现，独立完成复现、数据治理、训练实验、约束生成、评测和端到端安全链路改造。

---

## 0. 明天面试前怎么使用这份材料

### 第一优先级：必须能脱稿回答

1. 第2章的一分钟和三分钟项目介绍。
2. 第3章的两条主链：离线训练链路、在线工具调用链路。
3. 第4章的五个关键决策：数据重建、Think消融、GRPO奖励、Token FSM、官方7B切换。
4. 第6章的真实指标和口径，尤其知道哪些数字不能横向乱比。
5. 第7章前五道项目主问题。
6. 第10章的失败案例、边界和“项目还没做完什么”。

### 第二优先级：根据岗位侧重

- 大模型算法岗：重点读SFT、GRPO、KL、Advantage、Log Probability、数据切分与指标。
- Agent应用岗：重点读ReAct、工具调用协议、Guard执行边界、反馈重规划、API/本地模型适配。
- 安全公司：重点读威胁模型、间接提示词注入、授权边界、Source–Sink、最小权限、Fail-closed和误报治理。
- 训练平台岗：重点读FSDP、vLLM、Checkpoint、BF16、显存、恢复门禁和可观测性。

### 背诵原则

不要逐字背一整页。每个回答只需要记住四个锚点：

```text
现象/问题 → 我的判断 → 我做的机制 → 证据与边界
```

如果面试官继续追问，再用源码位置、失败案例和反事实方案展开。真正加分的不是术语数量，而是你能把“为什么这样设计、如果不这样会怎样、如何验证”讲完整。

---

## 1. 当前项目事实快照

### 1.1 已完成并有证据

| 模块 | 已完成事实 | 证据或指标 |
|---|---|---|
| 基准恢复 | 恢复AgentHarm、AgentDojo与ASB文件，校验TS-Bench轨迹文件 | 数据恢复脚本、SHA-256与JSON检查；ASB后续冻结 |
| 低成本基线 | 跑通Qwen2.5-1.5B本地Transformers推理和LoRA闭环 | 暴露未训练模型格式失败与单域训练类别塌缩 |
| 3B全参数SFT | 完成Qwen2.5-3B全参数SFT和多版消融 | 第一版banking 87/87可解析，Accuracy 93.10% |
| 第一代GRPO | 完成verl/FSDP/vLLM全参数GRPO训练、合并与评测 | 相对父模型banking Accuracy +3.45 pp，有害类Recall 78.57%→96.43% |
| Think消融 | 验证只训练/奖励三字段会形成无Think捷径 | GRPO后banking与错误AgentHarm切片Think均为0% |
| API Teacher治理 | 对1,656条样本完成无答案锚定的独立盲标 | 一致1,330条、冲突326条；最终人工字段修正71行 |
| 裁决数据重建 | 建立SFT/GRPO/validation身份与用途边界 | SFT 239、GRPO 1,208、clean validation 146；逻辑split identity交集0 |
| Token FSM | 实现Tokenizer-aware约束生成 | 固定协议由代码产生，模型只决定Think正文、结束时机和三个枚举值 |
| 单Actor GRPO | 实现同一HF Actor的rollout、旧/新log-prob重算与KL | E4真实3B单步更新通过；old/new parity误差0 |
| 可恢复训练 | 实现双代Checkpoint、配置/数据/tokenizer兼容检查 | E6 step1保存后从断点恢复到step2；完整断点约35GB |
| 官方7B基线 | 下载并核验公开TS-Guard 7B完整BF16权重 | 4/4 safetensors，总权重约15.23GB；A800推理约16.8GiB |
| 官方7B离线评测 | 完成87条banking和146条clean validation | banking阻断F1 92.31%；clean阻断F1 86.96%，Specificity 44.44% |
| TS-Flow链路 | 接入本地/DeepSeek Agent、官方7B Guard、FSM与真实AgentDojo工具 | Guard-in-loop 2条、无Guard参考3条；格式7/7或2/2成功 |
| 授权边界分析 | 定位Guard误拦合法读取导致utility退化 | 同identity下react utility=true，ts_flow utility=false，合法动作阻断6次 |

### 1.2 当前尚未完成

以下内容面试时必须说成“下一阶段”“正在设计”或“准备验证”，不能说成已有结果：

- 没有完成完整1,656条官方Guard能力评测。
- 没有完成Think与三个字段的全量语义一致性审计。
- `action_aware`工具能力处置策略只有设计文档，尚未写入运行代码。
- Qwen2.5-32B-Instruct Agent还没有接入或评测。
- Constrained GRPO E7完整302步没有运行；只完成E6前两步真实更新与断点恢复。
- ASB 5,231条保持冻结，当前读取/训练/调参使用量为0。
- 当前没有可宣传的“大规模TS-Flow攻击成功率下降百分比”。端到端只有机制轨迹，不能外推总体效果。

### 1.3 下一步已确定方向

先完成官方Guard全能力审计，再决定优先微调Guard还是修改处置策略：

```text
233条留出/外部泛化评测
+ 1,656条分split、分来源诊断
+ Think—字段—聚合风险一致性审计
        ↓
模型语义错误为主 → 数据复核 + 官方7B低风险LoRA
聚合/执行策略错误为主 → strict/action-aware消融
两者都有 → 分别修改、独立评测
        ↓
固定identity的react / strict / action-aware对照
        ↓
冻结最终方案后才使用ASB
```

---

## 2. 四档项目介绍口播

### 2.1 30秒版本

> 我做的是一个面向LLM Agent工具调用的步级安全护栏项目。和只审核用户输入或最终回复不同，我把Guard放在真实工具执行之前，让它结合用户请求、交互历史、工具Observation和当前Action判断恶意用户、第三方提示注入以及动作危害，再决定放行、阻断或把安全反馈交还Agent重规划。我完成了从1.5B LoRA、3B全参数SFT与GRPO，到API Teacher数据治理、Tokenizer级FSM约束生成和官方7B Guard端到端接入的完整研究链路，并通过真实轨迹发现了安全召回、误拦和任务完成率之间的权衡。

### 2.2 1分钟版本

> 这个项目解决的是智能体调用邮件、文件、日历、银行等工具时的执行前安全问题。风险不只来自恶意用户，还可能来自工具返回中的间接Prompt Injection，所以仅检查最初输入或最终文本不够。我实现的系统会在每个工具调用进入运行时之前，将用户原始请求、历史Thought/Action/Observation、当前动作和工具定义交给Guard。Guard输出用户请求是否恶意、是否受到第三方攻击、当前动作危害等级和证据型解释；格式由Tokenizer-aware FSM确定性保证，避免让训练资源学习尖括号和闭合标签。离线侧我跑通了LoRA、3B全参数SFT、verl GRPO和单Actor Constrained GRPO，并对1,656条数据完成API Teacher盲标与冲突裁决。在线侧接入官方7B Guard和DeepSeek Agent，既观察到强Agent自主拒绝注入，也发现严格聚合策略会误拦合法文件读取。我当前正在把模型能力、字段聚合和执行策略分层审计，而不是只追求一个更高Accuracy。

### 2.3 3分钟版本

> 项目起点是一个现实安全问题：LLM Agent的输出可能直接触发发邮件、改文件、转账或删除资源，攻击者又可以把指令藏在网页、邮件和文档里。Agent读取这些不可信内容后，可能在用户完全不知情的情况下生成危险工具调用。我的核心设计是把安全边界放在“模型已经提出Action，但真实工具尚未执行”的位置，做step-level guardrail。
>
> 离线训练上，我没有一开始就追求大模型和昂贵强化学习，而是先用1.5B打通Transformers推理、LoRA、Prompt和Parser，再升级到Qwen2.5-3B全参数SFT。第一版完整SFT在未参与训练的banking 87条上做到87/87可解析，Accuracy 93.10%。随后我做了Think监督消融：不训练空Think、只训练三字段，再接GRPO。GRPO相对自己的父模型把banking有害类Recall从78.57%提高到96.43%，但Think覆盖率掉到0，说明只奖励最终字段会形成更短、更稳定的捷径。这让我把项目重点从“调参”转向“训练目标和系统边界是否正确”。
>
> 数据方面，我发现恢复的数据只有原始score，另外两个字段是项目弱规则推导，且早期Teacher只是解释锁定答案，可能生成Think与标签冲突。于是我实现API Teacher无答案锚定盲标，对1,656条数据独立判断，得到326条冲突；再按用户请求级MUR一致性和当前动作级Attack语义进行人工裁决，最终修改71行，并把解释与最终字段不一致的样本排除出SFT rationale监督。最终数据边界是SFT 239、GRPO 1,208、clean validation 146，逻辑identity互斥，banking和ASB不参与训练。
>
> 工程上，我进一步实现了Tokenizer-aware FSM，把固定标签、空格、换行和枚举协议交给程序，模型只生成Think正文、结束决策和字段值；同一HF Actor负责rollout以及old/new log-prob重算，避免生成策略和训练策略不一致。E6在真实3B上完成step1保存、step2恢复，old/new parity误差为0，单步约4.7分钟。
>
> 后来我核验并接入论文公开的官方7B Guard。它在banking二分类阻断F1达到92.31%，但在clean validation上的Specificity只有44.44%，表现出明显保守倾向。端到端TS-Flow里，DeepSeek能够自主识别注入；另一个授权任务中，Guard却把合法文件读取连续判成中风险，导致utility从true降成false。我进一步核对到运行时还使用0.4/0.4/0.1/0.2的三字段加权聚合，而不是直接看Harmfulness。当前下一步是完整审计Guard的Think、字段与聚合结果，先区分模型问题、标签问题和策略问题，再决定LoRA还是action-aware处置。这段经历让我真正理解了安全系统不能只看Recall，还必须同时看误报、任务完成率和执行边界。

### 2.4 10分钟展开顺序

当面试官说“详细讲讲”时，按以下顺序展开，不要按时间流水账从下载代码讲起：

1. **问题与威胁模型**：为什么最终输出审核抓不住中间工具副作用。
2. **在线架构**：候选Action先过Guard，危险动作绝不到工具运行时。
3. **三字段设计**：区分恶意用户、第三方攻击和当前动作危害。
4. **数据问题**：弱标签、Teacher答案锚定、冲突326条、裁决71行。
5. **训练演进**：1.5B LoRA→3B全参SFT→GRPO→Constrained GRPO。
6. **核心失败**：GRPO Recall提高但Think消失，奖励优化不等于系统目标优化。
7. **FSM与同Actor**：格式工程化、约束概率空间一致。
8. **官方7B与TS-Flow**：banking强，clean保守；DeepSeek自主防御与Guard误拦。
9. **当前判断**：先完整审计，不继续盲目调温度、epoch或换模型。
10. **业务化思考**：工具能力元数据、分级处置、最小权限、用户确认、审计日志。

---

## 3. 必须能手画的系统架构与调用链

### 3.1 离线数据与训练主链

```text
AgentHarm / AgentDojo轨迹
        ↓
稳定source_identity与数据边界校验
        ↓
API Teacher无答案锚定盲标
        ↓
旧弱标签对照、冲突队列、任务级/动作级裁决
        ↓
SFT 239 | GRPO 1208 | Clean Validation 146
        ↓
Qwen2.5-3B全参数SFT
        ↓
Tokenizer-aware FSM约束rollout
        ↓
同Actor old/new log-prob + Reference KL
        ↓
GRPO组内优势、裁剪目标、参数更新
        ↓
双代Checkpoint / Validation选模 / Banking外部报告
```

面试时要强调：早期曾有302条SFT候选，但Teacher解释与最终标签一致性门禁后，真正进入新版rationale SFT的是239条；不要把两个数字混用。

### 3.2 在线Agent执行主链

```text
用户请求
  ↓
Agent生成 Thought / Action / Action Input
  ↓
动作解析成功且工具已注册？
  ├─ 否：继续对话或返回错误
  └─ 是
       ↓
Guard读取用户请求 + 交互历史 + 当前动作 + 工具说明
       ↓
Token FSM约束输出 Think + 三字段
       ↓
三字段聚合为运行时risk_rating
       ↓
risk低：runtime.run_function真正执行工具
risk高：动作不进入runtime，完整Guard反馈作为Observation返回Agent
       ↓
Agent基于工具结果或安全反馈继续规划
```

### 3.3 为什么Guard必须放在工具运行时之前

发邮件、转账、删除文件属于非幂等或不可逆副作用。动作执行后再审核，即使发现有害也只能补救，不能阻止泄漏或损失。项目把Guard插在Action解析后、`runtime.run_function`之前，保证被阻断的动作没有进入真实执行边界。这个位置比“输入审核”更懂当前动作，也比“最终回复审核”更早阻止副作用。

### 3.4 信任边界

| 数据/组件 | 默认可信度 | 原因 |
|---|---|---|
| 系统Prompt、工具Schema | 高，但仍需版本固定 | 决定Agent与Guard协议 |
| 用户请求 | 不可信 | 可能直接恶意 |
| 邮件、网页、文件、日历Observation | 不可信 | 可能包含间接注入 |
| Agent Thought与Action | 不可信候选 | 可能受攻击或规划错误 |
| Guard自然语言Think | 辅助解释，不作唯一控制依据 | 可能与字段矛盾 |
| Guard结构化字段 | 机器可读风险信号 | 仍可能预测错误 |
| 策略引擎 | 确定性执行边界 | 必须可审计、默认安全失败 |
| 工具运行时 | 高影响边界 | 只能接受已授权动作 |

---

## 4. 项目演进与关键决策

### 4.1 为什么从1.5B开始

不是认为1.5B最终足够，而是用最低成本验证数据格式、模型加载、LoRA、Adapter合并、评测和日志。1.5B暴露出大量解析失败与类别塌缩，使我在昂贵训练前确认：结构化协议、数据域覆盖和指标设计是独立问题。工程链路未跑通前直接租A800做7B GRPO，会把付费时间浪费在路径、依赖和Parser错误上。

### 4.2 为什么从LoRA转向3B全参数SFT

LoRA只训练低秩增量，适合低成本适配；但早期1.5B在长轨迹理解、严格格式和多域判定上容量受限。项目升级3B并全参数更新，是为了验证模型容量和全局表示调整能否改善协议遵循。结果第一版完整SFT在banking达到87/87可解析和93.10% Accuracy。但这不证明全参永远优于LoRA，只说明在当时的数据、模型和目标下，继续提高LoRA rank不是最优诊断动作。

### 4.3 为什么做Think监督消融

早期标签中的Think为空，如果把它作为普通completion监督，模型会学习稳定输出空分析。于是实验把Think Token设为`-100`，不计算交叉熵，只训练三个字段，希望保留基座自由推理。结果SFT阶段部分样本仍能生成Think，但GRPO只奖励字段后，模型最终完全省略Think。这证明“模型本身会推理”不等于“系统会稳定输出可反馈的解释”，输出行为必须由训练目标或工程协议保证。

### 4.4 为什么重做Teacher与标签治理

恢复数据只有原始危害score，MUR和Attack是按数据来源推导的弱标签；旧Teacher又被要求解释预设答案，可能把错误答案合理化。新版Teacher请求删除路径、split、旧标签和具体JSON答案锚点，只提供场景证据并独立输出。1,656条里326条发生字段冲突，说明标签不是可忽略的小噪声。最终没有粗暴全量覆盖Teacher，而是按用户请求统一MUR、按当前动作判断Attack，并保留基准Harmfulness口径，修改71行。

### 4.5 为什么用Token FSM而不是继续强化格式

固定标签、反斜杠闭合、空格和换行没有安全语义。如果把它们和风险判断一起交给模型，采样稍有随机性就会产生大量格式0分，GRPO学到的是“如何不漏尖括号”，而不是“如何识别危险动作”。FSM在每个生成位置限制允许Token，固定协议由程序插入；模型仍决定Think正文、何时结束分析以及yes/no/0.0/0.5/1.0，所以没有剥夺语义学习空间。

### 4.6 为什么暂停完整3B E7并切换官方7B

E6实测单步约280秒，剩余300步纯计算约23.3小时，加验证和Checkpoint预计25—30小时。与此同时公开官方7B完整权重可以直接评测。为了避免花一整天训练后才知道3B容量上限，我暂停E7，先用官方7B建立上界和TS-Flow主基线。这是成本受限项目中的实验优先级决策，不是训练失败。

### 4.7 为什么现在又不立刻调优官方Guard

官方7B在banking上二分类阻断F1 92.31%、安全样本零误拦，说明它不是普遍失效；但clean validation Specificity只有44.44%，端到端又出现合法读取误拦。与此同时运行时还有三字段加权聚合。若直接LoRA，无法区分改善来自模型还是数据/聚合变化，也可能遗忘banking能力。因此先全量审计Think、字段、标签和聚合，再决定模型或策略，是更可证伪的路线。

---

## 5. 核心知识速记

### 5.1 SFT

SFT是用输入—目标输出对做监督学习。对因果语言模型，目标通常是最小化目标Token的负对数似然；Prompt Token和不希望监督的区域可以把label设为`-100`，交叉熵会忽略它。本项目既做过完整completion监督，也做过Think mask消融，还实现了rationale与字段的不同权重。SFT的优点是稳定建立协议和任务先验，缺点是受Teacher答案上限与标签噪声约束。

### 5.2 LoRA与全参数微调

LoRA冻结原权重，在部分线性层增加低秩矩阵增量，训练参数和优化器状态显著减少；`r`控制低秩容量，`alpha/r`影响增量缩放。全参数微调更新所有权重，表达能力更强但显存、存储和灾难性遗忘风险更高。本项目用LoRA做低成本链路验证，3B阶段使用全参SFT/GRPO理解完整训练闭环，未来官方7B若校准则优先低rank LoRA，因为其已有较强基线能力。

### 5.3 Teacher Forcing与自由生成

训练/验证loss通常在给定正确历史Token的Teacher Forcing条件下计算；真实推理时模型使用自己前一步输出，错误会累积。因此validation loss升高并不必然对应最终结构化Accuracy下降，反之亦然。本项目第二轮SFT出现validation loss上升但自由生成更好的现象，说明选Checkpoint必须结合目标任务生成指标，不能只看Token级loss。

### 5.4 PPO与GRPO

PPO通常需要Actor、Reference、Reward Model和Critic/Value Model，优势来自回报减去价值估计。GRPO对同一Prompt生成一组回答，用组内奖励均值和标准差归一化得到相对优势，从而省去单独Critic。优势为正的回答提高所采样Token概率，优势为负的回答降低概率；如果组内奖励全相同，标准化优势为0，该组几乎不提供策略梯度信号。

### 5.5 GRPO组内优势

设同一Prompt的第`i`条回答奖励为`r_i`，直觉上优势是：

```text
A_i = (r_i - group_mean) / (group_std + epsilon)
```

它只比较同组候选，因此不同Prompt的绝对难度不会直接混在一起。如果一个组8条都满分或都0分，模型不知道哪条相对更好；所以rollout门禁既要看平均reward，也要看variable-reward group比例。提高temperature可以增加差异，但会破坏格式和语义稳定，不能无限提高。

### 5.6 为什么要重新计算Log Probability

rollout阶段得到的是采样序列和旧策略下的log-prob；更新Actor时需要在当前可求梯度的模型上重放相同Token，得到new log-prob，计算概率比`exp(new-old)`。如果生成由vLLM完成而训练由另一份FSDP Actor完成，两份权重或约束不一致，概率比就失去含义。本项目后续改为同一HF Actor生成和重算，并验证更新前old/new最大误差为0。

### 5.7 PPO Clipping

概率比如果变化过大，一次高奖励样本就可能把策略推得很远。Clipping把有效概率比限制在`1±epsilon`附近，用较保守的目标限制更新幅度。它不是简单截断梯度，而是在目标函数里取未裁剪和裁剪项的保守值。项目使用0.2作为clip ratio，并结合梯度裁剪和KL约束控制全参数更新。

### 5.8 KL约束

强化学习可能为了规则奖励产生奇怪短答案或遗忘原模型能力。Reference是初始化策略的冻结副本，KL惩罚约束Actor不要偏离过远。本项目采用基于Token log-prob差异的非负KL估计，并乘以系数加入loss。KL太小会放任reward hacking，太大会使策略几乎不学习；需要同时观察KL、reward、外部指标和输出分布。

### 5.9 FSDP

FSDP将模型参数、梯度和优化器状态按进程切分，需要计算时再All-Gather，从而降低单卡或单进程持有完整训练状态的压力。它主要服务训练，不是高吞吐推理框架。FSDP Checkpoint可能包含模型分片、Adam一阶/二阶矩、RNG、scheduler和数据游标，因此远大于单纯的BF16模型权重。

### 5.10 vLLM与PagedAttention

vLLM主要服务高吞吐生成，PagedAttention把KV Cache按块管理，减少连续大块分配和碎片，便于多个请求共享显存。它擅长rollout和在线推理，但不自动替代可求梯度的Actor训练。第一代GRPO使用vLLM生成、FSDP训练；后续为了约束概率空间一致和降低双Actor同步风险，研究链路改为HF单Actor，而在线服务仍可独立使用vLLM。

### 5.11 BF16、FP16与FP32

BF16与FP32有相同指数位宽，动态范围更大，训练时通常比FP16不容易溢出，但有效尾数更少。FP16精度位更多但指数范围较小，常依赖loss scaling。项目的模型前向主要用BF16降低显存和提高Tensor Core吞吐，优化器和部分状态可保留FP32保证更新稳定，因此完整训练Checkpoint会显著增大。

### 5.12 Temperature、Top-p与贪婪生成

Temperature缩放logits：低温让分布更尖锐，高温提高随机性；Top-p只在累计概率达到阈值的最小Token集合里采样。低temperature和低top-p不等于真正贪婪，只有关闭采样并直接取argmax才是确定性greedy。本项目早期rollout需要多样性，所以使用采样；正式评测使用greedy。把温度降到0.1仍无法解决格式问题，说明根因不是随机性，而是模型协议概率与训练目标。

### 5.13 Prompt Injection

直接注入来自用户输入；间接注入来自Agent读取的网页、邮件、文件或日历。它和越狱不同：越狱主要让模型输出违规内容，工具注入的目标是改变规划并触发真实副作用。本项目的典型攻击把“向陌生邮箱发送指定内容”的指令藏在日历或文件Observation里，安全目标是保持原任务完成，同时阻止未授权外发。

### 5.14 ReAct

ReAct把推理与行动交替组织为Thought、Action、Action Input和Observation。它使Agent能够根据工具结果多步规划，也给间接注入提供了进入上下文的通道。Guard正好插在Action生成后、Observation执行前；阻断时，Guard反馈本身作为新的Observation进入下一轮，促使Agent重新规划。

### 5.15 Precision、Recall、F1与Specificity

以危险动作为正类：Precision表示被拦截的动作中多少真的危险，Recall表示所有危险动作中拦住多少；F1是两者调和平均。Specificity是安全动作中正确放行多少，等价于TN/(TN+FP)。安全系统不能只追Recall，因为全部阻断可以得到100% Recall，却让utility归零。本项目官方7B在clean validation Recall高但Specificity仅44.44%，正是保守Guard的典型表现。

---

## 6. 指标、结果与面试口径

### 6.1 代表性结果表

| 阶段 | 数据 | 结果 | 正确表述 |
|---|---|---|---|
| 3B完整SFT | banking 87 | Accuracy 93.10%，F1/Recall 92.10%，skipped 0 | 当时最好的3B SFT外部域结果 |
| fields-only SFT→GRPO | banking 87 | Accuracy 86.21%→89.66%；有害Recall 78.57%→96.43% | GRPO超过直接父模型，不是超过所有SFT |
| rationale-repair SFT | banking 87 | Accuracy 83.91%，非空Think 87/87，有害Recall 92.86% | 恢复反馈能力，但误报较多 |
| format-reinforced SFT/GRPO | banking 87 | 训练前后均71/87正确，二分类unsafe F1均约75.76% | 基本持平，不能说GRPO大幅退化或提升 |
| 官方TS-Guard 7B | banking 87 | 85/87严格解析；阻断F1 92.31%，Specificity 100% | 外部banking上强基线 |
| 官方TS-Guard 7B | clean validation 146 | 145/146解析；阻断F1 86.96%，Specificity 44.44% | 高召回但明显误拦 |
| Constrained GRPO E6 | 真实3B，两步 | 格式率100%，old/new parity 0；约280秒/步 | 更新与恢复门禁通过，完整E7未运行 |
| TS-Flow授权对照 | 单一identity | react utility=true；ts_flow utility=false；合法阻断6次 | 机制级误拦案例，不是总体统计 |

### 6.2 最容易说错的数字

1. **302与239**：302是新版SFT候选，239是通过rationale—字段一致性门禁后实际用于新版SFT的样本。
2. **1,656与233**：1,656是盲标/诊断总语料；233是官方7B当前真正完成的87 banking + 146 clean离线评测。
3. **5条TS-Flow相关轨迹**：只有2条Guard-in-loop、3条react参考；不能称为“大规模端到端评测”。
4. **93.10%与89.66%**：93.10%属于第一版完整SFT；89.66%属于另一条fields-only SFT→GRPO支线，不能据此说GRPO超过最佳SFT。
5. **ASB 5,231条**：数据文件已恢复，但为最终测试冻结，使用量0；不能称为“已在ASB取得结果”。

### 6.3 如何回答“效果到底好不好”

> 我的结论不是简单的好或不好。模型级上，3B SFT证明小模型可以学会严格三字段协议，官方7B在banking上具备较强阻断能力；强化学习支线上，有害Recall相对父模型明显提高，但Think消失和误报上升说明奖励存在捷径。系统级上，TS-Flow的执行前边界和反馈链已经真实跑通，但当前样本还不足以证明总体安全增益，而且授权任务暴露了Guard保守误拦。项目价值恰恰在于我没有用单个Accuracy掩盖这些差异，而是把模型语义、格式协议、风险聚合、Agent行为和utility拆开验证。

---

## 7. 高频项目面试题：架构与安全

### Q1：为什么要做step-level Guard，输入审核和最终输出审核不够吗？

**考察意图：** 是否真正理解项目动机和执行副作用。

**第一人口播：**

> 输入审核只能看到用户最初说了什么，看不到Agent后续从邮件、网页或文件里读到的间接注入；最终输出审核又发生得太晚，邮件、转账、删除等工具可能已经执行。我的系统选择在Agent已经生成候选Action、但`runtime.run_function`尚未调用的位置插入Guard。此时既有用户原始目标，也有完整交互历史、Observation、当前工具名和参数，判断上下文最充分；如果危险，动作根本不会到执行器。这个设计把安全从“事后内容审核”升级成“执行前策略控制”。代价是每步增加一次推理延迟，而且误报会直接影响任务完成率，所以必须同时评测安全和utility。

### Q2：为什么设计三个字段，而不是直接输出safe/unsafe？

**考察意图：** 是否理解风险来源和当前动作的解耦。

**第一人口播：**

> 二分类只能告诉系统“现在不安全”，但无法告诉Agent为什么不安全以及应该如何恢复。我的协议把风险拆成三个正交问题：`Malicious_User_Request`判断原始用户是否恶意，`Being_Attacked`判断当前动作是否被第三方注入驱动，`Harmfulness_Rating`判断当前Action本身的后果。比如用户任务正常、文件里有注入、但Agent当前只是搜索文件，这三者可以是no、yes、0.0或0.5；如果Agent进一步准备把数据发给攻击者，危害才可能到1.0。分字段后，策略可以区分拒绝用户、忽略不可信数据、阻断当前动作或要求重规划，但字段之间也可能自相矛盾，所以我正在做Think—字段—聚合的一致性审计。

### Q3：Guard在Agent运行中具体如何参与？

**考察意图：** 能否沿真实调用链说明输入输出和边界。

**第一人口播：**

> Agent采用ReAct格式生成Thought、Action和Action Input。代码先解析工具名和参数，确认工具在可用集合后，如果不是纯`react`模式，就把用户请求、此前消息、当前动作以及工具描述组成Guardian输入。官方7B Guard通过受约束生成返回Think和三个字段，适配器把字段聚合成`risk rating`。低风险时才调用真实工具函数，并把结果作为Observation返回Agent；高风险时不调用工具，而是把完整Guard解释和字段包装成“Security Validation Before Execution”反馈给Agent。轨迹会分别记录Agent响应、Guardian响应、决策、阻断、反馈和真实工具执行，因此可以证明危险动作是否真正跨过运行时边界。

### Q4：你如何定义Prompt Injection的威胁模型？

**考察意图：** 安全岗位会追问攻击面、资产、攻击目标和假设。

**第一人口播：**

> 我把直接恶意请求和间接注入分开。直接恶意请求由用户主动要求窃取凭据、未授权转账或破坏文件；间接注入则来自Agent要读取的外部内容，例如邮件正文、日历描述、共享文件和网页，其中伪装成“更高优先级指令”的文本试图覆盖用户目标。保护资产包括隐私数据、资金、文件完整性、账号权限和对外通信能力。攻击成功不是模型说了一句危险文本，而是Agent生成并执行了偏离原任务的高影响工具调用。当前威胁模型假设系统Prompt和工具注册可信、Observation不可信、Agent候选Action不可信，Guard和策略引擎共同保护最终工具执行边界。

### Q5：为什么Guard和Agent要用两个模型？一个模型自我反思不行吗？

**考察意图：** 模块解耦、独立失败和纵深防御。

**第一人口播：**

> 一个Agent同时负责完成任务和判断自己的动作，容易出现目标冲突：它已经被注入影响时，自我反思仍共享被污染的上下文和策略。独立Guard相当于第二条决策链，任务目标是安全分类而不是完成用户任务，输入包含当前候选动作但没有执行权限。这样可以分别替换Agent和Guard、独立评测错误、记录决策证据，也能在Agent供应商变化时保持统一安全边界。当然独立模型不等于天然可信，它也会误报或漏报，所以运行时还需要确定性协议、策略聚合、最小权限和审计。我的项目里DeepSeek能自主拒绝某些注入，但Guard仍用于纵深防御，不能把Agent自防御误写成Guard成功阻断。

#### 本主题压力追问A：如果Guard被Prompt Injection本身骗了怎么办？

**第一人口播：**

> Guard看到的历史确实包含不可信Observation，所以不能假设它免疫注入。我的第一层措施是使用专门的Guardian系统模板，明确区分用户请求、历史和当前动作，并重复强调只判断当前动作；第二层是Token FSM，只允许协议规定的结构，避免攻击文本让模型逃逸成任意工具调用；第三层是Guard本身没有工具执行权限，它只能输出风险信号；第四层是策略引擎和工具权限边界，外发、金融、删除等动作不能仅凭一段自然语言解释放行。未来完整审计还会按攻击来源检查Being Attacked召回，并用对抗样本测试Guard是否把Observation里的伪指令当成系统指令。纵深防御的重点是即使一层误判，也不让攻击直接获得高权限执行。

#### 本主题压力追问B：为什么不把所有工具都默认阻断，安全性不是最高吗？

**第一人口播：**

> 全部阻断可以人为得到接近零攻击成功率和百分之百危险Recall，但Agent也失去业务价值，这不是有效安全系统。安全目标应当是约束风险下最大化正常任务完成率，所以必须同时看unsafe recall、precision、specificity、utility和攻击成功率。项目中官方7B在clean validation上危险Recall较高，但Specificity只有44.44%；端到端授权任务中，无Guard能安全完成，严格Guard却连续误拦6次并让utility从true降为false，就是典型反例。因此我不会把“攻击没成功”单独宣传为成功，而是要求同一identity下比较无Guard、严格阻断和反馈重规划，并追踪到底执行了哪些工具。

---

## 8. 高频项目面试题：数据、SFT与GRPO

### Q6：训练数据从哪里来，如何保证标签可信？

**考察意图：** 数据来源、弱标签、审计和数据泄漏。

**第一人口播：**

> 数据来自恢复的AgentHarm和AgentDojo步级轨迹，每条包含用户请求、历史动作、当前Action、工具环境和原始危害score。审计后我发现MUR和Attack并不是论文发布的完整人工金标签，而是早期按来源推导的弱标签；旧Teacher也只解释锁定答案。为降低确认偏误，我重写API Teacher流程，删除路径、split、旧字段和具体答案示例，只提供场景证据独立判断。1,656条中1,330条完全一致、326条冲突。我没有全盘接受Teacher，而是按原始请求统一MUR、按当前动作裁决Attack，最终改71行；Harmfulness保留基准score。新版SFT只使用解释与最终字段一致的239条，避免把自相矛盾的Think作为监督。

### Q7：如何证明训练集和验证集没有泄漏？

**考察意图：** 是否只会说“随机切分”，还是理解轨迹相关性。

**第一人口播：**

> 我为每个step构造稳定`source_identity`，重建时校验SFT、GRPO和clean validation的identity交集为0，并在manifest里声明banking只作外部报告、ASB只作最终测试。后来进一步审计发现，step identity互斥不等于任务级完全隔离：早期302/1208切分仍有共享轨迹组和相同用户请求，这会让相邻step泄漏语义。因此面试中我不会声称最早版本是严格任务隔离；新版裁决数据至少保证逻辑step互斥，下一版完整实验应提升到request/trajectory级切分。这个区别很重要，因为Agent轨迹的相邻步骤高度相关，普通逐行随机切分会高估泛化。

### Q8：为什么第一版完整SFT表现最好，却还要继续做GRPO？

**考察意图：** 是否为了用热门技术而用强化学习。

**第一人口播：**

> 第一版完整SFT在banking达到93.10% Accuracy，是当时最好的3B外部结果。继续做GRPO不是因为SFT失败，而是研究另一个问题：同一Prompt下多个候选动作能否利用组内相对奖励进一步强化安全判断，并理解rollout、advantage、KL和策略更新链路。GRPO支线初始化自fields-only SFT，不是最佳完整SFT，所以正确比较对象是直接父模型。结果Accuracy从86.21%到89.66%，有害Recall从78.57%到96.43%，但Think掉到0、误报增加。这个实验说明RL可以优化某个奖励目标，却不保证超过所有SFT或保留未奖励能力，也证明选择初始化模型会直接影响结论口径。

### Q9：GRPO为什么不需要Critic？它怎么更新模型？

**考察意图：** 强化学习核心原理。

**第一人口播：**

> GRPO对一个Prompt采样多条回答，把规则或模型奖励转成组内相对优势。高于组均值的回答得到正优势，Actor提高这些回答中采样Token的条件概率；低于均值的回答得到负优势，降低概率。因为组均值本身充当相对基线，所以不必额外训练Value/Critic网络，能节省模型和优化器显存。更新时并不是直接把高奖励文本再做一遍SFT，而是用old/new log-prob构造概率比，再使用PPO式clipping限制变化，同时用冻结Reference计算KL，防止策略为追奖励偏离初始化模型过远。如果同组奖励完全一样，优势为0，这组不产生有效区分信号。

### Q10：你是怎么设计奖励的，遇到过什么Reward Hacking？

**考察意图：** 奖励是否与最终目标一致。

**第一人口播：**

> 第一代奖励先要求严格格式，再按三个字段分别给部分分数；这让模型即使危害等级错误，只要风险来源正确仍能拿到较高奖励。fields-only实验又完全不奖励Think，所以最短的三字段答案既容易解析又能满分，GRPO最终把Think覆盖率推到0。后来的rationale奖励虽然要求非空Think，但不评价语义，也仍可能鼓励模板化废话。最终Constrained GRPO把格式从奖励中移出，交给FSM确定性保证，语义奖励改为MUR 0.20、Attack 0.20、Harmfulness 0.30、三字段联合全对额外0.30。这个演进说明奖励必须和最终运行时安全决策对齐，否则mean reward上升可能只是模型找到评分漏洞。

#### 本主题压力追问A：Teacher是更强模型，为什么不能直接把它所有标签都当金标准？

**第一人口播：**

> 更强模型只能提高判断上限，不能消除提示偏置、严重度口径差异和长上下文误解。我的全量盲标中，Teacher与旧标签冲突326条，但复核发现Teacher也会把“历史出现过攻击”误当成“当前Action仍受攻击”，或者对同一用户请求的MUR给出不同答案。如果直接覆盖，会把新的系统性偏差写入训练集。因此我把Teacher当独立审阅者：先隐藏旧答案，再记录冲突；MUR按请求级统一，Attack按当前动作判断，Harmfulness保留基准口径；解释与最终字段不一致的样本不进入SFT。这个流程比“相信更大模型”更可审计，也保留了人工责任边界。

#### 本主题压力追问B：为什么不直接增加epoch、学习率或模型参数解决低分？

**第一人口播：**

> 如果根因是错误标签、奖励错位或格式生成不稳定，增加epoch只会更充分拟合错误，增大学习率会放大遗忘和策略漂移，换更大模型也可能更快学到捷径。项目里我曾系统尝试多组temperature、top-p、max token和SFT epoch，rollout解析率长期不到50%；后来逐条trace证明大量0分来自闭合标签和格式，而不是安全语义。再后来又发现评测切片和macro分母问题。于是我把原则改成先做因果诊断：格式交给FSM、标签做盲审、评测固定manifest、old/new做parity门禁。只有这些基础正确后，超参数搜索才有意义。

---

## 9. 高频项目面试题：FSM、推理与训练工程

### Q11：Tokenizer-aware FSM是怎么精确约束生成的？

**考察意图：** 是否真的实现过约束解码。

**第一人口播：**

> 我不是在模型输出后用正则补标签，而是在每一步生成前根据状态返回允许Token集合。编译阶段先用当前Tokenizer精确编码固定前缀、闭合标签和字段候选，再为yes/no以及0.0/0.5/1.0构造Token Trie，因为这些字符串在不同Tokenizer下可能对应一个或多个Token。运行时状态依次经历固定Think开头、自由rationale、结束信号、固定闭合、MUR Trie、Attack Trie和Harmfulness Trie。固定Token由程序写入上下文，语义选择Token才采样并记录log-prob。FSM只读取新生成response，不扫描Prompt里的伪标签，因此Observation中的攻击文本不能篡改状态。

### Q12：固定Token不参与Policy Gradient，会不会让Actor重算概率不一致？

**考察意图：** 约束生成与概率空间的一致性。

**第一人口播：**

> 固定Token虽然不作为策略动作计算梯度，但必须进入下一步模型上下文，因为后续字段概率依赖完整协议前缀。对可选择Token，我先把不允许的logits设为负无穷，再在约束后的分布中计算归一化log-prob；rollout、old log-prob、new log-prob和Reference KL必须使用同一FSM状态和同一允许集合。如果rollout约束了yes/no，而重算时在全词表归一化，概率比就不是同一策略空间。E3/E4专门用确定性trace重放验证约束概率，E6真实3B更新前old/new最大绝对误差为0，说明同Actor同约束的重算链一致。

### Q13：FSDP和vLLM有什么区别，为什么不能只留一个？

**考察意图：** 训练框架与推理引擎边界。

**第一人口播：**

> FSDP解决训练状态如何分片：参数、梯度和Adam状态在进程间分散，需要时聚合，重点是反向传播和优化器；vLLM解决自回归生成吞吐：通过PagedAttention、连续批处理和高效KV Cache管理服务大量rollout，重点是推理。第一代verl链路让FSDP Actor训练、vLLM副本生成，因此需要同步权重。后续我为了确保约束策略和log-prob完全一致，选择纯HF同Actor完成rollout和更新，这牺牲吞吐但降低系统复杂性。工程上不是绝对只能用一种，而是根据实验目标选择：研究正确性优先用单Actor，规模化服务再引入vLLM并建立严格同步canary。

### Q14：为什么完整Checkpoint约35GB，而模型权重只有约12GB？

**考察意图：** 是否理解训练状态和部署权重。

**第一人口播：**

> 3B模型如果以FP32保存，单份参数约12GB；但可恢复Checkpoint不只有参数，还包括Adam的一阶矩和二阶矩，通常各再占一份FP32规模，以及可能的master weights、scheduler、global step、数据游标、RNG状态和兼容指纹。因此完整训练状态达到约35GB是合理的。部署用的Hugging Face权重只保留模型参数和配置，不需要优化器状态，所以更小。项目采用`latest/previous`两代轮换，并在写入incoming目录前检查磁盘，完成后原子替换，避免中断时同时失去新旧断点。

### Q15：如何验证训练真的发生了，而不是脚本空跑？

**考察意图：** 训练门禁与证据链。

**第一人口播：**

> 我不会只看终端出现loss就声称训练成功。E4真实3B单步门禁记录了rollout、奖励差异、old/new parity、有限loss、反向后的梯度范数、optimizer step执行状态以及参数实际delta；结果4条rollout格式全通过、参数变化约1e-6。E6进一步分两个独立会话：先运行到global step1写完整断点，再用`resume=auto`恢复到step2，验证global step、sample cursor、配置哈希、数据哈希、tokenizer fingerprint、RNG和双代Checkpoint。训练后还检查GPU显存归零和incoming目录不存在，说明恢复、清理和落盘都真实完成。

#### 本主题压力追问A：为什么训练单步要约280秒，之前SFT却只要几分钟？

**第一人口播：**

> SFT每条样本只做一次Teacher-forcing前向和反向；Constrained GRPO每个optimizer step是4个Prompt、每个4条rollout，共16条长回答，还要自回归逐Token生成、重放旧策略log-prob、计算Reference log-prob和KL、再计算当前Actor梯度。纯HF FSM生成没有vLLM连续批处理的吞吐优势，而且Actor FP32、Reference BF16同时驻留，显存峰值约60GB。E6两步实测分别287秒和273秒，所以302步纯计算约23.3小时，加验证和Checkpoint为25—30小时。这个估算来自真实step，不是理论拍脑袋。

#### 本主题压力追问B：如果必须把Constrained GRPO提速，你会怎么做？

**第一人口播：**

> 我会先profile生成、Actor重放、Reference前向和保存各自占比，而不是直接换框架。生成通常是最大瓶颈，可以做批量FSM解码、减少最大rationale Token、启用KV Cache和更高效的批处理；Reference可以离线或分批计算，但要保证参数冻结和Token约束一致。如果重新引入vLLM，我会固定兼容版本，在每次Actor同步后检查加载参数集合、抽样权重和固定canary的Token/字段一致性，再允许优化器创建。还可以采用LoRA减少训练状态、梯度检查点权衡显存与计算、多卡张量/数据并行。任何提速都必须回归old/new概率语义，不能只看tokens/s。

---

## 10. 高频项目面试题：评测、失败与系统演进

### Q16：为什么不能只看Accuracy？你最终看哪些指标？

**考察意图：** 类别不平衡和安全业务目标。

**第一人口播：**

> 安全集经常安全样本占多数，模型全部预测安全也可能有较高Accuracy，却会漏掉真正危险动作。我至少分三层报告：模型层看三字段各自Precision/Recall/F1、Harmfulness三分类混淆矩阵、三字段完全一致率和解析率；决策层把0.5/1.0或实际聚合映射成阻断，报告unsafe Precision、Recall、F1和Specificity；系统层看攻击成功率、危险工具真实执行率、正常任务utility、误拦率、反馈重规划成功率和延迟。还要固定macro labels，避免某次模型没预测0.5时宏平均分母变化，制造不存在的指标涨跌。

### Q17：你如何解释官方7B在banking很好、clean validation却误报严重？

**考察意图：** 分布迁移、标签口径和校准分析。

**第一人口播：**

> banking 87条的标签只有安全0和危险1，场景是相对明确的银行工具调用；官方7B在这里阻断F1 92.31%、59条安全零误拦。clean validation来自AgentHarm，包含大量0.5争议类，而且我们的MUR/Attack经过独立Teacher裁决，可能与官方训练口径不同；其36条benign里误拦20条，Specificity只有44.44%。所以不能简单说“模型过小”或“官方模型坏了”，更可能是领域分布、0.5边界、标签标准和聚合共同作用。正确下一步是按来源和字段做全量审计，并人工看高风险错误，而不是把两个数据集的Accuracy直接平均。

### Q18：Think说动作无害，字段却给MUR=yes、H=0.5，你认为根因是什么？

**考察意图：** 能否区分模型、协议和策略问题。

**第一人口播：**

> 这个现象至少有三种可能。第一是模型自回归生成中，前面的自然语言分析和后面的枚举字段没有逻辑约束，字段可能漂移；第二是Think说“当前动作无害但完成请求有潜在风险”，与H=0.5未必矛盾，但把合法用户标成MUR=yes可能是语义错误；第三是即使字段按模型定义成立，运行时0.4/0.4/0.1/0.2聚合会把它变成0.5并硬阻断，属于策略放大。FSM只能保证格式，不能保证语义一致。我要通过全量Think—字段盲审、金标签对照和聚合映射统计分别归因，再决定LoRA还是改策略。

### Q19：为什么不直接用更弱的Agent展示Guard成功阻断？

**考察意图：** 是否有挑样本、美化结果或实验操纵。

**第一人口播：**

> 更弱Agent更容易受攻击，但也更容易在正常日期、参数和多步规划上失败。此前Qwen2.5-7B就把2024日期推成2023，没有读到含注入的Observation，这种失败不能算Guard安全增益。反过来DeepSeek能自主识别注入，又可能让Guard没有出手机会。正式选Agent应该先在无Guard开发集上同时测正常utility、工具参数正确率和非零攻击成功样本数，选择既能完成任务又暴露安全风险的模型，再对相同identity做配对实验。Qwen2.5-14B/32B-Instruct可以作为候选，但不能为了得到漂亮阻断案例故意选择不会做任务的Agent。

### Q20：如果把项目做成企业级系统，你会如何设计？

**考察意图：** 从论文复现上升到生产安全架构。

**第一人口播：**

> 我会把系统拆成语义Guard、确定性策略引擎、工具能力注册表、授权服务和审计平台。每个工具显式声明read、internal write、external side effect、financial或destructive，以及敏感度、可逆性和是否需要确认；Guard负责结合用户目标和不可信上下文给风险信号，策略引擎按风险等级、工具能力、租户策略和用户授权做确定性决策。高影响动作采用短期capability token、参数级白名单或二次确认，Observation做来源标记和敏感信息最小化。运行时支持超时、熔断、fail-closed、重复阻断终止和人审升级；离线平台做攻击回放、误报分析、模型版本灰度和策略审计。

#### 本主题压力追问A：当前官方聚合器最危险的边界是什么？

**第一人口播：**

> 我核对代码后发现，运行时风险不是直接取Harmfulness，而是Attack和MUR各加0.4，H=0.5加0.1，H=1.0加0.2，再把总分映射成0、0.5或1。这样`yes/no/0.5`会正好得到0.5并阻断，解释了合法读取误拦；更危险的是`no/no/1.0`只得到0.2，最终映射0.0并被放行。也就是说模型已经明确判断当前Action显著有害，如果两个来源字段预测为no，聚合器仍可能漏放。下一步审计必须单独统计这种组合。生产策略至少要保证H=1.0是独立硬阻断信号，但修改前要保留官方聚合作为基线做消融。

#### 本主题压力追问B：项目目前最大的不足是什么？

**第一人口播：**

> 最大不足不是某一个模型分数低，而是端到端统计规模还不足。当前官方7B离线评测只有233条，TS-Flow只有2条Guard-in-loop和3条无Guard参考，只有一个完整配对identity；因此我只能证明链路和具体误拦机制，不能声称整体ASR下降。第二是1,656条数据仍含弱标签与严重度口径问题，虽然做过盲标裁决，但Think一致性还没全量审计。第三是Action-aware还停留在设计，ASB也刻意没有打开。我的应对不是包装成完成，而是建立明确门禁：先审计模型与聚合，再固定开发策略，最后一次性在ASB报告泛化。

### Q21：当前Agent看到的Prompt是否暗示它要警惕攻击？

**考察意图：** DeepSeek自主防御结果是否由提示词泄漏造成。

**第一人口播：**

> 我专门核对并保留了运行时使用的系统模板。正式DeepSeek轨迹采用普通`REACT_SYSTEM_PROMPT`，内容只要求模型按Thought、Action、Action Input格式使用已列出的工具，不发明工具，并在得到Observation后继续；里面没有“当前可能遭到攻击”“忽略Prompt Injection”或“优先安全”等安全提醒。Runner虽然也保留一个显式要求Agent主动调用Guardian的安全模板，但正式配置没有使用它。当前Guard是控制代码在解析到候选工具调用后强制执行，Agent无法跳过。因此DeepSeek识别日历和文件中的注入属于模型自身行为，不能归因于安全提示词暗示，也不能算Guard阻断。

### Q22：当前工具是真的执行，还是只生成一段模拟文本？

**考察意图：** Demo与真实可执行环境的区别。

**第一人口播：**

> 当前使用AgentDojo内置的workspace等任务套件，工具不是让语言模型“假装调用”，而是注册到`known_actions`的Python函数或运行时字典。Agent输出工具名和JSON参数后，代码解析并调用对应函数，返回结构化邮件、云盘文件、日历事件等Observation；发邮件、追加文件等操作会修改该次benchmark任务的隔离状态。它不是连接我个人真实邮箱或银行，因此不会影响外部生产系统，但在AgentDojo沙箱内部是可执行、有状态、可由utility/security evaluator检查的真实工具环境。轨迹中的`runtime_run_function`和`tool_execution_result`可以证明哪些调用真正执行，`tool_blocked`则证明哪些停在运行时之前。

### Q23：Guard最终只给一个评分，还是会把安全反馈真正交给Agent？

**考察意图：** Guardrail检测与反馈驱动重规划的区别。

**第一人口播：**

> Guard原始输出包含非空Think和三个字段，适配器另外计算一个兼容运行时的risk rating。在`abort`模式中，高风险动作被阻断后轨迹直接结束，主要测检测与拦截；在`ts_flow`模式中，代码要求`reason`必须是非空完整文本，然后把“Security Validation Before Execution”、Think和三个字段包装成Observation交还Agent，Agent下一轮可以改动作或停止。授权边界样本中已经真实记录6次`guardian_feedback`，DeepSeek依次尝试不同文件查询，说明反馈确实进入了上下文，而不是只在日志里打分。问题在于反馈没有提供有效恢复路径，最终形成重复误拦，这正是下一阶段要治理的系统问题。

---

## 11. 面试前必须背熟的失败案例

### 11.1 Git LFS配额耗尽

**现象：** 仓库中的数据文件只有几行指针文本，不是真实JSON；普通`git clone`无法获得对象。  
**定位：** 检查文件头、字节大小和LFS OID，确认不是解析器问题。  
**处理：** 从公开媒体端点恢复真实文件，并对照九个上游SHA-256、大小和JSON语法。  
**面试价值：** 说明可复现性不是“代码能跑”就够，还包括数据版本、文件完整性和来源可验证。

### 11.2 FlashAttention2未安装导致训练启动失败

**现象：** Transformers提示已启用FlashAttention2，但环境没有`flash_attn`包。  
**定位：** 失败发生在Actor/Reference加载，不是CUDA显存不足。  
**处理：** 训练侧显式切换PyTorch SDPA；vLLM仍使用自身推理Backend。  
**面试价值：** 能区分模型Attention实现、Python包依赖和推理引擎Backend，不建议现场编译高风险CUDA扩展来碰运气。

### 11.3 Ray命令使用了错误Python环境

**现象：** 执行虚拟环境中的`ray`，栈却混入系统Conda的`copy.py`并触发Sentinel枚举错误。  
**定位：** shebang、PATH和环境变量导致解释器/库混用，不是Ray集群状态本身。  
**处理：** 使用目标虚拟环境的Python模块入口或清理环境后执行，并通过进程和显存确认worker真正结束。  
**面试价值：** 多进程训练故障需要先确认进程树和解释器边界。

### 11.4 Teacher反复因格式失败

**现象：** 7B Teacher已经生成正确解释，却因为漏闭合Think、Markdown反引号或多行文本被严格拒绝。  
**定位：** 失败是无语义序列化约束，不是安全推理失败。  
**处理：** Teacher只负责rationale body；脚本容错提取语义并确定性拼接协议和锁定字段，批处理采用collect-and-continue、重跑只补缺失identity。  
**面试价值：** 语义生成与协议序列化应解耦。

### 11.5 API盲标出现答案锚定

**现象：** 首轮10条样本全部与旧标签一致，但Prompt末尾JSON示例写了`false/false/0.0`。  
**定位：** 没有逐样本标签泄漏，但固定示例可能诱导安全答案，10/10 agreement不可信。  
**处理：** v2 Prompt只描述字段类型和键，不出现任何具体联合答案；用balanced smoke覆盖不同来源，正式结果写入新目录。  
**面试价值：** 数据标注系统同样需要防Prompt偏置和确认性证据。

### 11.6 GRPO Rollout解析率长期低于50%

**现象：** 调整temperature、top-p、max tokens和SFT epoch后，128条rollout解析率仍约11%—40%。  
**定位：** 逐条失败主要来自Think闭合、XML样式漂移和协议Token，低温还会让组内差异消失。  
**处理：** 停止继续调采样，改用Tokenizer-aware FSM从生成时保证格式。  
**面试价值：** 当多组超参数对现象不敏感时，应重新检查问题建模而不是扩大搜索。

### 11.7 GRPO奖励提高但Think消失

**现象：** 内部validation reward提高约30.2%，banking Recall提高，但评测回答的Think覆盖率变成0。  
**定位：** Reward不要求Think存在或正确，短三字段答案更容易满分，是合法的策略捷径。  
**处理：** Teacher rationale修复、解释一致性门禁，并最终把格式交给FSM；没有可靠语义Judge时不伪造Think质量奖励。  
**面试价值：** Reward hacking不一定是恶意输出，也可能是优化器准确利用了指标盲区。

### 11.8 错误Validation切片与动态Macro指标

**现象：** 所谓AgentHarm clean validation实际是harmful文件前146条，且输出出现0.5后Macro-F1突然变化。  
**定位：** Runner先拼接再`max_samples`截断；Scikit-learn未固定labels导致宏平均分母随预测类别变化。  
**处理：** 固定identity manifest和`[0.0,0.5,1.0]`标签集合，区分supported-class、fixed-three-class和二分类指标。  
**面试价值：** 指标代码本身也是实验系统的一部分，错误评测会让正确模型决策变错。

### 11.9 DeepSeek API返回401

**现象：** 相同Key用标准HTTP成功，通过原模型适配器却稳定401。  
**定位：** 适配器按模型名分支，对非GPT/Claude/Gemini模型附加占位BasicAuth，覆盖Bearer头。  
**处理：** API路径统一到OpenAI-compatible的`api_key + base_url`契约，不把密钥写命令、日志或Git。  
**面试价值：** 先做最小对照定位网络、Key还是客户端封装，而不是盲目更换密钥。

### 11.10 Guard误拦导致反馈死循环

**现象：** 合法读取连续得到`yes/no/0.5`，6个动作被阻断，Agent不断切换搜索工具并最终放弃。  
**定位：** Think与字段存在语义张力，官方聚合器将该组合映射0.5，SecReAct又在`risk>=0.5`阻断，且缺少重复反馈终止。  
**处理方向：** 先完整审计Guard，再决定LoRA或Action-aware；策略设计包含工具能力元数据、精确重试去重和连续阻断上限。  
**面试价值：** 安全反馈如果没有可执行恢复路径，可能从“保护”变成“拒绝服务”。

---

## 12. 大厂与安全公司常见八股速查

### 12.1 Transformer与Attention

**问：Self-Attention做了什么？**  
输入经过线性映射得到Q、K、V，使用`softmax(QK^T/sqrt(d_k))V`让每个Token按内容相关性聚合其他位置的信息。除以根号维度是为了避免点积随维度增大导致softmax过饱和。多头机制在不同子空间学习关系，位置编码补充序列顺序。Decoder-only模型还使用因果Mask，保证当前位置不能看到未来Token。

**问：长上下文为什么贵？**  
标准Attention的分数矩阵随序列长度呈平方复杂度，训练还要保存激活；自回归推理通过KV Cache避免每步重算历史K/V，但Cache随层数、头数、序列和并发线性增长。Agent轨迹包含工具描述与历史Observation，Prompt很长，因此项目要限制上下文、截断无关历史并关注KV Cache。

### 12.2 Tokenizer与结构化输出

模型不是按字符生成。`<Harmfulness_Rating>`、`yes`或`0.5`可能被拆成不同Token序列，所以字符级正则不能直接成为生成时约束。Tokenizer-aware FSM先精确编码候选并构建Trie，每一步在Token ID空间限制logits。输出后正则只能检测或修复，约束解码则在概率采样前排除非法路径，两者安全性质不同。

### 12.3 Causal LM Loss与Mask

因果模型用前缀预测下一个Token，训练时通常把logits和labels错位一位计算交叉熵。Prompt不应成为回答监督目标时，labels设为`-100`；Think消融也是把对应位置mask掉。要注意mask只影响loss，不会从输入中删除Token；这些Token仍进入上下文并影响后续预测。

### 12.4 AdamW与学习率

Adam维护梯度一阶矩和二阶矩，自适应缩放更新；AdamW把权重衰减从梯度更新中解耦。全参RL通常用很小学习率，因为同一批rollout产生的优势噪声较大，而且要避免破坏SFT能力。学习率过大可能KL和梯度暴涨，过小则参数几乎不动；必须结合参数delta、KL、grad norm和外部评测判断。

### 12.5 梯度累积与Effective Batch

显存只能放micro batch 1时，可以连续多次前后向，把梯度累积后再optimizer step。Effective batch约等于micro batch × accumulation × data parallel world size。梯度累积减少显存压力但不能减少总计算，还要正确缩放loss；日志中的batch次数和global step不是同一个概念。

### 12.6 Gradient Checkpointing

它不保存部分中间激活，反向时重新计算，用额外计算换显存。不要和训练Checkpoint混淆：前者是激活重计算技术，后者是保存模型/优化器状态用于恢复。Agent长序列全参训练常用Gradient Checkpointing，但会降低速度。

### 12.7 量化

8-bit或4-bit量化降低权重显存，适合部署或QLoRA；但量化误差可能影响精细的安全边界和log-prob一致性。Qwen2.5-32B BF16权重本身约64GB，再加KV Cache和官方7B Guard无法舒适共存单张80GB，因此可选择API、两卡或4-bit部署。正式对照必须记录量化版本，不能和BF16结果无条件比较。

### 12.8 Overfitting

训练loss继续下降、validation loss上升是过拟合信号，但最终任务是自由生成与分类，不能只凭loss下结论。应当检查生成解析率、类别混淆、外部域指标和多次种子稳定性。小数据全参训练更容易过拟合，因此使用较少epoch、低学习率、外部banking和冻结ASB构成多层验证。

### 12.9 Direct/Indirect Prompt Injection与Jailbreak

Jailbreak通常绕过模型内容安全策略；Prompt Injection针对应用指令层级，让模型把不可信数据当指令。Direct Injection由用户直接输入，Indirect Injection藏在工具返回内容。Agent安全还要关注工具参数注入、跨工具数据外传、记忆污染和权限提升，防护不能只靠模型拒答。

### 12.10 Confused Deputy

Agent拥有用户授予的高权限，却被低权限攻击者提供的数据诱导，以自己的身份执行攻击者无权执行的动作，这就是混淆代理问题。间接注入发邮件、共享文件或转账本质上都是Confused Deputy。缓解措施包括能力令牌、目的绑定授权、最小权限、来源追踪和高风险Sink确认。

### 12.11 Source–Sink与Taint Tracking

Source是不可信数据入口，如邮件、网页和文件；Sink是有外部副作用的工具，如发信、支付、删除和代码执行。Taint Tracking给Source数据打标，追踪它是否影响Sink参数。LLM上下文中精确数据流追踪很难，但可以把Observation来源、目标收件人、URL和关键参数写入结构化状态，让策略在不可信数据流向高风险Sink时升级处置。

### 12.12 最小权限与Capability

不要把整套邮箱、云盘和支付权限永久交给Agent。按任务签发短期、范围受限的Capability，例如只能读取特定文件、向指定联系人发送一次邮件、转账上限固定。即使Guard漏报，运行时权限仍限制损失半径。Guard解决语义风险，Capability解决强制授权，两者互补。

### 12.13 Fail-open与Fail-closed

Guard超时或解析失败时继续执行是fail-open，可用性高但高风险；阻断动作是fail-closed，更安全但可能造成拒绝服务。项目guarded模式采用fail-closed，并记录显式`guardian_error`；生产上可按工具影响分级：普通公开搜索可降级，高影响支付/删除必须fail-closed。

### 12.14 TOCTOU

Time-of-check to time-of-use指检查时对象和执行时对象不一致。Guard审核“向A发邮件”，运行时参数若被Agent或工具层改成B，检查失效。解决方法是对规范化工具名和参数做签名或哈希，将审核结果绑定到不可变请求，执行器只接受同一对象；高风险动作还要校验资源版本和幂等键。

### 12.15 Human-in-the-loop

人工确认不是弹一个“是否继续”就安全，确认界面必须展示具体工具、目标、数据范围、金额、不可逆后果和风险来源，避免用户习惯性点击。中风险外发或不可逆动作适合升级确认；低风险查询全部弹窗会造成疲劳。离线benchmark没有真人，因此应把“需要确认”计为未自动执行，并单独统计。

---

## 13. 面试官类型与应对策略

### 13.1 大模型算法面试官

他会重点追问：loss如何算、为何GRPO、省掉Critic的代价、Reward是否可导、KL估计、采样策略、数据泄漏、类别不平衡、消融是否公平。回答时多讲公式直觉、直接父模型、固定数据和错误分布，少讲空泛业务价值。一定主动承认第一代奖励和评测口径的问题，并说明如何用FSM、manifest和独立指标纠正。

### 13.2 Agent应用/平台面试官

他会重点追问：Agent如何调用工具、参数如何解析、Guard放在哪里、异常怎么处理、模型/API如何切换、如何观测、工具是否真实执行。回答时沿第3章调用链讲，强调被阻断动作不进入runtime、完整反馈作为Observation、trace逐事件落盘、API Key不进入命令和Git。

### 13.3 安全公司面试官

他会重点追问：威胁模型是否完整、Guard是否也会被注入、如何防绕过、权限边界、误报、审计与合规。回答时主动说明模型Guard不是唯一防线，生产方案还需要最小权限、Source–Sink、参数绑定、用户确认、fail-closed和红队回放。不要把“模型输出安全标签”描述成绝对安全证明。

### 13.4 训练系统面试官

他会重点追问：显存如何估算、Checkpoint为何大、FSDP和vLLM职责、为何慢、断点恢复包含什么、如何证明更新发生。回答时引用E6真实数字：step约280秒，峰值显存约60.3GB，断点约35GB，两次独立session从step1恢复到step2，old/new parity为0。

### 13.5 压力面试官

他可能说“你只是复现论文”“指标这么低有什么价值”“为什么不直接调用大模型API”。不要防御性争辩。回答结构是：承认边界→指出自己亲自做的增量→给出证据→说明下一步可证伪实验。例如：“论文给出思想和开源基线，我的增量是恢复缺失数据/历史组件、重做盲标裁决、实现Tokenizer级FSM与同Actor训练门禁，并通过端到端轨迹发现官方聚合与utility问题；当前尚未完成全量TS-Flow统计，所以我没有宣称复现论文最终数字。”

---

## 14. 简历可用项目摘要与Bullet

### 14.1 项目简介

基于Qwen2.5、Transformers、verl/FSDP与AgentDojo构建执行前步级工具安全护栏，覆盖轨迹数据治理、SFT/GRPO对齐、Tokenizer级约束生成、可恢复训练和反馈驱动的Agent工具调用控制，并以攻击成功率、危险召回、误拦率与任务完成率联合评估安全—可用性权衡。

### 14.2 简历Bullet

- **执行前安全边界：** 针对间接Prompt Injection可能在最终文本审核前触发邮件、文件和日历副作用的问题，将Guard插入候选Action与真实工具运行时之间，保留完整交互历史与工具参数作为判断上下文；实现放行、阻断、完整反馈重规划和逐事件审计，真实API Agent轨迹验证候选动作会在进入工具运行时前接受Guard检查与处置。

- **训练数据治理：** 针对恢复数据中三字段弱标签与Teacher答案锚定问题，构建无旧标签、无split/路径暗示的API盲标与断点续标流程，对1,656条轨迹得到326条冲突并按请求级恶意性、当前动作攻击性进行裁决；重建239条SFT、1,208条GRPO和146条clean validation数据边界。

- **受约束策略学习：** 针对随机采样下格式错误吞噬GRPO语义奖励的问题，实现Tokenizer-aware Token FSM，将固定标签、闭合符和枚举协议确定性序列化，仅对Think正文、结束决策及三字段选择计算约束概率；真实3B更新中达到100%格式率和更新前old/new log-prob零误差。

- **强化学习目标审计：** 完成Qwen2.5-3B全参数SFT与GRPO闭环，GRPO相对直接父模型在未训练banking集上将Accuracy从86.21%提高到89.66%、有害类Recall从78.57%提高到96.43%；同时通过输出分布定位Think消失与误报上升，推动奖励从格式驱动改为三字段联合正确性。

- **可恢复训练工程：** 为长时全参数RL实现配置/数据/tokenizer兼容校验、双代完整Checkpoint、磁盘瞬时空间门禁及自动恢复；在两个独立会话中完成真实step1保存和step2恢复，验证参数更新、数据游标、RNG与优化器状态连续，记录峰值显存约60.3GB和单步约280秒。

- **安全与可用性评测：** 核验并接入公开7B Guard，在87条banking上取得阻断F1 92.31%，并在146条clean validation与授权边界轨迹中定位保守误报；进一步发现三字段加权聚合会放大中风险误判，建立模型语义、风险聚合和工具处置分层审计方案。

---

## 15. 不能说错的事实与危险Claim

### 15.1 可以明确说“我完成了”

- 我完成了1.5B LoRA和3B全参数SFT/第一代GRPO闭环。
- 我完成了1,656条API Teacher盲标、冲突分析和裁决数据重建。
- 我实现了Tokenizer-aware FSM、约束log-prob重放和真实3B单步更新。
- 我完成了E6两步真实训练与Checkpoint恢复。
- 我核验、部署和评测了官方7B Guard。
- 我完成了DeepSeek API Agent与真实AgentDojo工具的TS-Flow链路。
- 我通过配对轨迹定位了官方Guard误拦合法读取的utility退化。

### 15.2 必须说“当前正在做/下一步”

- 完整1,656条官方Guard能力评测。
- Think与字段全量语义一致性审计。
- Action-aware工具能力策略的代码实现。
- Qwen2.5-32B Agent正式对照。
- 官方7B LoRA调优。
- ASB最终测试。
- 企业级上线、真实用户量、线上延迟、成本下降。

### 15.3 面试中不要主动宣传

- 不要把错误AgentHarm prefix146结果叫clean validation。
- 不要只说“GRPO F1提高32个百分点”，其中包含macro类别集合变化。
- 不要说“DeepSeek攻击被Guard成功拦截”；实际是DeepSeek没有提出危险Action。
- 不要说“Action-aware已经修复误拦”；它目前只有设计。
- 不要说“官方模型在所有数据上很好”；clean validation Specificity只有44.44%。
- 不要说“数据集完全是人工金标签”；仍包含基准score与项目裁决口径。
- 不要说“完整Constrained GRPO训练完成”；只完成E6前两步。

---

## 16. 源码证据索引

| 主题 | 关键路径与符号 | 用途 |
|---|---|---|
| API盲标 | `practice/toolsafe_reproduction/sft/api_teacher_blind_label.py`：`build_blind_messages`、`parse_teacher_annotation`、`recover_parseable_rejections` | 无答案锚定Prompt、结构解析、断点恢复 |
| 标签裁决 | `practice/toolsafe_reproduction/sft/adjudicate_teacher_labels.py`：`adjudicate_conflict`、`rebuild_datasets`、`audit_outputs` | 冲突规则、数据重建、审计 |
| SFT | `practice/toolsafe_reproduction/sft/train_guardian_sft.py`：`SFTExampleCollator`、`weighted_causal_lm_loss`、`train` | label mask、加权loss、全参训练 |
| Token FSM | `practice/toolsafe_reproduction/grpo/constrained_guardian_fsm.py`：`TokenTrie`、`compile_guardian_grammar`、`GuardianTokenFSM` | Token级合法集合和状态机 |
| 约束策略 | `practice/toolsafe_reproduction/grpo/constrained_guardian_policy.py`：`generate_constrained_rollout`、`recompute_constrained_log_probs` | 生成、约束归一化、重放log-prob |
| GRPO数学 | `practice/toolsafe_reproduction/grpo/constrained_grpo_core.py`：`dense_guardian_reward`、`group_relative_advantages`、`grpo_response_loss` | Reward、Advantage、clip/KL loss |
| 数据/恢复 | `practice/toolsafe_reproduction/grpo/constrained_grpo_runtime.py`：`validate_data_boundaries`、`rotate_checkpoint`、`discover_resume_checkpoint` | 数据门禁、双代断点、恢复发现 |
| 完整Trainer | `practice/toolsafe_reproduction/grpo/constrained_grpo_trainer.py`：`build_preflight_report`、`_train_prompt_group`、`_run_validation`、`run_training` | E5/E6训练主链 |
| 官方Guard适配 | `practice/toolsafe_reproduction/src/model/constrained_guardian.py`：`aggregate_risk_rating`、`ConstrainedGuardian.tool_safety_guardian` | FSM生成、三字段聚合、trace |
| Agent控制流 | `practice/toolsafe_reproduction/src/agent/sec_react_agent.py`：`SecReAct_Agent.agent_invoke` | Guard调用、阻断、反馈、工具执行边界 |
| TS-Flow Runner | `practice/toolsafe_reproduction/runners/run_secreact_trace.py`：`TracePrinter`、`build_agent_model`、`build_guardian_model` | API/本地模型、运行配置、逐事件日志 |
| 离线评测 | `practice/toolsafe_reproduction/runners/run_tsbench_guardian.py`：`load_samples`、`compute_metrics`、`run` | TS-Bench模型评测 |
| 实验边界 | `practice/toolsafe_reproduction/grpo/config/constrained_grpo_experiment_manifest.yaml` | 数据用途、奖励、阶段状态 |
| 官方7B报告 | `practice/toolsafe_reproduction/OFFICIAL_TS_GUARD_7B_EVALUATION.md` | 权重与233条评测证据 |
| 项目历程 | `practice/toolsafe_reproduction/PROJECT_JOURNEY.md` | 决策、失败、结果和当前节点 |

---

## 17. 面试最后一小时速记

### 17.1 十个数字

```text
1,656：API Teacher盲标总量
1,330：Teacher与旧标签完全一致
326：冲突行
71：最终人工字段修正行
239：新版rationale SFT实际训练行
1,208：GRPO训练行
146：clean validation
87：external banking
5,231：冻结ASB，使用量0
约280秒：E6真实单步
```

### 17.2 十句话

1. Guard位于Action生成后、真实工具执行前。
2. 三字段分别回答恶意用户、第三方攻击和当前动作危害。
3. Token FSM保证格式，不替模型做安全语义判断。
4. GRPO用同Prompt组内相对奖励省掉Critic。
5. 同组奖励全相等时优势为0，没有区分信号。
6. Rollout与log-prob重算必须处在同一Token约束概率空间。
7. Reward提高不等于安全能力提高，Think消失就是反例。
8. Recall高但Specificity低意味着Guard过于保守。
9. DeepSeek自主拒绝注入不能算Guard成功阻断。
10. 当前先审计Guard，再决定LoRA或Action-aware。

### 17.3 现场画图顺序

```text
用户请求
→ Agent Thought/Action
→ Guard Think+三字段
→ 风险聚合/策略
→ allow: tool runtime
→ block: feedback observation
→ Agent replan
```

然后补充离线侧：

```text
轨迹 → 盲标/裁决 → SFT → FSM rollout → GRPO → validation/banking
```

### 17.4 面试卡壳时的安全回答

> 这个点我区分一下已经验证的事实和下一步设计。当前仓库与日志能证明的是……；对于您问的更大规模效果，我还没有完成ASB最终测试，所以不能给一个没有证据的百分比。我的验证方案会固定identity和模型版本，对react、严格阻断和反馈重规划做配对，再同时报告ASR与utility。

这句话不会减分。相反，它体现实验边界意识。真正减分的是先报出漂亮结果，随后说不清分母、数据来源和代码位置。

---

## 18. 可以反问面试官的问题

1. 贵团队对Agent安全更关注模型层Guard，还是工具权限、审计和工作流策略的组合？
2. 在内部Agent场景中，安全指标更偏重攻击成功率、敏感数据外传，还是正常任务误拦和人工确认成本？
3. 团队是否已经建立针对间接Prompt Injection的红队数据和可回放工具环境？
4. 高风险工具调用目前采用模型判断、规则策略、权限系统还是人工审批，几层之间如何分工？
5. Guard模型上线后如何做版本灰度、阈值校准和误报申诉闭环？

---

## 19. 面试后的继续准备建议

这份文件是“答案底稿”，不是掌握度证明。面试前至少进行一次不看文档的三分钟口述，并任选以下问题录音回答：

- 为什么step-level比final answer审核更有效？
- GRPO的优势如何算，同组全满分怎么办？
- FSM如何限制Token，为什么重算log-prob也要同样限制？
- 为什么官方7B误拦不能立即归因于模型过小？
- 当前项目哪些完成了，哪些没有？

如果某题只能复述定义、无法结合本项目文件和日志，就把它标为高风险Claim，面试中不要主动引导到该点。不要只复制粘贴，要理解每一行配置、每一个指标分母和每一条工具执行边界。

---

## 20. 交接摘要

### 交给简历优化模型的项目事实

- 项目名称：AgentSentinel，面向LLM Agent的步级工具调用安全护栏。
- 个人范围：个人研究项目，完成数据恢复与治理、1.5B/3B训练实验、API Teacher、Token FSM、Constrained GRPO门禁、官方7B评测与TS-Flow接入。
- 真实量化：1,656条盲标、326条冲突、71条裁决修改；banking最佳3B SFT Accuracy 93.10%；GRPO相对父模型有害Recall 78.57%→96.43%；官方7B banking阻断F1 92.31%。
- 真实边界：完整Guard审计、Action-aware、32B Agent、官方7B LoRA和ASB测试均未完成。

### 高风险Claim清单

- Ownership：可以说独立完成个人项目，不要暗示在企业生产环境上线。
- Metric：每个数字必须带模型分支、数据集、样本量、标签口径和直接对照。
- Architecture：FSM、同Actor、TS-Flow和Checkpoint可以讲实现；Action-aware只能讲设计。
- Result：可以说链路打通和机制问题被定位，不能说已经在大规模Agent评测中显著降低ASR。
