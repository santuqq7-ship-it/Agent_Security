# Agent Security

面向 LLM 智能体工具调用安全的个人研究与工程项目。项目围绕“Agent 在真实工具执行前如何
识别越权、提示注入和潜在有害动作，并利用安全反馈重新规划”展开，覆盖数据治理、监督微调、
GRPO、Token 级约束解码、步级 Guard 评测以及反馈驱动的 Agent 控制流。

## 当前能力

- 重建并审计步级工具调用安全数据，显式隔离 SFT、GRPO、验证集和最终测试集；
- 完成 Qwen2.5-3B 全参数 SFT、GRPO 与多轮格式强化实验；
- 使用更强 Teacher API 进行三字段盲标、冲突复核与训练数据净化；
- 实现 tokenizer-aware Grammar/FSM，使 Think 与三个安全字段稳定按协议生成；
- 实现纯 Hugging Face/PyTorch constrained GRPO 原型与可恢复训练链路；
- 完成官方 7B Guard 在 AgentDojo banking 和独立 clean validation 上的评测；
- 正在接入真实 AgentDojo 工具环境，对比无护栏、检测即终止和完整反馈重规划三种模式。

## 目录

- `practice/toolsafe_reproduction/`：核心源码、训练与评测脚本、配置、测试和项目历程；
- `docs/superpowers/specs/`：阶段设计规格；
- `docs/superpowers/plans/`：实施计划；
- `practice/toolsafe_reproduction/PROJECT_JOURNEY.md`：按时间记录的实验、决策与结论；
- `practice/toolsafe_reproduction/OFFICIAL_TS_GUARD_7B_EVALUATION.md`：官方 7B Guard 评测报告；
- `智能体工具调用安全项目小结.md`：面向项目复盘与求职讲解的系统总结。

## 数据与权重

仓库不托管模型权重、训练断点、虚拟环境、完整数据集或原始生成轨迹。相应来源、构造方法、
固定版本与运行命令记录在 manifest 和项目文档中。上游研究代码也不复制进本仓库，来源见
`docs/UPSTREAM_PROVENANCE.md` 与 `practice/toolsafe_reproduction/source_manifest.json`。

## 项目状态

当前以官方 7B Guard 作为安全判断模型，使用 Token FSM 工程化保证输出协议；下一阶段先用
本地 7B Agent 完成小规模 TS-Flow 控制流验收，正式 Agent 评测再切换到能力更强的 API 模型。

详细实验指标、失败分析和路线变更请阅读
[`PROJECT_JOURNEY.md`](practice/toolsafe_reproduction/PROJECT_JOURNEY.md)。
