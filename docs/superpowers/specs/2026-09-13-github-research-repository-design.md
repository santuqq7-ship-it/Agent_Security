# Agent Security 研究主仓库设计规格

日期：2026-09-13

远端：`https://github.com/santuqq7-ship-it/Agent_Security.git`

状态：待用户核对书面规格后实施

## 1. 目标

将当前项目整理为一个可公开展示、可复现、可持续维护的研究主仓库。仓库保留从数据治理、
SFT、GRPO、Grammar/FSM、官方 7B 评测到 TS-Flow 集成的重要代码、技术决策和可核验结论，
但不托管模型权重、训练断点、运行环境、原始大数据、海量轨迹或任何凭据。

后续每当出现重要实验结果、技术选型变化、训练策略变化、关键故障修复或阶段突破时，必须
先把证据与结论同步到项目文档，再形成独立 Git commit 并推送远端。普通临时调试、缓存和
无结论的重复试跑不创建里程碑 commit。

## 2. 仓库结构

项目根目录 `/Users/qixinyao.1/Desktop/Agent-Security` 初始化为新的 Git 仓库，默认分支为
`main`，远端 `origin` 指向用户新建的 GitHub 仓库。

首批纳管内容包括：

- 根目录项目说明、学习总结和项目成果文档；
- `practice/toolsafe_reproduction/` 下的源码、测试、配置、脚本与技术文档；
- `PROJECT_JOURNEY.md`、`PROJECT_MEMORY.md` 和官方 7B 评测报告；
- `docs/superpowers/specs/` 与 `docs/superpowers/plans/` 中的设计和实施计划；
- 数据 manifest、依赖清单和可复现说明；
- 少量、经过筛选且不含敏感输入的聚合评测证据。

以下内容始终排除：

- `ToolSafe/`：保留为独立上游仓库，不嵌入、不复制、不作为 submodule；
- `practice/models/`、模型权重、Adapter 权重和训练 checkpoint；
- Python/Conda 虚拟环境、缓存、日志与操作系统文件；
- `practice/toolsafe_reproduction/private/` 及所有 API 配置；
- 训练/评测原始 JSONL trace、完整生成内容和可重建的大数据文件；
- PDF 论文和其他非源码研究输入。

官方代码来源、commit 与数据来源通过 `source_manifest.json` 和文档固定。个人修改优先放在
`practice/toolsafe_reproduction/src/` overlay 内，避免直接改写并重新发布上游仓库。

## 3. 证据保存规则

研究仓库不能只有结论，也不能把所有原始产物全部上传。采用“聚合证据 + 可复现命令 +
外部大产物路径”的三级记录：

1. 文档记录实验目的、固定输入、模型版本、关键参数、指标和结论；
2. 小型、结构化且不含隐私的指标 JSON 可以进入专门的 evidence 目录；
3. 权重、断点和完整 trace 只记录生成命令、哈希或路径，不进入 Git。

现有结果目录默认继续整体忽略。需要长期保留的指标先经过字段审计，再复制到轻量 evidence
目录，避免通过复杂的 `.gitignore` 例外规则意外纳入同目录大文件。

## 4. 里程碑提交规范

每个 commit 只表达一个可复核主题，使用以下前缀：

- `feat:` 新增工程能力；
- `fix:` 修复已定位的缺陷；
- `eval:` 新的评测证据与分析；
- `experiment:` 完成训练、消融或实验门禁；
- `decision:` 记录技术选型或策略变更；
- `docs:` 更新说明、总结或面试材料；
- `chore:` 仓库治理与不改变实验语义的维护。

重要代码变更与对应测试尽量在同一 commit。重大实验不得只提交结果而不记录使用的配置、
模型身份和数据边界。失败但改变后续路线的实验也应提交，明确写出“失败现象、根因证据、
放弃或修正理由”，避免只保留成功故事。

## 5. 安全与质量门禁

每次提交前执行：

1. `git status` 与 staged diff 核对；
2. 大文件扫描，拒绝单文件超过 20 MiB；
3. 常见 API key、Bearer token、私钥头和凭据文件名扫描；
4. 检查嵌套 Git 仓库没有被暂存为 gitlink；
5. 对受影响模块运行聚焦测试；重大里程碑运行完整可行测试集；
6. 核对文档中的“已完成”与真实产物一致，不把计划写成已完成事实；
7. 提交后检查 commit 内容，再推送 `origin/main`。

如果密钥曾被暂存或提交，不能仅靠删除文件解决，必须停止推送、撤销提交并轮换密钥。远端
推送失败时保留本地 commit，报告认证或网络错误，不通过把 token 写入 remote URL 来绕过。

## 6. 首次同步拆分

首次同步不使用一个无法审阅的巨型 commit，而拆分为以下里程碑：

1. `chore: initialize research repository governance`：根仓库、`.gitignore`、治理说明；
2. `docs: record project roadmap and completed milestones`：现有历程、评测总结与学习材料；
3. `feat: add reproducible guardian training and evaluation pipeline`：SFT/GRPO/FSM/runner 源码、
   配置、脚本与测试；
4. `docs: add ts-flow agent smoke-test design`：当前已确认方向的设计规格。

每次提交之间都核对 staged 文件清单和总大小。若远端仓库不是空仓库，则先只读拉取远端
分支信息并合并其 README/license 历史，不强推、不覆盖远端提交。

## 7. 后续自动遵循的项目规则

仓库新增 `VERSION_CONTROL_POLICY.md`，把本规格中稳定的规则放在项目根目录。此后每个阶段：

1. 实施前记录设计或决策；
2. 实施后运行验证；
3. 将指标、异常和结论写入 `PROJECT_JOURNEY.md`；
4. 生成语义清晰的独立 commit；
5. 推送至公开 GitHub 仓库；
6. 在交付消息中给出 commit hash，并说明哪些大产物只保存在本地或云端。

“后续自动遵循”表示在用户继续委托本项目工作时，把 Git 同步视作阶段交付的一部分；不表示
创建后台定时任务，也不在没有新里程碑时产生空提交。

## 8. 首次同步验收

- `origin` 精确指向 `https://github.com/santuqq7-ship-it/Agent_Security.git`；
- 默认分支是 `main`；
- 远端可见上述分阶段 commit；
- Git 对象与工作树中不存在权重、断点、虚拟环境、private 目录或 PDF；
- 不存在超过 20 MiB 的已跟踪文件；
- 不存在已知 API key、Bearer token 或私钥；
- `ToolSafe/` 未作为嵌套仓库或 gitlink 提交；
- 关键源码、配置、测试、项目历程和两份设计规格均已纳管；
- 本地未跟踪的大文件保持原状，不因 Git 初始化而删除或移动。
