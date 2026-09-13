# Version Control Policy

本仓库用于保存 Agent 工具调用安全项目中可公开、可审计、可复现的工程资产。

## 提交什么

- 源码、测试、配置、依赖清单和数据构造脚本；
- 设计规格、实施计划、项目历程和评测分析；
- 不含敏感输入的小型聚合指标；
- 改变研究路线的失败实验、根因证据和决策。

## 不提交什么

- 模型权重、Adapter、优化器状态和训练 checkpoint；
- 虚拟环境、缓存、第三方 vendor 副本和嵌套上游仓库；
- 完整数据集、Parquet、原始 JSONL trace 和批量生成日志；
- API key、token、私钥、Authorization header 或 `private/` 内容；
- 论文 PDF 和可以从固定来源重新下载的大文件。

## 里程碑规则

重要节点先更新 `practice/toolsafe_reproduction/PROJECT_JOURNEY.md`，再根据内容使用：

- `feat:` 新工程能力；
- `fix:` 已定位缺陷修复；
- `eval:` 新评测结果；
- `experiment:` 训练、消融或实验门禁；
- `decision:` 技术选型或策略变更；
- `docs:` 项目说明；
- `chore:` 仓库治理。

一次提交只表达一个可复核主题。失败但推动路线变化的实验同样需要记录，不能只保留成功结论。

## 推送前门禁

1. 核对 `git status` 和 staged diff；
2. 拒绝单文件超过 20 MiB；
3. 扫描常见密钥、Bearer token 和私钥头；
4. 确认 `ToolSafe/` 未被提交为 gitlink；
5. 运行与改动对应的测试；
6. 确认文档中的完成状态与真实证据一致；
7. 提交后复核 commit，再推送 `origin/main`。

大产物只在文档中记录来源、路径、规模或摘要。远端认证失败时不得把访问令牌写入 remote URL。
