# Qwen2.5-3B GRPO Rollout 验收设计

## 目标与边界

使用三字段全参数 SFT 最终模型，在未参与 SFT 的 1208 条 GRPO 数据上
执行一次 `16 prompts × 8 rollouts` 的无更新验收。

本阶段禁止：

- 创建优化器；
- 执行 backward；
- 修改 Actor 参数；
- 启动 verl Trainer 或正式 GRPO。

## 权重完整性

现有验收脚本只支持单个 `model.safetensors`。新增通用 checkpoint 摘要：

- 单文件 checkpoint 保持现有 SHA-256 行为；
- 分片 checkpoint 读取 `model.safetensors.index.json`；
- 校验索引引用的所有分片存在且非空；
- 按固定文件名顺序，对索引和全部分片计算可复现的组合 SHA-256；
- 缺失或空分片立即失败，不跳过完整性检查。

## 3B 验收配置

- Actor：`sft_qwen25_3b_full_three_field` 最终模型；
- 数据：`sft_grpo_disjoint/grpo_train.parquet`，共 1208 条；
- Prompt：分层选择 16 条；
- 每条 Prompt：采样 8 个回答；
- 采样：`temperature=1.0`、`top_p=1.0`；
- 奖励：现有 `agentsafety_v2_uniform.py` 三字段规则奖励；
- 产物：128 个回答的 JSONL trace 和 acceptance report。

## 验收标准

- 解析率不低于 50%；
- 非零奖励率不低于 50%；
- 至少 50% Prompt 组包含两个不同奖励；
- 报告必须记录 `training_started=false`、`optimizer_created=false`、
  `backward_executed=false`。

如果组内奖励变化不足，停止在验收阶段分析奖励饱和原因，不启动训练。

## 测试

增加三类单元测试：单文件摘要、分片摘要稳定性、缺失分片失败。随后运行
现有 rollout acceptance 测试套件和一次真实 `16×8` 无更新验收。
