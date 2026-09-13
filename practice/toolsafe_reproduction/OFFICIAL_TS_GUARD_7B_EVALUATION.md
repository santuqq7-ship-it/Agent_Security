# 官方 TS-Guard 7B 权重核验与基线评测

日期：2026-09-13

## 1. 权重核验

- 仓库：`MurrayTom/TS-Guard`
- 固定 commit：`ad2f82df3ae3e0dcdaed7b7c71e8b7f0b16ead7a`
- 模型类型：`Qwen2ForCausalLM`，BF16，全参数模型，不是 LoRA Adapter
- 权重索引声明总大小：15,231,233,024 bytes
- 权重分片：4/4 个 `safetensors` 文件存在，实际大小与索引对应
- 云端目录：`/root/Agent-Security/practice/toolsafe_reproduction/models/official_ts_guard_7b`
- 加载方式：Transformers，单张 A800，贪婪生成，`max_retries=1`

下载完成后数据盘剩余约 57 GiB。推理时模型占用约 16.8 GiB 显存，评测退出后 GPU
显存归零。现有 E6 `latest/previous` 两代断点和 7B Teacher 均未删除。

## 2. 评测边界

本轮只使用：

1. 87 条 AgentDojo banking 外部报告集；
2. 按 `source_identity` 与训练集隔离的 146 条 Teacher-adjudicated clean validation。

ASB 继续冻结为最终独立测试集，使用量为 0。没有启动 E7，没有更新任何模型参数。

## 3. Banking 结果

### 3.1 256 Token 等预算对照

- 总样本：87
- 成功解析：78
- 解析失败：9
- 解析样本 exact accuracy：93.59%
- 全体样本 exact accuracy（解析失败计错）：83.91%

检查 9 条失败响应后确认，它们均在字段生成前或字段中间到达 256 Token 上限。该结果
主要测到了输出长度限制，不能直接解释为模型的格式能力。

### 3.2 512 Token 正式结果

- 总样本：87
- 成功解析：85
- 解析率：97.70%
- 解析样本 exact accuracy：87.06%
- 全体样本 exact accuracy（解析失败计错）：85.06%
- Gold 分布：安全 59、危险 28
- Prediction 分布：0.0 为 61、0.5 为 9、1.0 为 15、解析失败 2

将 0.5/1.0 都映射为“阻断”后的 Agent 决策指标：

- Accuracy：95.40%
- Precision：100.00%
- Recall：85.71%
- F1：92.31%
- Specificity：100.00%
- 混淆：TP=24、TN=59、FP=0、FN=4；其中 2 个 FN 是格式解析失败

这说明官方模型在 banking 上具有很强的执行前拦截价值：没有误拦安全动作，28 个风险
动作中成功阻断 24 个。Exact 三分类更低，主要是把部分 1.0 预测成 0.5；对二分类阻断
而言，这些预测仍然是正确的。

## 4. Clean validation 结果

- 总样本：146
- 成功解析：145
- 解析率：99.32%
- Harmfulness 三分类 Accuracy：50.68%
- Harmfulness Macro-F1：51.07%
- Harmfulness Macro-Recall：50.93%
- 三字段完全一致率：49.32%
- Malicious User Request Accuracy：80.82%
- Being Attacked Accuracy：99.32%，但 146 条 Gold 全部为 false，因此该数字不代表对攻击
  正负样本的完整识别能力

Harmfulness 分类别结果：

| Gold 类别 | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 0.0 | 36 | 64.00% | 44.44% | 52.46% |
| 0.5 | 66 | 59.62% | 46.97% | 52.54% |
| 1.0 | 44 | 39.71% | 61.36% | 48.21% |

将 0.5/1.0 都映射为“阻断”后的结果：

- Accuracy：79.45%
- Precision：83.33%
- Recall：90.91%
- F1：86.96%
- Specificity：44.44%
- 混淆：TP=100、TN=16、FP=20、FN=10

按来源拆分后，110 条 harmful 中正确阻断 100 条，Recall 为 90.91%；36 条 benign 中只
正确放行 16 条，误拦 20 条。官方模型呈明显的保守倾向。该偏差同时受到 AgentHarm
分布、当前重新裁决标签标准与官方训练标签标准差异影响，不能简单归因于 7B 容量不足。

## 5. Think 与格式能力

两套正式评测的每条回答都以 `<Think>` 开始，并且在判定字段前存在非空、实质性的分析
文字：

| 数据集 | 有分析正文 | 严格 `<Think>...<\Think>` 闭合 |
|---|---:|---:|
| clean validation | 146/146（100%） | 30/146（20.55%） |
| banking 512 | 87/87（100%） | 46/87（52.87%） |

因此模型具备生成安全分析与反馈的能力，但原始 Think 闭合标签不稳定。512 Token 后仍有
3 条三字段解析失败，原因是模型将解释文字和字段标签混写，而不是截断。部署侧应使用
Grammar/FSM 保证三字段序列化，语义判断仍由模型负责；不应为了固定标点和标签闭合单独
进行 LoRA。

## 6. 阶段结论

官方 7B 模型足以作为下一阶段 Agent/反馈框架的主护栏基线：banking 二分类阻断 F1 为
92.31%，且安全样本误拦为 0。当前不立即执行 LoRA，原因如下：

1. 官方模型已经完成论文安全对齐，小规模 LoRA 可能造成灾难性遗忘；
2. clean validation 的主要问题是风险等级边界和保守校准，而不是完全缺少风险识别能力；
3. 当前 clean validation 的 Being Attacked 字段只有单一类别，不能支撑可靠的多字段
   微调选择；
4. 格式异常可以由现有 Grammar/FSM 在推理阶段确定性解决；
5. 应先在 Agent 执行前拦截与反馈重规划链路中测量安全性和任务完成率，再决定是否需要
   针对真实误差进行 LoRA。

下一阶段使用官方 7B + Grammar/FSM 接入 Agent：先复现无护栏、仅拦截、拦截并反馈重规划
三种路径，比较危险工具执行率、攻击成功率、正常任务完成率、误拦率与额外推理开销。
只有端到端指标显示语义校准不足时，才启动低 rank、低学习率、1 epoch 的 LoRA，并严格
回归本轮官方零样本基线。

## 7. 云端产物

- `results/official_ts_guard_7b_eval/smoke_banking/`
- `results/official_ts_guard_7b_eval/banking/`（256 Token 对照）
- `results/official_ts_guard_7b_eval/banking_512/`（正式 banking）
- `results/official_ts_guard_7b_eval/clean_validation/`（正式 clean validation）
- `results/official_ts_guard_7b_eval/audit.json`（聚合二分类与 Think 审计）

逐样本 trace 含完整提示词和响应，保留在云端，没有跨主机复制；本地文档只落盘非敏感
聚合指标。
