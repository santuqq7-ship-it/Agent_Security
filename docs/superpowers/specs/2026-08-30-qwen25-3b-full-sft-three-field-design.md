# Qwen2.5-3B 三字段全参数 SFT 设计

## 目标

使用 `Qwen2.5-3B-Instruct` 完成全参数监督微调，使模型稳定输出 ToolSafe
TS-Guard 的三个安全字段。训练只负责 SFT，不启动 GRPO。

## 数据边界

- 候选来源：`grpo_reconstructed/train.parquet`，共 1510 条已审计记录。
- 按数据来源、子集和三个目标字段做确定性分层拆分：约 20%（302 条）用于
  SFT，约 80%（1208 条）仅留给后续 GRPO；两部分 Prompt 不重叠。
- 验证来源：`grpo_reconstructed/validation.parquet`，共 146 条 clean validation。
- 外部测试：AgentDojo banking 87 条，禁止进入训练和验证。
- 沿用 Parquet 中的 TS-Guard Prompt 和 `reward_model.ground_truth`，不重新读取
  原始 JSON 进行另一套划分。拆分结果记录 source identity，便于审计零重叠。

## 三字段映射

- `Malicious_User_Request` 直接映射为 `yes/no`。
- `Prompt_Injection` 映射为输出字段 `Being_Attacked` 的 `yes/no`。
- `Harmfulness_Rating` 保留 `0.0/0.5/1.0`，不做 strict 或 loose 二值化。

监督目标严格使用当前解析器协议：

```text
<Think> <\Think>
<Malicious_User_Request> no <\Malicious_User_Request>
<Being_Attacked> yes <\Being_Attacked>
<Harmfulness_Rating> 1.0 <\Harmfulness_Rating>
```

数据没有人工标注的推理过程，因此 `<Think>` 保持为空，不合成虚构解释。

## 训练方式

- 基座：`practice/models/Qwen2.5-3B-Instruct`。
- 更新范围：全部模型参数，不创建 LoRA adapter。
- SFT 只使用 302 条分层子集；后续 GRPO 只使用剩余 1208 条。
- 精度：A800 上优先 BF16。
- 序列长度：与已审计 Parquet 的 4096-token 过滤边界一致。
- micro batch：1；使用梯度累积形成更大的有效 batch。
- 保存独立 checkpoint，不覆盖 3B 基座目录。

## 验收标准

1. SFT 与 GRPO 训练 Prompt 零重叠，且都不存在 banking；三字段标签只含合法取值。
2. 训练前脚本能读取一小批数据并完成一次前向/反向冒烟测试。
3. 正式训练由用户手动启动。
4. 训练后先检查三字段完整解析率，再比较 clean validation 与 banking 指标。
