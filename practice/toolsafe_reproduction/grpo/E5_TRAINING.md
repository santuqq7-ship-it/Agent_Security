# ToolSafe Constrained GRPO v3：E5/E6/E7 操作手册

## 1. 当前阶段边界

E5 已实现“同一 Hugging Face Actor + Token FSM”的多步 GRPO runner、完整断点、
clean validation 选模和审计日志。E5 的真实数据 preflight 不加载模型、不创建优化器、
不执行反向传播，也不产生 3B 权重。

E6 已于 2026-09-13 按“方式1”通过：fresh run 完成 step 1 并写入完整断点，第二个独立
进程从该断点恢复至 step 2。当前 `latest=step 2/sample_cursor 8`，
`previous=step 1/sample_cursor 4`，两步共32条rollout。E7由用户手动去掉
`--max-steps`，从step 2继续完成302步正式训练。banking仍只作外部报告；ASB在最终
checkpoint冻结前禁止访问。

## 2. 唯一配置文件

配置文件：`grpo/config/a800_constrained_grpo_e5.yaml`。不要复制出多份相似 YAML；后续
只在这一份文件中修改实验参数，并让 preflight 重新生成语义哈希。

| 字段 | 当前值 | 含义与修改后果 |
|---|---:|---|
| `schema_version` | `1` | 配置协议版本；不匹配时启动即失败。 |
| `project_root` | `/root/Agent-Security` | 云端项目入口；解析后可能显示为其真实挂载路径 `/cloud/.../workspace`。 |
| `actor_model` | `.../policy_init` | fresh run 的 FP32 Actor 起点。 |
| `reference_model` | `.../reference` | 冻结 BF16 Reference，只计算 KL，不随训练更新。 |
| `manifest` | `...experiment_manifest.yaml` | 数据边界、FSM 版本和实验规则的机器可读清单。 |
| `train_file` | `.../grpo_train.parquet` | 唯一训练集，1208 条。 |
| `clean_validation_file` | `.../clean_validation.jsonl` | 唯一选模集，146 条；不反向传播。 |
| `output_root` | `.../constrained_grpo_v3_e5` | 日志、断点和 best Actor 的唯一输出目录。 |
| `epochs` | `1` | 每条训练 prompt 使用一次；当前共 302 个 optimizer step。 |
| `rollouts_per_prompt` | `4` | 同一 prompt 随机生成 4 个回答，只在这 4 个回答内计算相对 advantage。 |
| `prompt_groups_per_step` | `4` | 累积 4 个 prompt group 后更新一次，即每步 16 个回答。 |
| `temperature` | `0.8` | rollout 随机性；越高回答差异通常越大，但语义稳定性可能下降。 |
| `top_p` | `0.95` | nucleus sampling 阈值；只作用于 FSM 当前允许的 Token。 |
| `min_rationale_content_tokens` | `8` | Think 至少生成 8 个正文 Token 才允许自主结束。 |
| `max_rationale_tokens` | `192` | Think 上限；达到上限由 FSM 强制闭合并记录。 |
| `learning_rate` | `1e-6` | AdamW 每步更新尺度。 |
| `weight_decay` | `0.0` | 显式关闭与奖励无关的权重衰减，避免零 advantage 时参数仍因默认值漂移。 |
| `ppo_clip_ratio` | `0.2` | 将概率比限制在约 `[0.8, 1.2]`，抑制单步过大策略更新。 |
| `kl_coefficient` | `0.001` | 新 Actor 偏离冻结 Reference 的惩罚强度。 |
| `max_grad_norm` | `1.0` | 全局梯度裁剪阈值；日志记录的是裁剪前范数。 |
| `seed` | `20260909` | 数据顺序和采样随机性的起点；断点同时保存 CUDA RNG。 |
| `checkpoint_every_steps` | `10` | 每 10 步保存一次完整可恢复断点。 |
| `validate_every_steps` | `25` | 每 25 步及完整训练最终步评测全部 146 条 clean validation。 |
| `checkpoint_generations` | `2` | 只保留 `latest` 与 `previous` 两代完整断点。当前实现要求固定为 2。 |
| `minimum_free_gb_before_checkpoint` | `40` | 保存前磁盘余量下限，低于它会在旧 `latest` 仍完好时停止。 |
| `max_consecutive_zero_signal_steps` | `10` | 连续 10 步没有任何变奖励 prompt group 时停止烧卡；计数进入断点。 |
| `old_new_parity_tolerance` | `1e-5` | 更新前 new/old 全序列概率重算的最大允许误差。 |
| `resume` | `auto` | 默认自动选择最高的完整兼容断点。 |
| `max_steps` | `null` | 正式训练不设总步上限；E6 用命令行临时覆盖为 1、2。 |
| `terminal_verbosity` | `concise` | 终端只输出每组奖励和每步摘要；完整 prompt/response 写入 JSONL。 |

`resume`、`max_steps` 和终端详细度不进入语义配置哈希，因此 E6 的有界断点可以被 E7
继续使用。学习率、采样参数、奖励相关参数、数据路径等进入哈希，修改后旧断点会被拒绝，
防止把不同实验静默拼接在一起。

## 3. 三类存储产物

- `checkpoints/latest`：最新完整 Actor + AdamW + scheduler + RNG + 游标，约 36 GB。
- `checkpoints/previous`：上一完整断点，约 36 GB。写入新断点时旧 `latest` 始终先保留。
- `best_actor`：clean validation 最优的 Actor + tokenizer，不含优化器，约 12 GB。

`train_metrics.jsonl` 每个 optimizer step 一行；`validation_metrics.jsonl` 每次完整
validation 一行；`rollouts.jsonl` 保存可审计的 prompt、response、三字段、reward 和
advantage。日志采用追加写，每次进程有独立 `session_id`；如果中断导致最近几步没有
checkpoint，恢复后相同步号会出现在新 session 中，而不会伪装成旧进程的延续。

## 4. E5 无模型 preflight

```bash
cd /root/Agent-Security

PYTHONDONTWRITEBYTECODE=1 \
/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-grpo/bin/python \
practice/toolsafe_reproduction/grpo/constrained_grpo_trainer.py \
  --config practice/toolsafe_reproduction/grpo/config/a800_constrained_grpo_e5.yaml \
  --preflight-only \
  --preflight-report practice/toolsafe_reproduction/results/constrained_grpo_e5/e5_preflight.json
```

- `PYTHONDONTWRITEBYTECODE=1`：不在项目目录产生 `__pycache__`。
- `--config`：读取唯一正式 YAML。
- `--preflight-only`：到模型加载前就退出，绝不会训练。
- `--preflight-report`：把核验结果保存成小型 JSON，便于复盘。

通过标准为：1208/146 条、302 步、每步 16 rollout、split identity 交集为 0、
banking/ASB 行数为 0，且 `formal_training_started/model_loaded/optimizer_created` 全为
`false`。

## 5. E6 有界真实门禁

第一次只完成 step 1，并生成一个完整 `latest`：

```bash
cd /root/Agent-Security

PYTHONDONTWRITEBYTECODE=1 \
/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-grpo/bin/python \
practice/toolsafe_reproduction/grpo/constrained_grpo_trainer.py \
  --config practice/toolsafe_reproduction/grpo/config/a800_constrained_grpo_e5.yaml \
  --resume none \
  --max-steps 1
```

第二次必须从 step 1 恢复并只推进到 step 2：

```bash
cd /root/Agent-Security

PYTHONDONTWRITEBYTECODE=1 \
/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-grpo/bin/python \
practice/toolsafe_reproduction/grpo/constrained_grpo_trainer.py \
  --config practice/toolsafe_reproduction/grpo/config/a800_constrained_grpo_e5.yaml \
  --resume auto \
  --max-steps 2
```

这里的 `--max-steps` 是“训练从 step 0 起最多走到哪个全局步”，不是“本次再跑几步”。
第二条命令写 `2`，因此从 step 1 只再执行一步。E6 会真实更新参数并各写一个完整断点，
但不触发 step 25 的 clean validation。

E6 只在以下证据全部成立时通过：step 1/2 各有 4 个 prompt group 和 16 个 rollout；
old/new parity 不超过 `1e-5`；loss/梯度均有限；`latest` 为 step 2、`previous` 为 step 1；
第二次日志明确显示从 step 1 恢复；磁盘与显存仍安全。完整训练耗时只能依据这两个真实
step 的秒数估算，不能继续沿用旧 verl/vLLM 训练速度。

### 5.1 E6实测验收结果（2026-09-13）

| 指标 | step 1 | step 2 |
|---|---:|---:|
| 独立session | `20260913T091602.049605Z` | `20260913T092251.225913Z` |
| sample cursor | 4 | 8 |
| prompt groups / rollouts | 4 / 16 | 4 / 16 |
| mean reward | 0.83125 | 0.83750 |
| variable-reward groups | 1/4 | 2/4 |
| strict format rate | 100% | 100% |
| constraint violations | 0 | 0 |
| forced rationale close | 0% | 0% |
| old/new最大误差 | 0.0 | 0.0 |
| total loss | `5.8687e-7` | `6.8377e-7` |
| 裁剪前梯度范数 | 2.2018 | 3.5734 |
| optimizer step耗时 | 287.1秒 | 272.7秒 |
| 峰值allocated/reserved | 54.18/56.89GiB | 60.34/62.07GiB |

两个断点的协议、配置、数据、manifest和tokenizer摘要完全相同；两个进程session ID不同，
而状态连续从`global_step/sample_cursor=1/4`推进到`2/8`。`train_metrics.jsonl`为2行，
`rollouts.jsonl`为32行，且没有残留`incoming_step_*`目录。验收报告保存为
`results/constrained_grpo_e6/e6_acceptance.json`。

两步平均每个optimizer step约279.9秒。剩余300步仅训练计算约23.33小时；加上约30次
checkpoint写入、每25步及最终步的146条clean validation，E7总墙钟时间暂估25–30小时。
这只是基于前两步的工程预算，首次step 25 validation后应再用真实时间修正。

E6后两代完整断点各约35GB，数据盘剩余约71GB。按当前轮换顺序，下一次写35GB
`incoming`时瞬时余量约36GB；生成约12GB `best_actor`后，后续checkpoint瞬时余量预计
约24GB。容量仍能覆盖已知产物，但E7期间禁止下载或生成其他大权重，并应在每次validation
后核对磁盘。配置中的40GB门禁在写入前检查，不代表写入过程始终保留40GB。

## 6. E7 用户手动继续完整训练

E6 验收通过后，去掉 `--max-steps` 即从 step 2 继续至 step 302：

```bash
cd /root/Agent-Security

PYTHONDONTWRITEBYTECODE=1 \
/cloud/cloud-ssd1/Agent-Security/envs/toolsafe-grpo/bin/python \
practice/toolsafe_reproduction/grpo/constrained_grpo_trainer.py \
  --config practice/toolsafe_reproduction/grpo/config/a800_constrained_grpo_e5.yaml \
  --resume auto
```

`Ctrl+C` 可以安全终止，但正在计算、尚未进入完整 checkpoint 的 optimizer step 会丢失；
最多还可能丢失距上一个保存点之后的若干已完成步。重新运行同一 E7 命令即可从最高完整
兼容断点恢复。不要手工把半写入的 `incoming_step_*` 改名为 `latest`。

不要只复制粘贴，要理解每一行配置的含义：尤其是 `n=4` 的组内 advantage、每步 4 组
的梯度累积、`max_steps` 的全局上限语义，以及 `latest/previous/best_actor` 的不同用途。
