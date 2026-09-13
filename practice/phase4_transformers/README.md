# 阶段四 Task 1：Transformers 本地冒烟实验

这个目录是个人实践代码，与论文原始仓库 `ToolSafe/` 分开管理。
实验只生成候选工具调用和 logits 分析，不会连接或执行真实工具。

## 1. 进入项目根目录

```bash
cd "/Users/qixinyao.1/Desktop/Agent-Security"
```

`cd` 只改变当前终端的工作目录，便于后续使用统一的相对路径。

## 2. 运行单元测试

```bash
ToolSafe/.venv-phase4/bin/pytest practice/phase4_transformers/tests -q
```

- `ToolSafe/.venv-phase4/bin/pytest`：使用已经准备好的项目虚拟环境，避免调用系统 Python。
- `practice/phase4_transformers/tests`：只收集本实验测试，不收集原仓库中的 `scripts/api_test.py`。
- `-q`：使用简洁输出；当前应看到 `8 passed`，表示八个行为测试通过。

## 3. 运行一次本地模型推理

```bash
ToolSafe/.venv-phase4/bin/python \
  practice/phase4_transformers/phase4_transformers_smoke.py \
  --config practice/phase4_transformers/phase4_transformers_smoke.yaml
```

- `ToolSafe/.venv-phase4/bin/python`：使用同一个虚拟环境运行脚本。
- `phase4_transformers_smoke.py`：加载本地小模型、生成候选动作、计算 Token 熵并解析工具调用。
- `--config`：明确指定 YAML 配置，便于你复制配置并做自定义实验。
- 反斜杠 `\\`：表示命令在下一行继续输入，不是 Python 参数。

## 4. 保存自己的实验结果

```bash
ToolSafe/.venv-phase4/bin/python \
  practice/phase4_transformers/phase4_transformers_smoke.py \
  --config practice/phase4_transformers/phase4_transformers_smoke.yaml \
  > practice/phase4_transformers/run-01.json
```

`>` 是终端重定向符，会把标准输出保存为 JSON 文件；它不会修改 `ToolSafe/` 原始源码。

## 5. 可修改的实验参数

打开 `phase4_transformers_smoke.yaml` 后可以修改：

- `model_path`：本地模型权重目录，必须已经存在并包含 Transformers 文件。
- `device`：`auto` 自动选择 MPS，否则使用 CPU；当前机器会回退到 CPU。
- `max_new_tokens`：最多生成多少个新 Token。调大可能生成更完整的回复，但运行更慢。

修改后再次运行同一条命令，并比较以下字段：

- `response`：模型原始文本；
- `parsed_tool_name` / `parsed_tool_arguments`：是否符合 ToolSafe 的 ReAct 解析协议；
- `step_entropy_count`：实际生成步数；
- `mean_token_entropy`：平均 Token 熵；
- `first_generation_step.top_candidates`：第一步最可能的候选 Token 及概率；
- `note`：明确说明本实验没有执行工具。

请先运行测试和推理命令，再把完整终端输出贴回来。我们会逐字段分析，然后再进入 mini-Guardian。

## 6. 切换到 Qwen2.5-1.5B-Instruct

建议先下载 1.5B，而不是直接在本机尝试 7B。1.5B 的半精度权重约占 3GB，
还需要额外的运行时和 KV cache 内存；M3 Pro 36GB 统一内存可以承受这个规模。

先创建独立模型目录：

```bash
mkdir -p practice/models/Qwen2.5-1.5B-Instruct
```

当前虚拟环境已经安装 Hugging Face Hub，执行：

```bash
ToolSafe/.venv-phase4/bin/hf download \
  Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir practice/models/Qwen2.5-1.5B-Instruct
```

- `Qwen/Qwen2.5-1.5B-Instruct`：Hugging Face 上的模型标识；
- `--local-dir`：把权重保存到独立实践目录；
- 这个命令只下载模型，不会修改 `ToolSafe/src/`。

下载完成后，先检查目录：

```bash
find practice/models/Qwen2.5-1.5B-Instruct -maxdepth 1 -type f | sort
```

至少应看到模型配置、Tokenizer 文件和若干权重分片。然后运行 1.5B 配置：

```bash
ToolSafe/.venv-phase4/bin/python \
  practice/phase4_transformers/phase4_transformers_smoke.py \
  --config practice/phase4_transformers/phase4_transformers_1p5b.yaml
```

如果 Hugging Face 下载出现网络或权限错误，先把完整错误贴出来，不要重复下载或把模型文件放入 `ToolSafe/`。
