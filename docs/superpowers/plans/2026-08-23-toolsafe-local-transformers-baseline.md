# ToolSafe Local Transformers Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变 ToolSafe 原始 Agent、Guardian、环境和评测语义的前提下，用本地 Qwen2.5-1.5B Transformers 跑通真实 Guardian 评测和 SecReAct 端到端基线。

**Architecture:** 保留 `ToolSafe/src` 中未修改的任务、数据、parser、评测和环境模块，在 `practice/toolsafe_reproduction/src` 建立一个最小覆盖层。覆盖层只复制需要兼容 MPS/Transformers 的 `model.py`、`Agent_Core`、`SecReAct_Agent` 和评测入口，并通过 `PYTHONPATH` 优先加载覆盖层、回退到原始 `ToolSafe/src`。Guardian 新增 `model_type="analysis"` 分支，继续使用原始 Guardian Prompt、parser 和风险等级。

**Tech Stack:** Python 3.12；PyTorch 2.13；Transformers 5.15；Qwen2.5-1.5B-Instruct；Apple MPS/CPU；PyYAML；pytest；ToolSafe 原始 AgentDojo、ASB、AgentHarm 和 TS-Bench 数据。

## Global Constraints

- 本机不实现 vLLM、CUDA 张量并行或云端部署；这些内容放到后续 Linux NVIDIA 阶段。
- Agent 和 Guardian 当前都使用本地 Transformers `analysis` 配置。
- 原始 ToolSafe 目录只读；自定义兼容代码放在 `practice/toolsafe_reproduction/`。
- 保留原始 Agent prompt、消息循环、工具 parser、Guardian parser、`runtime.run_function(...)` 和真实评测数据。
- 不使用规则 Guardian、假工具、手工风险判断或放宽原始 parser 的方式通过测试。
- 允许减少 suite、task、样本、`max_turns`、`max_new_tokens`、batch、rollout 和 epoch，但必须记录缩减值。
- 当前仓库缺少 `TS-Guard/verl-main`；训练实现不在本计划中，后续单独核验训练源并制定训练计划。
- 用户亲自在终端执行所有实验命令，并贴出完整输出后再进入下一步。

---

### Task 1: 固定源码版本与本地依赖边界

**Files:**
- Create: `practice/toolsafe_reproduction/source_manifest.json`
- Create: `practice/toolsafe_reproduction/README.md`
- Read: `ToolSafe/src/model/model.py`
- Read: `ToolSafe/src/agent/agent.py`
- Read: `ToolSafe/src/agent/sec_react_agent.py`
- Read: `ToolSafe/src/guardian_evaluator/*.py`

**Interfaces:**
- Consumes: 当前 `ToolSafe` Git checkout、`ToolSafe/.venv-phase4`、1.5B 模型目录。
- Produces: 源码 commit、依赖版本、缺失 `TS-Guard/verl-main` 的可审计记录，以及后续覆盖层使用的绝对路径。

- [ ] **Step 1: 记录 ToolSafe commit 和训练目录状态**

```bash
cd "/Users/qixinyao.1/Desktop/Agent-Security/ToolSafe"
git rev-parse HEAD
test -d TS-Guard/verl-main && echo "TS-Guard training tree present" || echo "TS-Guard training tree missing"
```

`git rev-parse HEAD` 固定源码版本；第二行确认 README 引用的训练目录是否真实存在。

- [ ] **Step 2: 记录本地运行时版本**

```bash
cd "/Users/qixinyao.1/Desktop/Agent-Security"
ToolSafe/.venv-phase4/bin/python -c "import torch, transformers, yaml; print('torch=', torch.__version__); print('transformers=', transformers.__version__); print('mps_built=', torch.backends.mps.is_built()); print('mps_available=', torch.backends.mps.is_available()); print('yaml=', yaml.__version__)"
```

记录版本是为了区分模型行为问题和环境问题，不要使用系统 `python` 替代这条命令。

- [ ] **Step 3: 安装原始 API 导入所需的纯 Python 依赖**

```bash
cd "/Users/qixinyao.1/Desktop/Agent-Security"
ToolSafe/.venv-phase4/bin/pip install "openai>=1.59.7" "pydantic[email]>=2.7.1" "python-dotenv>=1.0.1" "rich>=13.7.0"
```

这些依赖只解决原始模块的导入边界；本机不安装 vLLM。若安装命令失败，保留完整错误，不继续复制代码。

- [ ] **Step 4: 写入 manifest 并检查练习目录不在原仓库**

`source_manifest.json` 至少记录：`tool_safe_commit`、`model_path`、`python_executable`、`torch_version`、`transformers_version`、`mps_available`、`training_tree_present`。验证：

```bash
cd "/Users/qixinyao.1/Desktop/Agent-Security"
test ! -e ToolSafe/practice && echo "practice overlay is outside ToolSafe"
```

- [ ] **Step 5: 用户贴出 Step 1-4 的完整输出，确认后再开始覆盖层实现**

---

### Task 2: 建立原始代码覆盖层与 Transformers Guardian 接口

**Files:**
- Create: `practice/toolsafe_reproduction/src/model/model.py`
- Create: `practice/toolsafe_reproduction/src/agent/agent.py`
- Create: `practice/toolsafe_reproduction/src/agent/sec_react_agent.py`
- Create: `practice/toolsafe_reproduction/src/model/__init__.py`
- Create: `practice/toolsafe_reproduction/src/agent/__init__.py`
- Create: `practice/toolsafe_reproduction/src/guardian_evaluator/__init__.py`
- Create: `practice/toolsafe_reproduction/tests/test_transformers_adapters.py`
- Read: `ToolSafe/src/utils/guardian_parser.py`
- Read: `ToolSafe/src/agent/agent_prompts.py`

**Interfaces:**
- `Model(model_name, model_path, model_type="analysis", api_base="", api_key="")` keeps the original constructor.
- `Guardian(model_name="TS-Guard", model_path=..., model_type="analysis")` keeps the original constructor.
- `Guardian.get_judgment_res(meta_info, max_turn=3) -> dict` keeps `risk rating`, `results`, `reason` keys.
- `Guardian.call_tool("tool_safety_guardian", arguments) -> dict` keeps the original SecReAct boundary.
- Unmodified imports fall back to `ToolSafe/src` through `PYTHONPATH`.

- [ ] **Step 1: Write the adapter contract tests first**

Tests must assert:

```python
def test_analysis_model_constructor_does_not_require_vllm():
    """The local Transformers path can import without CUDA/vLLM."""

def test_guardian_analysis_response_uses_original_parser(monkeypatch):
    """A valid TS-Guard XML response becomes risk rating plus parsed fields."""

def test_guardian_invalid_response_retries_without_rule_based_fallback():
    """Invalid model text remains invalid after the configured retry count."""
```

The tests may use a fake tokenizer/model only to test the adapter boundary; they must not replace Guardian logic with a risk rule.

- [ ] **Step 2: Copy only the original files required by the import boundary**

```bash
cd "/Users/qixinyao.1/Desktop/Agent-Security"
mkdir -p practice/toolsafe_reproduction/src/model practice/toolsafe_reproduction/src/agent practice/toolsafe_reproduction/tests
cp ToolSafe/src/model/model.py practice/toolsafe_reproduction/src/model/model.py
cp ToolSafe/src/agent/agent.py practice/toolsafe_reproduction/src/agent/agent.py
cp ToolSafe/src/agent/sec_react_agent.py practice/toolsafe_reproduction/src/agent/sec_react_agent.py
touch practice/toolsafe_reproduction/src/model/__init__.py practice/toolsafe_reproduction/src/agent/__init__.py
```

After copying, add a source header containing the original absolute path and `git rev-parse HEAD`. Do not edit the original copies.

- [ ] **Step 3: Make vLLM imports lazy in the copied files**

The copied files must use this import boundary:

```python
try:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
except ImportError:  # Local MPS/CPU mode does not need vLLM.
    LLM = None
    SamplingParams = None
```

The API and vLLM branches must raise a clear `RuntimeError` only when selected without their dependency. The `analysis` branch must remain importable.

- [ ] **Step 4: Add the Guardian Transformers branch**

The copied `Guardian.__init__` must load the same local model style as the Agent when `model_type == "analysis"`:

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
self.llm = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=True,
    dtype=torch.float16 if device == "mps" else torch.float32,
)
self.llm.to(device)
self.llm.eval()
```

The branch must add a single internal text-generation helper used by `get_judgment_res`, `tool_safety_guardian`, and `alignment_check`. It must preserve the original `GUARD_TEMPLATES`, retry count, `guardian_paser_map`, and return schema. No risk score may be assigned by Python rules.

- [ ] **Step 5: Run adapter tests and inspect the failure before fixing it**

```bash
cd "/Users/qixinyao.1/Desktop/Agent-Security"
ToolSafe/.venv-phase4/bin/pytest practice/toolsafe_reproduction/tests/test_transformers_adapters.py -q
```

The first run must be used to identify implementation errors; do not skip the test output.

- [ ] **Step 6: Run adapter tests again after the minimal implementation**

Expected result is all adapter tests passing, with no import of vLLM on the MPS path.

---

### Task 3: Run the original Guardian evaluator on a TS-Bench subset

**Files:**
- Create: `practice/toolsafe_reproduction/runners/run_guardian_experiment.py`
- Create: `practice/toolsafe_reproduction/config/guardian_agentharm_1p5b.yaml`
- Create: `practice/toolsafe_reproduction/src/guardian_evaluator/agentharm.py`
- Create: `practice/toolsafe_reproduction/tests/test_guardian_subset.py`
- Read: `ToolSafe/src/guardian_experiment.py`
- Read: `ToolSafe/src/guardian_evaluator/agentharm.py`
- Read: `ToolSafe/TS-Bench/agentharm-traj/*.json`

**Interfaces:**
- Runner arguments remain `--config PATH`, matching original `guardian_experiment.py`.
- Config uses `model.name: TS-Guard`, `model.type: analysis`, and the downloaded 1.5B path.
- Config adds only `task.max_samples: 2` to bound runtime; the copied evaluator slices the real dataset before the original loop and keeps all original sample fields.
- Output remains `meta_data.json`, `preds.json`, `labels.json`, and `metrics_strict.json`.

- [ ] **Step 1: Write the subset-boundary test**

```python
def test_agentharm_processor_limits_real_dataset_without_changing_sample_fields(tmp_path):
    """A two-sample run uses original samples and preserves instruction/history/action/env_info."""
```

- [ ] **Step 2: Copy the original runner and AgentHarm evaluator**

Copy the files without changing their prompt, parser, label, or save logic. Add `src/guardian_evaluator/__init__.py` so the overlay package is importable. In the copied runner, change only `build_guard_model` so both `analysis` and `local` construct the local `Guardian` branch. The only evaluator change is accepting `max_samples` and iterating over `data[len(meta_data):len(meta_data) + max_samples]`.

- [ ] **Step 3: Create the local Guardian config**

```yaml
experiment:
  inference_mode: true
  score_mode: strict
  output_root: ../practice/toolsafe_reproduction/results/guardian_agentharm_1p5b

model:
  type: analysis
  name: TS-Guard
  path: /Users/qixinyao.1/Desktop/Agent-Security/practice/models/Qwen2.5-1.5B-Instruct

task:
  name: agentharm
  max_samples: 2
```

- [ ] **Step 4: Run the real Guardian evaluator from the ToolSafe working directory**

```bash
cd "/Users/qixinyao.1/Desktop/Agent-Security/ToolSafe"
PYTHONPATH="../practice/toolsafe_reproduction/src:$PWD/src" \
  .venv-phase4/bin/python \
  ../practice/toolsafe_reproduction/runners/run_guardian_experiment.py \
  --config ../practice/toolsafe_reproduction/config/guardian_agentharm_1p5b.yaml
```

The working directory stays `ToolSafe` because the original evaluator resolves `./TS-Bench/...` relative to it.

- [ ] **Step 5: Inspect and explain the generated artifacts**

```bash
find ../practice/toolsafe_reproduction/results/guardian_agentharm_1p5b -type f -maxdepth 2 -print | sort
```

Read `meta_data.json` and compare `guard_res.reason`, `guard_res.results`, `guard_res.risk rating`, and `labels.json`. A malformed 1.5B response must be visible as a parser failure, not silently converted by a rule.

---

### Task 4: Run SecReAct through the original AgentDojo environment

**Files:**
- Create: `practice/toolsafe_reproduction/runners/run_main_experiment.py`
- Create: `practice/toolsafe_reproduction/config/agentdojo_secreact_1p5b.yaml`
- Create: `practice/toolsafe_reproduction/tests/test_secreact_config.py`
- Read: `ToolSafe/src/main_experiment.py`
- Read: `ToolSafe/src/task_executor/agentdojo_exec.py`
- Read: `ToolSafe/src/task_executor/agentdojo/agent_pipeline/agent_pipeline.py`
- Read: `ToolSafe/src/agent/agent_prompts.py`

**Interfaces:**
- Runner arguments remain `--config PATH`, matching original `main_experiment.py`.
- Agent config uses `model.type: analysis`, `agent.type: sec_react`, and `system_prompt_template: SEC_REACT_SYSTEM_PROMPT`.
- Guardian config uses `type: analysis`, `model_name: TS-Guard`, and the same 1.5B checkpoint.
- Task config uses one real suite, one user task, and `attack_type: tool_knowledge`.

- [ ] **Step 1: Write configuration tests**

```python
def test_secreact_config_selects_transformers_agent_and_guardian():
    """Both model branches are analysis and the agent is the original SecReAct type."""

def test_secreact_config_bounds_agentdojo_to_one_task():
    """The reduced run changes quantity only, not the AgentDojo execution path."""
```

- [ ] **Step 2: Copy the original runner and create the bounded config**

The copied runner must continue calling the original `build_agent`, `AgentdojoProcessor`, `AgentPipeline`, attack loader, environment runtime, and result writer. Do not replace them with a custom loop.

- [ ] **Step 3: Run one real AgentDojo task**

```bash
cd "/Users/qixinyao.1/Desktop/Agent-Security/ToolSafe"
PYTHONPATH="../practice/toolsafe_reproduction/src:$PWD/src" \
  .venv-phase4/bin/python \
  ../practice/toolsafe_reproduction/runners/run_main_experiment.py \
  --config ../practice/toolsafe_reproduction/config/agentdojo_secreact_1p5b.yaml
```

- [ ] **Step 4: Verify the tool-execution boundary**

Inspect `meta_data.json` and the AgentDojo log. Identify, in order, the model output, parsed `tool_name/tool_params`, Guardian request, Guardian feedback, and the call to the simulated runtime. A high-risk action must have feedback and must not reach `runtime.run_function(...)`.

---

### Task 5: Compare ReAct and SecReAct on the same real task

**Files:**
- Create: `practice/toolsafe_reproduction/config/agentdojo_react_1p5b.yaml`
- Create: `practice/toolsafe_reproduction/results/README.md`
- Create: `practice/toolsafe_reproduction/tests/test_comparison_artifacts.py`

**Interfaces:**
- Both configs use the same model, suite, task, attack type, max turns, and output root family.
- Only `agent.type` changes between `react` and `sec_react`.
- Comparison reads original result JSON, not a reimplementation of utility/security scoring.

- [ ] **Step 1: Write the comparison artifact test**

```python
def test_react_and_secreact_results_are_comparable():
    """The comparison requires the same task identifiers and exposes utility/security separately."""
```

- [ ] **Step 2: Run the ordinary ReAct baseline**

```bash
cd "/Users/qixinyao.1/Desktop/Agent-Security/ToolSafe"
PYTHONPATH="../practice/toolsafe_reproduction/src:$PWD/src" \
  .venv-phase4/bin/python \
  ../practice/toolsafe_reproduction/runners/run_main_experiment.py \
  --config ../practice/toolsafe_reproduction/config/agentdojo_react_1p5b.yaml
```

- [ ] **Step 3: Read both result trees and record the explanation**

Record for each run: tool calls, Guardian feedback presence, whether the runtime executed, utility result, security result, and any format errors. Do not convert `security` into defense rate without stating the transformation.

---

### Task 6: Training handoff without pretending the training tree exists

**Files:**
- Create: `practice/toolsafe_reproduction/training/README.md`
- Read: `ToolSafe/TS-Bench/`
- Read: paper PDF training and reward sections

**Interfaces:**
- Produces a training-readiness report, not a fake checkpoint or fake GRPO result.

- [ ] **Step 1: Record missing training source and current baseline outputs**

The report must state the exact ToolSafe commit, missing `TS-Guard/verl-main` path, baseline model path, dataset paths, and the output artifact locations from Tasks 3-5.

- [ ] **Step 2: Map future training inputs without training yet**

Document the exact TS-Bench fields that will become prompt context and target/reward signals. Keep SFT and GRPO as separate future experiments; do not silently substitute a rule classifier.

- [ ] **Step 3: Stop and create a separate training design before acquiring or writing training code**

The later training plan will specify data construction, SFT loss, GRPO reward functions, group size, KL coefficient, clipping, checkpointing, and the smallest truthful cloud/local run.

---

## Verification Checklist

- [ ] User ran all commands and pasted complete outputs.
- [ ] No new source code was added under `ToolSafe/src`.
- [ ] Both Agent and Guardian used `model_type: analysis`.
- [ ] Guardian outputs went through the original parser and risk mapping.
- [ ] At least one real TS-Bench subset produced JSON artifacts.
- [ ] At least one real AgentDojo task used the original runtime and SecReAct path.
- [ ] No vLLM process or production external tool was used on the Mac.
- [ ] Training was reported as unavailable until the official training source is obtained.
