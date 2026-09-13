# Official TS-Bench snapshot

This directory contains the official JSON objects referenced by the upstream
`ToolSafe/TS-Bench` Git LFS pointers.  It is kept in the independent practice
overlay so the original `ToolSafe` checkout remains untouched.

TS-Bench is not the raw ASB, AgentHarm, or AgentDojo task dataset.  It is a
step-level trajectory benchmark derived from those environments.  Each record
contains an instruction, interaction history, current tool action, tool
environment information, and a safety score.  The ASB records additionally
contain attack metadata such as attack type and attack success.

The nine files are grouped as follows:

- `agentdojo-traj/`: workspace, travel, slack, and banking trajectories;
- `agentharm-traj/`: harmful and benign step trajectories;
- `asb-traj/test/`: direct prompt injection success, indirect prompt injection
  success, and attack-failure trajectories.

The files were retrieved from the public GitHub media endpoint and verified
against the SHA-256 digests and byte sizes in the upstream LFS pointers.  To
repeat the recovery, run:

```bash
ToolSafe/.venv-phase4/bin/python \
  practice/toolsafe_reproduction/scripts/fetch_tsbench.py
```

The script is idempotent: an existing file is re-verified and skipped.  Use
`--force` only when a fresh download is required.

The recovered snapshot contains 7,182 step records (1,220 AgentDojo, 731
AgentHarm, and 5,231 ASB records).  The exact per-file manifest is in
`manifest.json`.
