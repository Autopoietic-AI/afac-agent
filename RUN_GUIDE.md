# AFAC Agent v1.1 运行指南

## 0. 项目根目录

所有命令都从实际代码根目录运行，也就是包含 `afac_agent/`、`config/`、`tools/` 的目录。

如果解压包外面还有一层目录，请进入内层：

```bash
cd "<AFAC_AGENT_V1_1_CODEX_READY>/afac_agent_v1"
```

不要把项目内部资产写成旧机器绝对路径。当前 A1 champion 默认从这里读取：

```text
artifacts/A1_v53q1_transition_stable_edge_h2_SAFE.csv
```

外部数据、OOF、Checkpoint 放在本机路径配置中：

```text
config/paths.local.yaml
```

可从示例复制：

```bash
cp config/paths.local.example.yaml config/paths.local.yaml
```

Windows CMD 可手工复制该文件。`paths.local.yaml` 不提交、不写密钥。

## 1. Doctor 预检

```bash
python -m afac_agent.doctor --project_root .
```

JSON 输出：

```bash
python -m afac_agent.doctor --project_root . --json
```

Doctor 检查：

- Python 与依赖可用性；
- ProjectState；
- Tool Registry；
- confirmed history memory records；
- Trajectory 状态；
- packaged champion CSV；
- 本地路径配置；
- 基本可写目录。

## 2. Dry Run

不加 `--execute` 时只做决策和安全检查：

```bash
python -m afac_agent.main --project_root .
```

初始状态应选择：

```text
IMPORT_CONFIRMED_HISTORY
```

结果状态应为：

```text
dry_run
```

## 3. 执行历史导入

```bash
python -m afac_agent.main --project_root . --execute
```

生成：

```text
memory/experiment_memory_a1.jsonl
output/trajectory_A1.json
```

成功后状态前进到：

```text
history_imported = true
next_required_capability = register_anchor
```

历史导入按 `version` 幂等去重；重复执行不会重复写入 19 条 confirmed history。

## 4. 登记线上冠军

再次运行：

```bash
python -m afac_agent.main --project_root . --execute
```

默认使用：

```text
artifacts/A1_v53q1_transition_stable_edge_h2_SAFE.csv
```

生成或复核：

```text
artifacts/online_anchor/online_anchor_manifest.json
```

Champion 登记按版本、线上分、CSV hash、行数和类别数幂等；重复执行不会修改 champion CSV。

## 5. 到达缺失输入时停止

当前 M0/M1 不训练模型，也不实现 M2 之后功能。走到需要 NPZ、OOF 或尚未绑定工具时，Agent 会返回：

```text
waiting_for_input
```

这不是实验失败，也不会消耗成功实验轮次。

## 6. Bootstrap

Windows:

```text
bootstrap_agent.bat
```

Git Bash:

```bash
bash bootstrap_agent.sh
```

如果 Python 不在 `PATH`，先设置：

```bash
export AFAC_PYTHON="/path/to/python"
```

Windows CMD:

```bat
set AFAC_PYTHON=C:\path\to\python.exe
```

Bootstrap 会先运行 Doctor，再串行执行已注册、已绑定、输入满足的工具，直到等待输入或工具未绑定。

## 7. 测试

```bash
python -m pytest -vv
```

M0/M1 测试覆盖：

- champion CSV hash 和格式不变；
- 历史导入幂等；
- champion 登记幂等；
- 缺失文件返回 `waiting_for_input`；
- 空 command template 返回 `waiting_for_input`；
- 工具失败不消耗成功轮次；
- ProjectState / Tool Registry / Memory / Anchor Manifest 校验；
- Doctor smoke；
- Windows 中文和空格路径。

## 8. 禁止事项

M0/M1 阶段禁止：

- 训练模型；
- 接入 LLM；
- 接入 A2；
- 修改 Test 预测；
- 修改 champion CSV；
- 修改当前线上分；
- 修改 Fold、Gate 或 OOF 定义；
- 删除历史文件；
- 自动推进 M2 之后功能。

## 9. M2 A1 Data Profiler

M2 only implements the A1 Data Profiler. It is read-only, deterministic,
idempotent, fold-aware when a canonical fold file is provided, and leakage-safe.
It does not train models, call GPU, generate predictions, create submissions,
write Memory, mutate Project State, or consume successful experiment rounds.

Dataset-only profile:

```bash
python -m afac_agent.profilers.a1_data_profiler \
  --npz_path "<path-to-A1.npz>" \
  --edges_csv "<optional-path-to-A1_edges.csv>" \
  --champion_csv artifacts/A1_v53q1_transition_stable_edge_h2_SAFE.csv \
  --out_dir artifacts/data_profile/a1_m2_v1
```

Legacy wrapper:

```bash
python tools/profile_a1_dataset.py \
  --npz_path "<path-to-A1.npz>" \
  --out_dir artifacts/data_profile/a1_m2_v1
```

Optional tiers:

- `fold_aware_structure` requires `--fold_file`.
- `full_anchor_oof` requires both `--fold_file` and the exact v53Q-1
  `--anchor_oof_npz`.

Missing optional Fold or OOF inputs degrade to the lower available tier and are
recorded in warnings. Use `--require_fold` or `--require_oof` when the desired
behavior is `waiting_for_input` instead.

M2 outputs:

```text
artifacts/data_profile/a1_m2_v1/
  a1_data_profile.json
  a1_profile_manifest.json
  a1_input_validation.json
  a1_node_buckets.csv
  a1_structure_profile.csv
  a1_train_neighbor_profile.csv
  a1_hop_profile.csv
  a1_class_profile.csv
  a1_confusion_transitions.csv
  a1_prediction_sink_source.csv
  a1_fold_profile.csv
  a1_shift_profile.csv
  a1_signal_inventory.json
  a1_problem_map.json
  A1_DATA_PROFILE_REPORT.md
```

## 10. M3A Tool Adapter Foundation

M3A introduces a minimal Adapter protocol and `AdapterRunner`. Registered tools
with `adapter_entrypoint` use the Adapter path; older tools keep the existing
`command_template` path.

The first real Adapter is read-only:

```text
A1_V53Q1_PATCH_AUDIT
```

It verifies the packaged v53Q-1 Champion CSV, the v46A-1 base CSV, v49A OOF/Test
meta CSVs, the audit markdown and optional patch source hash. It does not
execute patch replay and must not generate prediction CSVs.

Dry run:

```bash
python -m afac_agent.main run-adapter --tool A1_V53Q1_PATCH_AUDIT
```

Execute with explicit local inputs:

```bash
python -m afac_agent.main run-adapter \
  --tool A1_V53Q1_PATCH_AUDIT \
  --anchor_csv artifacts/A1_v53q1_transition_stable_edge_h2_SAFE.csv \
  --v53q1_base_csv "<local-v46A1-base-csv>" \
  --v49a_oof_meta_csv "<local-v49A-oof-meta-csv>" \
  --v49a_test_meta_csv "<local-v49A-test-meta-csv>" \
  --v53q1_audit_md artifacts/V53Q1_TRANSITION_STABLE_EDGE_H2_AUDIT.md \
  --v53q1_patch_py artifacts/a1_v53q1_transition_stable_edge_h2_patch.py \
  --execute
```

Outputs:

```text
artifacts/adapter_runs/A1_V53Q1_PATCH_AUDIT/<identity_hash>/
  execution_result.json
  input_manifest.json
  stdout.log
  stderr.log
  audit_details.json
```

`artifacts/adapter_runs/` and `config/paths.local.yaml` are ignored by git.
External historical paths may be supplied by CLI or local paths config only.
