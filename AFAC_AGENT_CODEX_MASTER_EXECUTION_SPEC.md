# AFAC2026 Agent：Codex 主执行规格

> 版本：v1.1
> 当前冠军：A1 `v53Q-1 = 0.7800`
> 核心目标：把此前“人工提出实验—运行—读取结果—分析—决定下一步”的流程，复现成可审计、可迁移、受预算约束的自动科研闭环。

---

# 1. 两个 Agent 必须分开

## 1.1 构建期 Codex

Codex负责建设软件工程：

```text
阅读现有代码
→ 整理目录
→ 包装旧模型
→ 编写统一Schema
→ 编写测试
→ 修复路径和环境
→ 运行Smoke Test
→ 提交代码变更
```

Codex不是比赛时决定模型的Agent。

## 1.2 运行期 AFAC Agent

AFAC Agent负责自动科研：

```text
读取数据
→ 生成数据画像
→ 定位主要矛盾
→ 检索历史实验
→ 从注册工具中选择动作
→ Safety Gate
→ 串行运行
→ 解析真实反馈
→ 更新Memory/State/Trajectory
→ CONTINUE / PIVOT / STOP / FINALIZE
```

运行期Agent不得自由改Python源码。

任何新的模型结构，先由构建期Codex实现成注册工具，经过测试后，运行期Agent才能调用。

---

# 2. 当前已经确认的项目起点

## 2.1 A1冠军

```text
Version: v53Q-1
Online Accuracy: 0.7800
```

组成：

```text
Graph-visible → v43C
Isolated → v46A-1 Balanced Seed Consensus
高置信Graph微补丁 → v53Q-1 Transition-Stable Edge-H2
```

本地完整OOF：

```text
8530 / 11001
0.77538405599
```

## 2.2 当前主要矛盾

### Graph-visible

- 图信号强；
- 弱专家存在独立Rescue；
- 缺少可靠的逐节点专家选择；
- Edge Utility可做可靠性证据，但直接删边和全图加权均未通过。

### Isolated

- 约贡献一半错误；
- 当前原始属性与已有专家概率无法形成安全救援头部；
- KNN、类别条件语义空间、OVR、Pairwise旧信号路线均已关闭；
- 下一次必须获取真正的新信号。

## 2.3 不得重复的路线

- ScaleNet；
- MagNet；
- 统一多尺度残差；
- 固定类别对规则；
- 离散可靠性Atlas；
- 通用Confidence Router；
- 硬删边；
- Soft Edge完整模型路线；
- ModernNCA当前路线；
- Global/Conditional Semantic KNN；
- Isolated OVR；
- Isolated Pairwise with current signals；
- 榜单反馈驱动的节点子集。

---

# 3. 总体建设路线

整个系统分成八个里程碑。必须按顺序完成，不要一次性堆满。

---

## M0：仓库与可复现地基

### 目标

让任何新机器都能明确知道：

- 数据在哪里；
- 模型在哪里；
- 当前冠军是什么；
- 环境是什么；
- 每个产物的Hash是什么。

### Codex任务

1. 建立统一目录。
2. 增加 `pyproject.toml` 或稳定的依赖锁。
3. 增加 `.env.example`，严禁提交密钥。
4. 增加 `paths.local.example.yaml`。
5. 建立数据、模型和产物 Manifest。
6. 修复所有绝对路径。
7. 增加路径预检命令。
8. 增加Git初始化、`.gitignore`和备份策略。

### 验收

```text
python -m afac_agent.doctor
```

必须输出：

- Python和CUDA；
- 依赖；
- 数据文件存在性；
- 模型文件存在性；
- 可写目录；
- Champion CSV Hash；
- 所有检查通过或明确失败原因。

---

## M1：确认历史、状态与冠军登记

### 已有

- 19条确认实验历史；
- `ExperimentMemory`；
- `ProjectState`；
- `TrajectoryLogger`；
- `REGISTER_ONLINE_ANCHOR`。

### 需要完善

1. 历史Schema校验。
2. 关闭分支索引。
3. 版本关系图。
4. Champion Registry支持A1和A2。
5. 运行失败不能消耗成功实验轮次。
6. 输入缺失应返回 `WAITING_FOR_INPUT`，不能记为模型失败。
7. 历史导入与冠军登记必须幂等。

### 验收

```text
IMPORT_CONFIRMED_HISTORY → success
REGISTER_ONLINE_ANCHOR → success
```

Memory中恰好导入确认记录，不重复。

---

## M2：数据画像与问题分解引擎

### A1数据画像

自动计算：

- 节点、特征、类别；
- CSR稀疏度；
- 入度、出度、总度；
- 有向性和互惠率；
- Train-Train边同质率；
- Connected Components；
- Isolated；
- Graph-visible；
- Degree 1 / 2–5 / 6+；
- 1-hop、exact-2-hop、multi-hop覆盖；
- 各类别规模、错误率、同质率、度分布；
- Train/Test特征Shift。

### A1问题桶

必须产生机器可读Bucket：

```text
visibility:
  isolated
  graph_visible

degree:
  0
  1
  2_5
  6_plus

hop_regime:
  no_train_neighbor
  one_hop_reliable
  exact2_more_reliable
  multihop_candidate

class_difficulty:
  high_absolute_error
  high_error_rate
  low_homophily
  class_sink_or_source
```

### A2数据画像

后续接入：

- Len0/1/2/3/4+；
- Seen/Novel；
- Repeat/Explore；
- Raw/Dedup/Counts；
- Candidate coverage；
- 用户与商品属性；
- Train/Test历史长度Shift。

### 验收

在A1上复现已知：

- `13752`节点；
- `767`维；
- `10`类；
- `11001/2751`；
- `2213`个训练Isolated。

不得Dense化完整特征矩阵。

---

## M3：旧模型工具化

这是最关键阶段：把“之前能跑的脚本”变成Agent可调用Tool。

### A1 Tool Adapter

至少包括：

1. `A1_V43C_GRAPH_ANCHOR`
2. `A1_V46A1_ISOLATED_EXPERT`
3. `A1_H2GCN_CHALLENGER`
4. `A1_EDGE_UTILITY_AUDIT`
5. `A1_V53Q1_PATCH_REPLAY`
6. `A1_SUBMISSION_FINALIZER`

每个Adapter必须：

- 接受标准路径和JSON配置；
- 不依赖硬编码绝对路径；
- 输出OOF/Test概率；
- 输出标准Result JSON；
- 输出日志；
- 保存模型和Hash；
- 支持 `--smoke_test`；
- 支持超时和中断；
- 失败返回结构化错误。

### A2 Tool Adapter

至少包括：

1. 当前D3N2D冠军Replay；
2. v42c DIN；
3. v48a相关专家；
4. Len0专家；
5. Len3重排器；
6. Submission Finalizer。

### 重要隔离

A榜的Frozen节点规则只能放在：

```text
A1_A_LIST_REPLAY_COMPATIBILITY
```

不能存在于：

```text
A1_GENERIC
B1
```

### 验收

每个Tool：

```text
--smoke_test
```

在小数据或单Fold上真实运行，并输出合法Schema。

---

## M4：统一验证与反馈系统

### A1 Result Schema

```json
{
  "version": "",
  "task": "A1",
  "overall": {},
  "folds": [],
  "visibility_buckets": [],
  "degree_buckets": [],
  "hop_buckets": [],
  "classes": [],
  "transitions": [],
  "rescue_damage": {},
  "oracle": {},
  "distribution_shift": {},
  "runtime": {},
  "safety": {},
  "artifacts": {},
  "decision_evidence": {}
}
```

### A2 Result Schema

按Len、Seen/Novel、Repeat/Explore、Slot输出。

### Gate Failure Auditor

Agent不能只看到 `PASS/STOP`，还要区分：

- 核心机制失败；
- Gate过严但机制正；
- 参考模型失败；
- 候选模型失败；
- 分布漂移；
- 实现错误；
- 协议不匹配；
- 证据样本过少。

### 验收

用v49A回放测试，系统必须识别：

- Edge Utility的可靠性机制为正；
- 整图路线仍STOP；
- 不误判为“所有Edge Utility都无效”。

---

## M5：确定性科研Planner

先不要使用LLM。

### Planner输入

- 数据画像；
- 当前冠军；
- 错误桶；
- 最近实验；
- 关闭分支；
- 保留信号；
- 预算；
- 工具预估时间；
- Oracle与信息增量。

### Planner动作

```text
PROFILE
RUN_ANCHOR
AUDIT_UPPER_BOUND
RUN_CHALLENGER
AUDIT_COMPLEMENTARITY
TRAIN_ROUTER
PIVOT_NEW_SIGNAL
CLOSE_BRANCH
RETURN_TO_ANCHOR
FINALIZE
WAIT_FOR_INPUT
```

### 动作价值

可使用可解释评分：

```text
priority =
error_bucket_mass
× estimated_success_probability
× independent_information_gain
× transferability
÷ runtime_cost
÷ leakage_risk
```

不要求数学绝对精确，但每项必须记录。

### 验收

历史Replay中：

- ModernNCA失败后不能再次选择ModernNCA；
- v52A后不能选择同信号MLP/XGBoost续命；
- Isolated应转向新信号；
- Graph-visible新专家必须先做Oracle审计。

---

## M6：LLM Planner（可选增强）

只有M0–M5全部稳定后再接。

### LLM正确职责

- 总结状态；
- 在已注册动作中排序；
- 提出假设；
- 解释为什么选择；
- 生成结构化Decision JSON；
- 不能输出任意命令和源代码。

### LLM禁止职责

- 自由编辑模型；
- 自由选择Test节点；
- 修改Gate；
- 生成未注册Shell命令；
- 输出最终标签；
- 在Prompt中携带A榜冠军答案。

### 失败降级

LLM不可用、超时、JSON解析失败时：

```text
自动回退Deterministic Planner
```

Agent仍能完整运行。

---

## M7：完整自动闭环

最终目标：

```text
afac-agent run \
  --task A1 \
  --dataset B1.npz \
  --budget-seconds 7200 \
  --mode competition
```

自动执行：

1. Doctor；
2. Profile；
3. Build validation；
4. Run minimum anchor；
5. Error Atlas；
6. Select one registered experiment；
7. Run serially；
8. Parse feedback；
9. Update memory/state；
10. Continue/Pivot/Stop；
11. Train final full-data model；
12. Generate submission；
13. Validate format；
14. Save trajectory and manifest。

### 终止条件

- 预算不足；
- 连续多轮无信息增量；
- 候选Oracle不足；
- 所有目标分支关闭；
- 当前冠军稳定；
- 安全检查失败。

---

## M8：B榜与双任务整合

### B榜原则

从头画像：

```text
A榜机制经验 → 可迁移
A榜样本级答案 → 禁止迁移
```

### 双任务调度

A1和A2分别拥有独立：

- 状态；
- Memory；
- Budget；
- Trajectory；
- Champion。

最终有一个Meta Controller决定研发优先级，但比赛运行预算按任务隔离。

---

# 4. 当前真实故障与修复

本次运行事实：

- `IMPORT_CONFIRMED_HISTORY`成功；
- 导入19条；
- `REGISTER_ONLINE_ANCHOR`失败；
- 原因是Anchor路径仍指向旧目录，而Agent实际位于新目录。

这不是模型、Planner或Memory故障。

修复方案：

```text
ANCHOR="$PWD/artifacts/A1_v53q1_transition_stable_edge_h2_SAFE.csv"
```

不要继续硬编码：

```text
C:/Users/李天皓/agent比赛/model_pro/a1/new-a1/...
```

所有内部资产优先使用：

```text
project_root相对路径
```

所有外部大数据使用：

```text
paths.local.yaml
```

---

# 5. API与密钥策略

## 5.1 Codex开发工具

使用Codex CLI、IDE或桌面端建设代码时：

- 优先通过ChatGPT账户登录；
- 不要求手工复制API Key；
- 密钥与认证不写入仓库。

## 5.2 比赛运行期LLM

v1–v5确定性Agent不依赖LLM，因此现在不需要先配置比赛LLM Key。

到M6时：

1. 从官方赛题规则确认允许的服务商、模型名和接口；
2. 再申请对应Key；
3. 仅通过环境变量传递；
4. 提供无Key降级路径；
5. 不在YAML、日志、Trajectory中打印密钥。

环境变量接口预留：

```text
AFAC_LLM_ENABLED=0
AFAC_LLM_PROVIDER=
AFAC_LLM_MODEL=
AFAC_LLM_API_KEY=
AFAC_LLM_BASE_URL=
```

`.env.example`只能放空值。

---

# 6. 本地Codex与云端Codex如何选择

## 推荐：本地Codex

原因：

- 数据和历史模型在Windows本机；
- 需要调用Conda、GPU和大量NPZ/Checkpoint；
- 需要连续运行测试；
- 不适合上传完整比赛数据与模型。

推荐环境：

- Git仓库；
- Codex CLI或IDE；
- 初期使用审批模式；
- 每个里程碑单独commit。

## 云端Codex

适合：

- 纯代码重构；
- Schema、单元测试、文档；
- GitHub PR；
- 不需要本地大数据和GPU的任务。

云端不能直接看到你的本地绝对路径和大模型资产。要使用云端，必须：

- 将代码推到私有GitHub仓库；
- 提供Synthetic Fixture；
- 把真实数据与Checkpoint排除在Git之外；
- 本地再运行集成测试。

## 最佳组合

```text
云端/网页Codex：
重构、测试、文档、PR

本地Codex：
路径接入、真实数据、GPU、集成Smoke Test
```

---

# 7. Codex每阶段工作方式

每次只给Codex一个里程碑，避免它一次改完整系统。

标准指令：

```text
先阅读AGENTS.md、PROJECT_STATE.md和本阶段规格。
只实现M<n>，不要提前实现后续阶段。
先列出审计结果和修改计划。
再编写测试。
再实现。
运行全部相关测试。
最后报告：
- 修改文件
- 测试命令与结果
- 尚未解决问题
- 是否改变验证定义或冠军
```

---

# 8. 第一轮Codex任务

当前不要立即做v2全模型重训。

第一轮固定任务：

```text
M0 + M1 Stabilization
```

内容：

1. 修复所有绝对路径；
2. 增加Doctor；
3. 增加路径配置文件；
4. 增加Schema校验；
5. 修复失败轮次预算记录；
6. 增加Anchor缺失时的明确等待；
7. 对历史导入和Anchor登记写集成测试；
8. 保证Windows Git Bash和CMD均可运行；
9. 更新README和PROJECT_STATE；
10. 不接LLM、不训练模型。

验收后再进入：

```text
M2 Data Profiler
```

---

# 9. 最终成功标准

系统完成不是“Agent能跑起来”，而是同时满足：

- 新数据可自动画像；
- 已关闭路线不会重复；
- 旧强模型可标准调用；
- 每次只运行一个实验；
- 指标和错误桶自动解析；
- Planner能解释动作；
- 无LLM仍可运行；
- LLM不可自由执行；
- 预算、状态和轨迹一致；
- A/B榜样本信息隔离；
- 一条命令可从数据跑到安全提交；
- 新对话或新机器只看文档即可继续。
