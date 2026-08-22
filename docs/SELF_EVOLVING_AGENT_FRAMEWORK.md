# 通用自我迭代智能体框架（Self-Iterating Agent Framework, SIAF）

> 一份面向"能真正自我改进的 Agent"的架构设计文档。
> 本文不描述某个具体竞赛方案，而描述一套**与任务无关**的通用框架；
> AFAC2026 四任务只是它的一次完整实例化与压力测试。

---

## 摘要（Abstract）

大模型驱动的"自动科研/自动工程" Agent 面临一个根本性陷阱：**执行不等于进展（Execution ≠ Progress）**。一个 Agent 可以调用 LLM、写出成堆的 Artifact、生成提交文件、汇报 `status=completed`，却在科学上毫无产出——所有轮次都是只读诊断、预算超限、指标口径混乱、无效结果被当作冠军部署。

本文提出一套通用自我迭代智能体框架（SIAF）。核心思想是：**把"迭代"关进一套机器可校验的合同（Contract）体系里，让"空转"在类型系统层面无法被表示**。框架由九个正交的平面（Plane）组成：任务抽象、认知、编译、权限、执行、评价、资源、监督、收口。LLM 只负责"提出假设"与"批判结果"；所有准入、晋升、部署、完成的决策都由确定性门（Deterministic Gate）执行，LLM 无法绕过。

本框架在一次真实竞赛（AFAC2026 Task 3，覆盖节点分类与序列推荐共 4 个子任务）中完整落地，并通过一次真实的"假完成"故障（invalid scientific deployment）驱动的修复验证了其必要性：修复后，同类故障在框架内**无法被构造出来**。

---

## 1. 动机（Motivation）：一次"完美"的空转

框架的需求不是设计出来的，是被一次真实故障逼出来的。一次正式运行产出了表面上完整的轨迹：Manifest、每轮 Artifact、提交 CSV、`status=completed`。审计后发现：

- 3 轮全部是**确定性诊断**（只读统计，没有训练任何模型）；
- `scientific_rounds_used = 0`，却**部署了诊断结果**作为提交；
- 实际运行 8340 秒，超出 7200 秒硬预算 19 分钟，状态仍是 `completed`；
- 商品数量在不同阶段分别是 14,065 与 40,011（**把用户 ID 当成了商品**）；
- 一轮 No-op（预测与父代完全相同）仍被登记进 Portfolio；
- Critic 三轮都判 `revise`，但下一轮只是把诊断**改了名字**，底层计算完全不变。

**结论：一个能编排工具调用的 Agent，不等于一个能做科研的 Agent。**
编排（Orchestration）解决"动作能否发生"；科学（Science）要求"动作是否产生可验证的新知识"。SIAF 的全部合同，都是为了堵住这两者之间的缝隙。

---

## 2. 十条设计原则（Design Principles）

| # | 原则 | 含义 |
|---|------|------|
| P1 | **执行不等于进展** | `completed` 状态必须证明科学资格（真实实验轮数、可部署候选），而不是证明 Artifact 存在 |
| P2 | **实验权限类型化** | 每个执行有强类型（诊断/筛选/确认/部署），诊断在类型层面**永远不可部署** |
| P3 | **数量必须有出处** | 任何规模值（商品数/用户数/测试数）携带 source_file、source_column、计数规则、哈希；无出处的数字不合法 |
| P4 | **配对评价合同** | 父子比较必须同 Fold、同 Panel、同指标、同 K、同 Evaluator 版本，缺一即阻断 |
| P5 | **语义修订强制** | Critic 判 `revise` 后，下一轮的语义基因组哈希必须真实变化；改名字不算修订 |
| P6 | **预算用实测说话** | 估时在第一次真实实验后用实测值更新；硬截止用 `monotonic` 时钟；执行中有熔断器 |
| P7 | **No-op 退款与隔离** | 预测与父代完全相同的实验退还科学轮次，且永不进入有效 Portfolio |
| P8 | **诚实的能力注册表** | 缺失依赖（如 lightgbm/torch）如实登记 `available=false`；回退方案不得冒充科学最优 |
| P9 | **Anchor 兜底部署** | 没有候选赢得部署权时，回退到在当前 Fold 上重新验证过的基线，并如实声明 fallback |
| P10 | **冻结资产与法医保全** | 冠军/Fold/历史哈希在每次运行前后校验；无效运行**标记而非删除**，作为回归测试夹具 |

---

## 3. 架构总览：九个平面（The Nine Planes）

```
┌─────────────────────────────────────────────────────────────────────┐
│ L8 收口层 Completion & Deployment                                    │
│     完成合同 · 部署权限四段审计 · Anchor 兜底 · 终态枚举              │
├─────────────────────────────────────────────────────────────────────┤
│ L7 监督层 Supervision                                                │
│     心跳/STATUS/Dashboard · 执行身份 vs 缓存身份 · LLM 账本 · 法医标记│
├─────────────────────────────────────────────────────────────────────┤
│ L6 资源层 Resource                                                   │
│     monotonic 硬截止 · 部署预留 · 实测估时 · 折间熔断 · Cheap 成本合同 │
├─────────────────────────────────────────────────────────────────────┤
│ L5 评价层 Evaluation                                                 │
│     统一指标 · 错误分解互斥 · 目标指标合同（同Fold/Panel/Metric/K）   │
├─────────────────────────────────────────────────────────────────────┤
│ L4 执行层 Execution                                                  │
│     F0诊断→F1筛选→F2确认→F3全CV→F4部署 · 规范Fold · 晋升门 · 内存安全 │
├─────────────────────────────────────────────────────────────────────┤
│ L3 权限层 Permission                                                 │
│     实验类型系统（诊断不可部署）· No-op 退款与隔离 · Portfolio 分层   │
├─────────────────────────────────────────────────────────────────────┤
│ L2 编译层 Compilation                    ★ 框架的关键创新            │
│     提案→算子编译器 · 能力注册表接地 · 语义基因组哈希 · 阻断而非降级  │
├─────────────────────────────────────────────────────────────────────┤
│ L1 认知层 Cognition                                                │
│     问题层级（全局→分桶→机理）· LLM 提案者 · LLM 批判者 · 科研记忆   │
├─────────────────────────────────────────────────────────────────────┤
│ L0 任务抽象层 Task Abstraction                                       │
│     任务适配器 · 数据合同（出处+哈希）· 全量/抽样分离 · 跨阶段对账    │
└─────────────────────────────────────────────────────────────────────┘
```

每一层只与相邻层通过**显式合同对象**通信；LLM 只出现在 L1（提案与批判），
它提交的提案必须经过 L2 编译、L3 授权、L4 执行、L5 评价，才能影响 L8 的收口。

---

## 4. 自我迭代主循环（The Self-Iteration Loop）

一轮（Round）的完整生命周期：

```
问题选择 ──▶ LLM 提案 ──▶ 编译为算子 ──▶ 安全门 ──▶ 基因组哈希 ──▶ 真实执行
 (L1)        (L1)          (L2)          (L3/M5)      (L2)          (L4, 按阶梯选Fold)
                                                                      │
 下次决策 ◀── 批判+尸检 ◀── Portfolio 更新 ◀── 无操作审计 ◀── 统一评价 ◀─┘
 (继续/修订/  (L1 M6C)      (L3 分层: 诊断/   (L5 同预测→退款,   (L5 指标+合同)
  换族/换题/                  科研/部署)        不进组合)
  停止)
```

关键语义：

- **诊断不消耗科学轮次**；只有发生真实训练 + 新 Fold 预测 + 模型比较，才计一轮科学实验。
- **晋升（Screen→Confirm）需要配对证据**：同 Fold 父代指标缺失时，状态是
  `blocked_missing_parent_target_metric`，而不是伪造"无提升"。
- **修订必须语义化**：`revise` 后 `semantic_genome_hash` 不变 = `duplicate_revision`，
  禁止执行；连续两次重复 → 强制切换算子族或问题。
- **停滞逐级升级**：无改善第 1 轮 → 修订/换族；第 2 轮 → 换问题/全局探索；
  只有全局探索后仍无可行路径才停止。

---

## 5. 合同栈（The Contract Stack）

### 5.1 数据合同（Data Contract, L0）

每个规模字段是 `ScaleValue{value, source_file, source_column, counting_rule,
deduplicated, sampled, membership_hash}`。Profiler 允许抽样加速，但必须显式记录
`profile_scope`（full / sampled_stratified / sampled_head）与
`n_total` / `n_profiled` 的分离——抽样结果不得冒充全量。四个阶段
（输入发现 / 数据画像 / 实验执行 / 部署合法性）的实体空间必须对账一致，
否则 `blocked_data_contract_mismatch`，不得进入问题选择。

### 5.2 实验权限模型（Permission Model, L3）

| 类型 | Fold | 消耗轮次 | 进 Portfolio | 可成 Incumbent | 可部署 |
|------|------|---------|--------------|----------------|--------|
| DETERMINISTIC_DIAGNOSTIC | 0 | 否 | 否（只进诊断登记处） | 否 | **否** |
| CACHED_REPLAY | — | 否 | 否 | 否 | **否** |
| SCREEN_EXPERIMENT | 1–2 | 是 | 是（screen_only） | 否 | **否** |
| CONFIRM_EXPERIMENT | 3 | 是 | 是 | 是 | 是 |
| FULL_CV_EXPERIMENT | 5 | 是 | 是 | 是 | 是 |
| DEPLOYMENT_MODEL | full | — | 部署组合 | — | 是 |

### 5.3 目标指标合同（Target Metric Contract, L5）

每个问题节点声明 `target_panel_id / target_metric_name / target_k /
target_direction`。父代与候选在**同 Fold、同 Panel、同指标、同 K、同
Evaluator 版本**下配对比较。目标面板的父代指标缺失时，输出
`blocked_missing_parent_target_metric`——绝不把总体指标写进分桶指标里
冒充结论。

### 5.4 资源合同（Budget Contract, L6）

`hard_deadline = start_monotonic + max_wall_clock_seconds`；
`research_deadline = hard_deadline − deployment_reserve`。
仅当 `remaining ≥ estimate + reserve + safety_margin` 才允许启动新实验。
估时在首个真实实验后切换为**实测值**。执行中在每折之间检查截止，
超时熔断并保留已完成折的部分结果。"Cheap"由秒数定义
（F0 ≤120s、缓存回放 ≤30s 且必须真命中缓存），而不是由名字定义。

### 5.5 完成与部署合同（Completion & Deployment, L8）

`status=completed` 必须同时满足：数据合同通过、指标合同通过、至少一次真实
算子执行、有效科学轮数 ≥1（或明确声明 anchor fallback）、Portfolio 中无
No-op、Incumbent 具有部署权限、部署四段审计（格式/科学/预算/数据）全部通过、
总墙钟在硬预算内、Artifact 哈希完整。终态枚举化：
`completed / completed_with_anchor_fallback / completed_smoke /
incomplete_no_scientific_experiment / incomplete_no_deployable_candidate /
blocked_data_contract_mismatch / failed_budget_contract /
failed_deployment_permission / failed_operator_compilation`。

---

## 6. 实例化案例（Case Study: AFAC2026 Task 3）

框架在一次真实竞赛中完整实例化，覆盖两种任务族：

| 子任务 | 类型 | 线上最佳 | 框架证据 |
|--------|------|---------|---------|
| A1 | 节点分类 | 0.7800（冻结冠军） | 闭环执行，融合候选入组合（宏平均保护未晋升） |
| A2 | 推荐 | 0.5093（冻结冠军） | 评估基建完成（NDCG@10 锚点 0.5925） |
| B1 | 节点分类 | 0.37974 | 闭环完成；后接入 v2 编排器并加装预算熔断 |
| B2 | 序列推荐 | 0.06838 → 修复中 | 故障运行法医标记；900 秒真实 LLM 冒烟通过 |

B2 冒烟（900 秒预算）的实测轨迹：12 次真实 LLM 调用（问题综合 / 提案 /
批判 / 尸检，每轮各一次）、2 次真实科学筛选实验、数据合同全阶段一致
（n_items=14,065 全部出自 item.csv）、No-op 隔离生效、冒烟模式正确拒绝部署、
完成合同通过、墙钟 599.5 秒。工程面：407 项测试通过，冻结资产哈希不变。

---

## 7. 局限与开放问题（Limitations & Open Problems）

- **检索/召回类算子的真实规模成本**仍是主要瓶颈（实测 ~3885 秒/折）；
  两小时级正式运行需要先完成检索索引缓存与稀疏化优化。
- **LLM 提案质量**是科学产出的上限：合同能保证"不做假"，不能保证"想到好点子"。
  接入有来源的实时方法调研（source-grounded method research）是下一步。
- **跨任务迁移**当前只允许 advisory-only 先验；如何在合同体系内安全地
  量化迁移收益仍开放。
- 框架假设**单机串行**；并行实验需要额外的隔离与合并合同。

---

## 8. 一句话总结

> **让 Agent 自我迭代不难；难的是让它无法用"看起来很忙"来骗你。**
> SIAF 的做法：LLM 只许提出假设，真理由确定性合同裁决——
> 数据要有出处、实验要有类型、比较要有配对、修订要有语义、
> 预算要用实测、完成要够资格、部署要有权限、错了要留证据。
