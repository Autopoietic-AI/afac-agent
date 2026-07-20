# AFAC Agent v1 架构设计

## 1. 总体原则

Agent不是预测模型，而是自动科研控制器。

```text
Task Adapter
→ Data Profiler
→ State Builder
→ Memory Retriever
→ Hierarchical Planner
→ Safety Gate
→ Serial Tool Runner
→ Result Parser
→ Feedback / Diagnosis
→ State Update
→ Trajectory
```

## 2. 两层决策

### 实验级Agent

决定：

- 当前属于数据层、模型层、可靠性层还是新信号层；
- 下一轮调用哪个注册工具；
- 预算分配；
- 是否继续、转向、关闭分支或冻结冠军。

### 模型级Router

仅在证据证明可分时，决定某个节点相信哪个专家。

两者不能混合。当前v53Q-1是模型层高置信规则；Agent负责决定它是否值得验证并记录结论。

## 3. 通用核心层

### L0 Safety

- 2小时预算；
- 禁止并行；
- 工具白名单；
- Test标签隔离；
- 提交格式；
- A榜样本级规则禁止迁移B榜；
- 实时Trajectory。

### L1 Data

A1画像：

- 节点、特征、类别；
- CSR稀疏度；
- 有向性、互惠率；
- 训练边同质率；
- Isolated；
- 度桶；
- 类别不平衡；
- Train/Test特征Shift。

A2画像以后接入：

- 用户与商品规模；
- 历史长度；
- Seen/Novel；
- 候选覆盖；
- 用户/商品类别特征；
- Test历史长度Shift。

### L2 Anchor

根据数据画像自动选择家族，而不是固定A榜答案：

```text
Graph-visible:
Directed graph anchor family

Isolated:
Tabular expert family

A2:
DIN + DCN-Mix family
```

### L3 Feedback

不能只返回Accuracy。

A1：

- Overall；
- Fold；
- Class；
- Graph-visible/Isolated；
- Degree Bucket；
- Rescue/Damage/Neutral；
- Oracle；
- Shift；
- Runtime。

A2：

- NDCG；
- 历史长度；
- Seen/Novel；
- Slot；
- Candidate coverage；
- History固定率；
- Shift。

### L4 Memory

每个实验保存：

```json
{
  "state": {},
  "hypothesis": "",
  "action": "",
  "result": {},
  "diagnosis": "",
  "lesson": "",
  "decision": ""
}
```

关闭分支写入状态，Planner不得重启。

### L5 Planner

动作价值按以下因素排序：

```text
错误桶规模
× 成功概率
× 独立信息增量
× 跨数据集可迁移性
÷ 运行时间
÷ 泄漏风险
```

v1先使用确定性Planner。Qwen只在状态、工具和结果Schema稳定后接入，而且只能对注册Tool排序。

## 4. A1问题路由

```text
未画像
→ PROFILE_A1_DATASET

无Anchor
→ RUN/REGISTER_ANCHOR

无统一OOF反馈
→ ANALYZE_ANCHOR_OOF

Graph-visible错误 + 新专家有Oracle
→ AUDIT_EXPERT_COMPLEMENTARITY

Isolated错误 + 旧信号路线关闭
→ AUDIT_NEW_ISOLATED_SIGNAL

预算不足或连续无增益
→ FINALIZE / STOP
```

## 5. 当前A1状态

线上冠军：

```text
v53Q-1
0.7800
```

当前主要矛盾：

- Graph-visible有互补但可靠选择困难；
- Isolated旧信号已耗尽。

Agent下一步不是继续手调阈值，而是：

1. 导入确认历史；
2. 登记线上冠军；
3. 复核数据画像和统一OOF接口；
4. 启动真正的新Isolated信号审计；
5. 同时为B榜准备Fresh Dataset模式。

## 6. 人类保留权限

仅保留：

- API密钥与比赛平台操作；
- 高风险外部资源审批；
- 最终提交确认；
- 对Agent异常的紧急停止。

不再由人手工：

- 选择下一个模型；
- 调阈值；
- 读取CSV后决定分支；
- 补写Trajectory；
- 重复关闭路线。
