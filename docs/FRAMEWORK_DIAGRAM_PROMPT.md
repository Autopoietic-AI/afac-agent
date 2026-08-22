# 框架图绘画提示词（Framework Diagram Prompts）

> 三种用法：
> 1. **Mermaid 源码**（推荐）——粘到 GitHub / mermaid.live 直接渲染，精确无歧义；
> 2. **英文图像生成提示词**——喂给 GPT-4o image / Midjourney / DALL·E；
> 3. **中文图像生成提示词**——喂给即梦 / 通义万相 / 文心一格。

---

## 1. Mermaid 源码（可直接渲染）

```mermaid
flowchart TB
    subgraph L8["L8 收口层 Completion & Deployment"]
        C1["完成合同<br/>科学资格而非文件存在"]
        C2["部署权限四段审计<br/>格式/科学/预算/数据"]
        C3["Anchor 兜底<br/>validated baseline fallback"]
    end
    subgraph L7["L7 监督层 Supervision"]
        S1["心跳 / STATUS / Dashboard"]
        S2["执行身份 vs 缓存身份"]
        S3["LLM 账本 · 法医标记(不删)"]
    end
    subgraph L6["L6 资源层 Resource"]
        R1["monotonic 硬截止 + 部署预留"]
        R2["实测估时(非注册表猜测)"]
        R3["折间熔断器 circuit breaker"]
    end
    subgraph L5["L5 评价层 Evaluation"]
        E1["统一指标 + 错误分解互斥"]
        E2["目标指标合同<br/>同Fold/同Panel/同Metric/同K/同Evaluator"]
        E3["No-op 检测 → 退款+隔离"]
    end
    subgraph L4["L4 执行层 Execution"]
        X1["F0诊断→F1筛选→F2确认→F3全CV→F4部署"]
        X2["规范Fold: 哈希稳定, 前缀子集, 不重划"]
        X3["晋升门: 配对证据才能晋升"]
    end
    subgraph L3["L3 权限层 Permission"]
        P1["实验类型系统:<br/>诊断/回放/筛选/确认/全CV/部署"]
        P2["诊断永远不可部署"]
        P3["Portfolio 分层: 诊断/科研/部署"]
    end
    subgraph L2["L2 编译层 Compilation ★"]
        G1["提案→算子编译器"]
        G2["能力注册表接地:<br/>不可用→blocked, 不降级"]
        G3["语义基因组哈希:<br/>revise必须改语义, 改名=重复"]
    end
    subgraph L1["L1 认知层 Cognition (LLM 只在这里)"]
        K1["问题层级: 全局→分桶→机理"]
        K2["M6B 提案者 (假设/来源/预算/停止条件)"]
        K3["M6C 批判者 (approve/revise/reject)"]
        K4["科研记忆: 事件溯源, 追加只写"]
    end
    subgraph L0["L0 任务抽象层 Task Abstraction"]
        D1["任务适配器: 任意任务→规范形"]
        D2["数据合同: 每个数有出处+哈希"]
        D3["全量/抽样分离 · 四阶段对账"]
    end

    K2 --> G1
    G1 -->|编译通过| P1
    G1 -.->|算子不可用| G2x["blocked_missing_adapter 🔒"]
    P1 --> X1
    X1 --> E1
    E1 --> E3
    E3 -->|有效实验| P3
    E3 -.->|No-op| E3x["退款+隔离 🔒"]
    P3 --> K3
    K3 -->|revise| G3
    G3 --> K2
    K3 -->|stop / promote| C1
    C1 --> C2 --> C3
    L6 -.约束.- L4
    L7 -.记录.- L1

    classDef gate fill:#ffe4e1,stroke:#c0392b,stroke-width:2px
    classDef llm fill:#e8f4fd,stroke:#2980b9
    class G2x,E3x,P2 gate
    class K1,K2,K3,K4 llm
```

---

## 2. 英文图像生成提示词（GPT-4o image / Midjourney / DALL·E）

```text
A clean, publication-quality software architecture diagram in flat vector
style, white background, title at top: "Self-Iterating Agent Framework (SIAF)".

The diagram shows NINE horizontal stacked layers, from bottom to top, each a
rounded rectangle band with a soft pastel fill and a bold label on the left
edge. From bottom to top:

  L0 Task Abstraction (light gray): "task adapters · data contracts with
  provenance · full-vs-sampled separation · cross-stage reconciliation"
  L1 Cognition (light blue): "problem hierarchy · LLM proposer · LLM critic ·
  research memory" — this is the ONLY layer containing LLM icons (two small
  brain/spark icons)
  L2 Compilation (light orange, marked with a star): "proposal-to-operator
  compiler · capability registry grounding · semantic genome hash"
  L3 Permission (light red): "typed experiment permissions · diagnostics can
  NEVER deploy (padlock icon) · tiered portfolio"
  L4 Execution (light green): "fidelity ladder F0 diagnostic → F1 screen →
  F2 confirm → F3 full-CV → F4 deployment · canonical folds · promotion gate"
  L5 Evaluation (light teal): "unified metrics · mutually-exclusive error
  decomposition · paired target-metric contract · no-op detector"
  L6 Resource (light yellow): "monotonic hard deadline · deployment reserve ·
  measured runtime estimates · between-fold circuit breaker"
  L7 Supervision (light purple): "live heartbeat · dashboard · append-only
  LLM ledger · forensic marking"
  L8 Completion & Deployment (dark navy, top band): "completion contract ·
  four-part deployment audit · validated-anchor fallback · terminal states"

A bold vertical arrow loop runs along the LEFT side showing the iteration
cycle with labeled steps: "problem → propose → compile → gate → genome →
execute → evaluate → portfolio → critique → next decision", curving back from
"critique" to "propose" to show self-iteration. Small red padlock icons mark
the deterministic gates (compiler, safety gate, evaluation contract,
portfolio, deployment). A dashed red arrow from the LLM critic labeled
"revise (must change semantic hash)" returns to the compiler.

On the RIGHT side, a vertical sidebar titled "Contract Stack" lists six small
chips: "Data Contract · Permission Model · Target Metric Contract · Budget
Contract · Deployment Contract · Completion Contract".

At the bottom, a one-line caption in italic: "Orchestration is not science:
every iteration must earn completion through contracts."

Typography: sans-serif, crisp, legible; generous whitespace; subtle drop
shadows; consistent 8-color pastel palette; no photographs, no 3D effects,
no gradients on text, no watermark, no decorative clutter.
Aspect ratio 4:5 (portrait), high resolution, suitable for a research paper
figure.
```

---

## 3. 中文图像生成提示词（即梦 / 通义万相 / 文心一格）

```text
一张论文级软件架构图，扁平矢量风格，纯白背景，顶部大标题：
"自我迭代智能体通用框架（SIAF）"。

画面主体为自下而上堆叠的九个圆角矩形层级带，每层左侧有粗体编号与名称，
填充柔和马卡龙色：

  最底层 L0 任务抽象层（浅灰）：任务适配器、带出处与哈希的数据合同、
        全量与抽样分离、四阶段对账
  L1 认知层（浅蓝）：问题层级（全局→分桶→机理）、LLM 提案者、LLM 批判者、
        科研记忆；全图只有这一层出现两个小脑袋/火花图标表示 LLM
  L2 编译层（浅橙，带一颗星标）：提案到算子编译器、能力注册表接地、
        语义基因组哈希
  L3 权限层（浅红）：实验类型系统、"诊断永远不可部署"（配挂锁图标）、
        分层组合（诊断/科研/部署）
  L4 执行层（浅绿）：保真度阶梯 F0诊断→F1筛选→F2确认→F3全CV→F4部署、
        规范Fold、晋升门
  L5 评价层（浅青）：统一指标、互斥错误分解、配对目标指标合同、
        无操作检测
  L6 资源层（浅黄）：单调硬截止、部署预留、实测估时、折间熔断器
  L7 监督层（浅紫）：实时心跳、仪表盘、追加只写 LLM 账本、法医标记
  最顶层 L8 收口层（深藏青）：完成合同、部署四段审计、Anchor 兜底、
        终态枚举

画面左侧有一条粗壮的纵向循环箭头，标注自我迭代步骤：
"问题→提案→编译→门禁→基因组→执行→评价→组合→批判→下一步"，
箭头从"批判"弯回"提案"，形成闭环。编译器、安全门、评价合同、组合、
部署五处各有一个红色小挂锁图标表示确定性门。另有一条红色虚线箭头从
批判者指回编译器，标注"revise（必须改变语义哈希）"。

画面右侧有一条窄侧栏，标题"合同栈"，内含六枚小胶囊标签：
数据合同、权限模型、目标指标合同、预算合同、部署合同、完成合同。

底部一行斜体小字标语：
"编排不等于科研：每一次迭代都必须凭合同赢得完成。"

字体为无衬线黑体，清晰易读，留白充足，轻微投影，整体不超过 8 种
柔和配色；不要照片元素、不要 3D 效果、不要文字渐变、不要水印、
不要装饰性杂物。竖版 4:5，高分辨率，适合作为论文插图。
```

---

## 4. 渲染后检查清单（生成图必须满足的要点）

- [ ] 九层顺序正确（L0 在底，L8 在顶）
- [ ] LLM 图标只出现在 L1 认知层（核心主张：LLM 不做裁决）
- [ ] 左侧循环箭头闭环（批判 → 提案）
- [ ] 五处挂锁：编译器 / 安全门 / 评价合同 / 组合 / 部署
- [ ] "诊断不可部署"有明确视觉表达
- [ ] 右侧合同栈六胶囊齐全
- [ ] 底部标语 "Orchestration is not science"
- [ ] 无乱码文字（图像模型常在长文字上出错——字多就改用 Mermaid 或 PPT 手绘）
