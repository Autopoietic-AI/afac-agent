# -*- coding: utf-8 -*-
"""基于问题层级的确定性科研Planner v1。"""

from __future__ import annotations

from .registry import ToolRegistry
from .schemas import AgentDecision, ProjectState


ACTION_BY_CAPABILITY = {
    "import_history": "IMPORT_CONFIRMED_HISTORY",
    "profile_dataset": "PROFILE_A1_DATASET",
    "register_anchor": "REGISTER_ONLINE_ANCHOR",
    "analyze_anchor_oof": "ANALYZE_ANCHOR_OOF",
    "new_isolated_signal": "AUDIT_NEW_ISOLATED_SIGNAL",
    "expert_complementarity": "AUDIT_EXPERT_COMPLEMENTARITY",
    "finalize": "FINALIZE_CURRENT_CHAMPION",
}


class HierarchicalPlanner:
    """
    第一版优先保证可审计和不重复失败。

    Planner输入不是一个Accuracy，而是：
    - 数据类型与数据画像；
    - 当前问题层级；
    - 错误桶；
    - 已关闭分支；
    - 已保留信号；
    - 预算；
    - 当前缺失能力。
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def select(self, state: ProjectState) -> AgentDecision:
        action = ACTION_BY_CAPABILITY.get(
            state.next_required_capability,
            "REVIEW_AND_REPLAN",
        )
        tool = self.registry.get(action)
        eligible_names = {
            item.name
            for item in self.registry.eligible(state)
        }

        if action not in eligible_names:
            return AgentDecision(
                decision="WAIT_FOR_INPUT",
                action="NONE",
                reason=(
                    f"期望动作{action}不满足当前状态、"
                    "关闭分支或预算约束。"
                ),
                contradiction_target=state.main_contradiction,
                expected_information_gain="0",
                expected_model_gain="0",
                risks=["状态或工具配置不完整"],
                frozen_contract={},
            )

        return AgentDecision(
            decision="RUN",
            action=action,
            reason=(
                f"当前处于{state.active_layer}；"
                f"主要矛盾为“{state.main_contradiction}”；"
                f"下一缺失能力为"
                f"“{state.next_required_capability}”。"
            ),
            contradiction_target=state.main_contradiction,
            expected_information_gain=(
                "完成当前层级的结构化证据闭环，"
                "避免继续依靠人工记忆选择实验。"
            ),
            expected_model_gain=(
                "本动作若为系统/诊断工具，不承诺直接提分；"
                "只在证据通过后启动模型动作。"
            ),
            risks=[
                "错误的状态输入会导致错误动作",
                "任何A榜样本级规则不得进入B榜通用工具",
            ],
            frozen_contract={
                "only_change": tool.description,
                "kept_fixed": [
                    state.online_version,
                    *state.active_experts,
                ],
                "forbidden": [
                    "并行实验",
                    "Test选阈值",
                    "重启已关闭分支",
                    "LLM自由生成未注册命令",
                ],
                "prediction_changing":
                    tool.prediction_changing,
                "submission_creating":
                    tool.submission_creating,
                "expected_runtime_seconds":
                    tool.expected_runtime_seconds,
            },
        )
