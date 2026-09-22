"""半开滑动窗口配额试算：按主体累计事件消耗并做链上原子记账。"""
from __future__ import annotations

from collections import deque

from app.models import (
    EPS,
    WindowPolicyDecision,
    WindowPolicyFinalState,
    WindowPolicySpec,
    WindowPrincipalState,
    WindowSimResult,
    WindowSimulateRequest,
    WindowSimulateResponse,
)
from app.simulation import _round


def _prune(events: deque[tuple[float, float]], window_start: float) -> None:
    """移除半开窗口 (window_start, now] 左端以外（timestamp <= start）的事件。"""
    while events and events[0][0] <= window_start:
        events.popleft()


def window_simulate(payload: WindowSimulateRequest) -> WindowSimulateResponse:
    if not payload.policies:
        raise ValueError("policies 不能为空")
    if not payload.requests:
        raise ValueError("requests 不能为空")

    policies: dict[str, WindowPolicySpec] = {policy.id: policy for policy in payload.policies}
    window_seconds = {policy.id: float(policy.window_seconds) for policy in payload.policies}
    limit = {policy.id: float(policy.limit) for policy in payload.policies}

    # (policy_id, principal) -> 仍可能参与当前/后续窗口的已接受事件；
    # 请求按时间顺序处理，因此 deque 中的 timestamp 单调不减。
    events: dict[tuple[str, str], deque[tuple[float, float]]] = {}
    principal_order: dict[str, list[str]] = {policy.id: [] for policy in payload.policies}

    def bucket_for(policy_id: str, principal: str) -> deque[tuple[float, float]]:
        key = (policy_id, principal)
        if key not in events:
            events[key] = deque()
            principal_order[policy_id].append(principal)
        return events[key]

    order = sorted(
        range(len(payload.requests)),
        key=lambda i: (payload.requests[i].timestamp, i),
    )
    results_by_index: dict[int, WindowSimResult] = {}

    for i in order:
        request = payload.requests[i]
        now = float(request.timestamp)
        cost = float(request.cost)

        # 先统一移除本请求将检查的各主体/策略中已落在窗口左端或更早的事件，
        # 再基于清理后的占用做判定；拒绝时也不会恢复这些已过期事件。
        decisions: list[WindowPolicyDecision] = []
        rejected_by: str | None = None
        reason: str | None = None

        for policy_id in request.policy_chain:
            bucket = bucket_for(policy_id, request.principal)
            start = now - window_seconds[policy_id]
            _prune(bucket, start)

            used_before = sum(amount for _, amount in bucket)
            remaining_before = limit[policy_id] - used_before
            is_rejecting_policy = rejected_by is None and used_before + cost > limit[policy_id] + EPS
            allowed_here = rejected_by is None and not is_rejecting_policy
            if allowed_here:
                used_after = used_before + cost
                remaining_after = remaining_before - cost
            else:
                # 即使本策略单独足够，链上已有先前策略拒绝，也不得预记账。
                used_after = used_before
                remaining_after = remaining_before
                if is_rejecting_policy:
                    rejected_by = policy_id
                    if cost > limit[policy_id] + EPS:
                        reason = "cost_exceeds_limit"
                    else:
                        reason = "window_limit_exceeded"

            decisions.append(
                WindowPolicyDecision(
                    policy_id=policy_id,
                    principal=request.principal,
                    window_start=_round(start),
                    window_end=_round(now),
                    limit=_round(limit[policy_id]),
                    used_before=_round(used_before),
                    used_after=_round(used_after),
                    remaining_before=_round(remaining_before),
                    remaining_after=_round(remaining_after),
                )
            )

        allowed = rejected_by is None
        if allowed:
            # 全链通过后才原子记账：所有策略在同一判定后登记同一事件。
            for policy_id in request.policy_chain:
                bucket_for(policy_id, request.principal).append((now, cost))

        results_by_index[i] = WindowSimResult(
            request_id=request.id,
            index=i,
            timestamp=_round(now),
            principal=request.principal,
            cost=_round(cost),
            allowed=allowed,
            rejected_by=rejected_by,
            reason=reason,
            policy_decisions=decisions,
        )

    final_timestamp = max(float(request.timestamp) for request in payload.requests)
    final_states: list[WindowPolicyFinalState] = []
    for policy in payload.policies:
        start = final_timestamp - float(policy.window_seconds)
        principal_states: list[WindowPrincipalState] = []

        for principal in principal_order[policy.id]:
            bucket = events[(policy.id, principal)]
            _prune(bucket, start)
            used = sum(amount for _, amount in bucket)
            principal_states.append(
                WindowPrincipalState(
                    principal=principal,
                    used=_round(used),
                    remaining=_round(float(policy.limit) - used),
                    event_count=len(bucket),
                )
            )

        final_states.append(
            WindowPolicyFinalState(
                policy_id=policy.id,
                window_seconds=_round(policy.window_seconds),
                limit=_round(policy.limit),
                window_start=_round(start),
                window_end=_round(final_timestamp),
                principals=principal_states,
            )
        )

    results = [results_by_index[i] for i in range(len(payload.requests))]
    allowed_count = sum(1 for result in results if result.allowed)

    return WindowSimulateResponse(
        allowed_count=allowed_count,
        rejected_count=len(results) - allowed_count,
        results=results,
        final_timestamp=_round(final_timestamp),
        final_states=final_states,
    )
