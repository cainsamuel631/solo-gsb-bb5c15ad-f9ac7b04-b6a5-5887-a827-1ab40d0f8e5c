"""半开滑动窗口试算（/window-simulate）引擎与 HTTP 层测试：pytest 运行。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.errors import RequestValidationErrors
from app.main import app
from app.models import WindowSimulateRequest
from app.windowing import window_simulate

client = TestClient(app)


def _run(policies, requests):
    return window_simulate(
        WindowSimulateRequest.model_validate({"policies": policies, "requests": requests})
    )


def _policy(policy_id="A", window_seconds=60, limit=5):
    return {"id": policy_id, "window_seconds": window_seconds, "limit": limit}


def _request(
    request_id,
    timestamp,
    cost,
    policy_chain=("A",),
    principal="alice",
):
    return {
        "id": request_id,
        "timestamp": timestamp,
        "principal": principal,
        "cost": cost,
        "policy_chain": list(policy_chain),
    }


# ---------- 半开滑动窗口 ----------

def test_sliding_window_expires_left_boundary_event():
    resp = _run(
        [_policy("A", window_seconds=10, limit=2)],
        [
            _request("r1", 0, 1),
            _request("r2", 10, 1),
        ],
    )
    d2 = resp.results[1].policy_decisions[0]
    assert d2.window_start == 0
    assert d2.window_end == 10
    assert d2.used_before == 0
    assert d2.remaining_before == 2
    assert resp.results[1].allowed is True
    assert d2.used_after == 1
    assert d2.remaining_after == 1


def test_event_strictly_inside_window_is_retained():
    # t=0.001 在 t=10 的窗口 (0,10] 内，不能再放第二笔；端点 t=0 已过期。
    resp = _run(
        [_policy("A", window_seconds=10, limit=2)],
        [
            _request("old", 0, 1),
            _request("active", 0.001, 1),
            _request("next", 10, 1),
        ],
    )
    decision = resp.results[2].policy_decisions[0]
    assert decision.used_before == 1
    assert resp.results[2].allowed is True
    state = resp.final_states[0]
    assert state.window_start == 0
    assert state.window_end == 10
    assert state.principals[0].used == 2
    assert state.principals[0].event_count == 2


def test_window_limit_rejected_at_boundary_remaining():
    resp = _run(
        [_policy("A", window_seconds=10, limit=3)],
        [
            _request("r1", 1, 2),
            _request("r2", 5, 2),
        ],
    )
    r1, r2 = resp.results
    assert r1.allowed is True
    assert r2.allowed is False
    assert r2.rejected_by == "A"
    assert r2.reason == "window_limit_exceeded"
    d = r2.policy_decisions[0]
    assert d.window_start == -5
    assert d.used_before == 2
    assert d.remaining_before == 1
    assert d.used_after == 2
    assert d.remaining_after == 1


def test_cost_exceeds_policy_limit_even_with_empty_window():
    resp = _run([_policy("A", limit=5)], [_request("r1", 0, 6)])
    result = resp.results[0]
    assert result.allowed is False
    assert result.rejected_by == "A"
    assert result.reason == "cost_exceeds_limit"
    assert result.policy_decisions[0].used_before == 0
    assert result.policy_decisions[0].remaining_before == 5


# ---------- 多策略与原子性 ----------

def test_policy_chain_requires_all_limits_and_records_atomically():
    resp = _run(
        [
            _policy("minute", window_seconds=60, limit=10),
            _policy("burst", window_seconds=10, limit=5),
        ],
        [
            _request("r1", 0, 5, ("minute", "burst")),
            _request("r2", 1, 6, ("minute", "burst")),
        ],
    )
    r2 = resp.results[1]
    assert r2.allowed is False
    assert r2.rejected_by == "burst"
    assert r2.reason == "cost_exceeds_limit"
    # minute 单独可容纳 11，但 burst 拒绝后 minute 不得预记账
    assert r2.policy_decisions[0].used_before == 5
    assert r2.policy_decisions[0].used_after == 5
    assert r2.policy_decisions[1].used_after == 5

    by_id = {state.policy_id: state for state in resp.final_states}
    assert by_id["minute"].principals[0].used == 5
    assert by_id["burst"].principals[0].used == 5


def test_first_over_limit_policy_follows_chain_order():
    resp = _run(
        [
            _policy("P1", limit=10),
            _policy("P2", limit=1),
            _policy("P3", limit=1),
        ],
        [_request("r1", 0, 2, ("P1", "P2", "P3"))],
    )
    result = resp.results[0]
    assert result.allowed is False
    assert result.rejected_by == "P2"
    assert [d.policy_id for d in result.policy_decisions] == ["P1", "P2", "P3"]
    # P3 虽也不足，但首个超限策略已经是 P2；所有策略均不记账。
    assert all(d.used_after == 0 for d in result.policy_decisions)


def test_allowed_request_records_cost_on_every_referenced_policy():
    resp = _run(
        [_policy("P1", limit=3), _policy("P2", limit=4)],
        [_request("r1", 0, 3, ("P1", "P2"))],
    )
    assert resp.allowed_count == 1
    assert [d.used_after for d in resp.results[0].policy_decisions] == [3, 3]
    assert [d.remaining_after for d in resp.results[0].policy_decisions] == [0, 1]


# ---------- principal 隔离、排序与终态 ----------

def test_policy_windows_are_isolated_per_principal():
    resp = _run(
        [_policy("A", window_seconds=60, limit=5)],
        [
            _request("alpha1", 0, 4, principal="alpha"),
            _request("beta1", 0, 4, principal="beta"),
            _request("alpha2", 1, 2, principal="alpha"),
            _request("beta2", 1, 1, principal="beta"),
        ],
    )
    assert [r.allowed for r in resp.results] == [True, True, False, True]
    rejected = resp.results[2]
    assert rejected.rejected_by == "A"
    assert rejected.reason == "window_limit_exceeded"

    state = resp.final_states[0]
    assert state.window_start == -59
    assert state.window_end == 1
    principals = {row.principal: row for row in state.principals}
    assert principals["alpha"].used == 4
    assert principals["alpha"].remaining == 1
    assert principals["alpha"].event_count == 1
    assert principals["beta"].used == 5
    assert principals["beta"].remaining == 0
    assert principals["beta"].event_count == 2


def test_requests_are_processed_chronologically_and_results_keep_original_order():
    resp = _run(
        [_policy("A", window_seconds=10, limit=2)],
        [
            _request("late", 9, 2),
            _request("early", 0, 1),
        ],
    )
    assert [r.request_id for r in resp.results] == ["late", "early"]
    assert [r.index for r in resp.results] == [0, 1]
    # 实际先处理 early：t=9 时 early 仍在 ( -1, 9] 内，占用 1，late 再要 2 被拒。
    assert resp.results[0].allowed is False
    assert resp.results[0].policy_decisions[0].used_before == 1
    assert resp.results[1].allowed is True


def test_same_timestamp_requests_keep_input_order():
    resp = _run(
        [_policy("A", window_seconds=10, limit=1)],
        [
            _request("r1", 0, 1),
            _request("r2", 0, 1),
        ],
    )
    assert resp.results[0].allowed is True
    assert resp.results[1].allowed is False
    assert resp.results[1].rejected_by == "A"


def test_final_states_expire_events_and_include_all_policies():
    resp = _run(
        [
            _policy("short", window_seconds=5, limit=10),
            _policy("long", window_seconds=20, limit=10),
            _policy("unused", window_seconds=30, limit=10),
        ],
        [
            _request("old", 0, 3, ("short", "long"), principal="p"),
            _request("new", 10, 4, ("short", "long"), principal="p"),
        ],
    )
    states = {state.policy_id: state for state in resp.final_states}
    assert resp.final_timestamp == 10
    assert states["short"].window_start == 5
    assert states["short"].principals[0].used == 4
    assert states["short"].principals[0].event_count == 1
    assert states["long"].window_start == -10
    assert states["long"].principals[0].used == 7
    assert states["long"].principals[0].event_count == 2
    assert states["unused"].principals == []
    assert states["unused"].window_start == -20


# ---------- Pydantic / 跨字段校验 ----------

def _validation_error(payload):
    with pytest.raises(RequestValidationErrors) as exc:
        WindowSimulateRequest.model_validate(payload)
    return exc.value.errors


def test_duplicate_policy_id_rejected():
    errors = _validation_error(
        {
            "policies": [_policy("A"), _policy("A", window_seconds=10)],
            "requests": [_request("r", 0, 1)],
        }
    )
    assert any(e["type"] == "duplicate_policy" for e in errors)


def test_unknown_policy_reference_rejected():
    errors = _validation_error(
        {
            "policies": [_policy("A")],
            "requests": [_request("r", 0, 1, ("A", "X"))],
        }
    )
    assert any(e["type"] == "unknown_policy_ref" and e["input"] == "X" for e in errors)


def test_duplicate_request_id_rejected():
    errors = _validation_error(
        {
            "policies": [_policy("A")],
            "requests": [_request("same", 0, 1), _request("same", 1, 1)],
        }
    )
    assert any(e["type"] == "duplicate_request" for e in errors)


def test_duplicate_policy_within_chain_rejected_by_pydantic():
    with pytest.raises(ValidationError):
        WindowSimulateRequest.model_validate(
            {
                "policies": [_policy("A")],
                "requests": [_request("r", 0, 1, ("A", "A"))],
            }
        )


@pytest.mark.parametrize(
    "policy",
    [
        {"id": "A", "window_seconds": 0, "limit": 1},
        {"id": "A", "window_seconds": 1, "limit": -1},
        {"id": "A", "window_seconds": float("nan"), "limit": 1},
    ],
)
def test_invalid_policy_values_rejected(policy):
    with pytest.raises(ValidationError):
        WindowSimulateRequest.model_validate(
            {"policies": [policy], "requests": [_request("r", 0, 1)]}
        )


def test_request_requires_nonempty_principal_and_positive_cost():
    with pytest.raises(ValidationError):
        WindowSimulateRequest.model_validate(
            {
                "policies": [_policy("A")],
                "requests": [
                    {
                        "id": "r",
                        "timestamp": 0,
                        "principal": "",
                        "cost": 1,
                        "policy_chain": ["A"],
                    }
                ],
            }
        )


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        WindowSimulateRequest.model_validate(
            {
                "policies": [{**_policy("A"), "unexpected": True}],
                "requests": [_request("r", 0, 1)],
            }
        )


# ---------- HTTP ----------

def test_http_window_simulate_ok():
    response = client.post(
        "/window-simulate",
        json={
            "policies": [
                {"id": "burst", "window_seconds": 10, "limit": 5},
                {"id": "minute", "window_seconds": 60, "limit": 10},
            ],
            "requests": [
                {
                    "id": "r1",
                    "timestamp": 0,
                    "principal": "tenant-a",
                    "cost": 4,
                    "policy_chain": ["burst", "minute"],
                },
                {
                    "id": "r2",
                    "timestamp": 1,
                    "principal": "tenant-a",
                    "cost": 2,
                    "policy_chain": ["burst", "minute"],
                },
                {
                    "id": "r3",
                    "timestamp": 1,
                    "principal": "tenant-b",
                    "cost": 5,
                    "policy_chain": ["burst"],
                },
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["allowed_count"] == 2
    assert data["rejected_count"] == 1
    assert data["results"][1]["allowed"] is False
    assert data["results"][1]["rejected_by"] == "burst"
    assert data["results"][1]["reason"] == "window_limit_exceeded"
    assert data["results"][2]["allowed"] is True

    burst = next(state for state in data["final_states"] if state["policy_id"] == "burst")
    principals = {row["principal"]: row for row in burst["principals"]}
    assert principals["tenant-a"]["used"] == 4
    assert principals["tenant-b"]["used"] == 5


def test_http_window_validation_error_is_422():
    response = client.post(
        "/window-simulate",
        json={
            "policies": [{"id": "A", "window_seconds": 10, "limit": 5}],
            "requests": [
                {
                    "id": "r",
                    "timestamp": 0,
                    "principal": "alice",
                    "cost": 1,
                    "policy_chain": ["missing"],
                }
            ],
        },
    )
    assert response.status_code == 422
    assert any(e["type"] == "unknown_policy_ref" for e in response.json()["detail"])
