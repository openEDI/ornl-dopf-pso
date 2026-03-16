"""Tests for DOPF federate."""

import json

import numpy as np
from dopf_federate.dopf_federate import build_pv_commands
from dopf_federate.dopf_pso import (
    _evaluate,
    compute_pv_bounds,
    compute_sensitivity_matrix,
    pso_optimize,
)
from oedisi.types.data_types import CommandList


def _make_simple_ybus(n=5):
    """Create a simple Y-bus matrix for testing."""
    y_bus = np.zeros((n, n), dtype=complex)
    for i in range(n):
        y_bus[i, i] = complex(10.0, -20.0)
        if i > 0:
            y_bus[i, i - 1] = complex(-5.0, 10.0)
            y_bus[i - 1, i] = complex(-5.0, 10.0)
    return y_bus


def test_compute_sensitivity_matrix():
    """Sensitivity matrix should be real-valued and square."""
    y_bus = _make_simple_ybus(5)
    bus_ids = ["1.1", "2.1", "3.1", "4.1", "5.1"]
    sens = compute_sensitivity_matrix(y_bus, bus_ids)
    assert sens.shape == (5, 5)
    assert np.all(np.isfinite(sens))
    assert sens.dtype == np.float64


def test_compute_sensitivity_matrix_diagonal_dominance():
    """Diagonal entries should have larger magnitude (self-sensitivity)."""
    y_bus = _make_simple_ybus(4)
    bus_ids = ["a", "b", "c", "d"]
    sens = compute_sensitivity_matrix(y_bus, bus_ids)
    for i in range(4):
        assert abs(sens[i, i]) >= abs(sens[i, (i + 1) % 4])


def test_compute_pv_bounds_basic():
    """Without active power info, full capacity available for reactive."""
    bounds = compute_pv_bounds([100.0, 200.0])
    assert len(bounds) == 2
    assert abs(bounds[0][1] - 100.0) < 1e-6
    assert abs(bounds[1][1] - 200.0) < 1e-6
    for q_min, q_max in bounds:
        assert abs(q_min + q_max) < 1e-6


def test_compute_pv_bounds_with_active():
    """With active < capacity, reactive headroom should exist."""
    bounds = compute_pv_bounds([100.0], [60.0])
    q_min, q_max = bounds[0]
    assert q_max > 0
    assert abs(q_min + q_max) < 1e-6
    expected = np.sqrt(100.0**2 - 60.0**2)
    assert abs(q_max - expected) < 1e-6


def test_evaluate_no_violations():
    """No violations should return sum of |delta_q|."""
    n = 5
    sens = np.eye(n) * 0.001
    base_v = np.ones(n)
    delta_q = np.array([10.0, -5.0])
    pv_indices = [1, 3]
    cost = _evaluate(delta_q, base_v, sens, pv_indices)
    assert abs(cost - 15.0) < 1e-6


def test_evaluate_with_violations():
    """Violations should produce cost >= 1000."""
    n = 5
    sens = np.eye(n) * 0.1
    base_v = np.ones(n) * 0.94  # already below 0.95
    delta_q = np.zeros(2)
    pv_indices = [0, 1]
    cost = _evaluate(delta_q, base_v, sens, pv_indices)
    assert cost >= 1000.0


def test_pso_optimize_returns_correct_shape():
    """PSO should return one kVAR value per PV bus."""
    n = 5
    pv_buses = ["1.1", "3.1"]
    pv_cap = [100.0, 100.0]
    base_v = np.ones(n)
    sens = np.eye(n) * 0.001
    pv_idx = [0, 2]

    result = pso_optimize(
        pv_buses,
        pv_cap,
        base_v,
        sens,
        pv_idx,
        num_particles=10,
        max_iterations=5,
    )
    assert result.shape == (2,)


def test_pso_optimize_empty_pv():
    """Empty PV list should return empty array."""
    result = pso_optimize([], [], np.ones(5), np.eye(5) * 0.001, [])
    assert len(result) == 0


def test_pso_optimize_minimizes_reactive():
    """With no violations, PSO should drive kVAR toward zero."""
    np.random.seed(42)
    n = 5
    pv_buses = ["1.1", "3.1"]
    pv_cap = [100.0, 100.0]
    base_v = np.ones(n)
    sens = np.eye(n) * 0.0001
    pv_idx = [0, 2]

    result = pso_optimize(
        pv_buses,
        pv_cap,
        base_v,
        sens,
        pv_idx,
        num_particles=20,
        max_iterations=20,
    )
    assert np.sum(np.abs(result)) < 50.0


def test_pso_optimize_corrects_violations():
    """With undervoltage, PSO should inject positive kVAR to raise voltage."""
    np.random.seed(42)
    n = 3
    pv_buses = ["1.1"]
    pv_cap = [500.0]
    base_v = np.array([0.94, 0.96, 0.98])
    sens = np.eye(n) * 0.0005
    pv_idx = [0]

    result = pso_optimize(
        pv_buses,
        pv_cap,
        base_v,
        sens,
        pv_idx,
        num_particles=30,
        max_iterations=30,
    )
    v_after = base_v + sens @ np.array([result[0], 0, 0])
    assert v_after[0] >= 0.949


def test_build_pv_commands():
    """Should produce one Command per PV bus."""
    pv_buses = ["7.1", "29.1", "55.1"]
    kvar = np.array([10.5, -20.3, 0.0])
    cmd_list = build_pv_commands(pv_buses, kvar)
    assert len(cmd_list.root) == 3
    assert cmd_list.root[0].obj_name == "PVSystem.7"
    assert cmd_list.root[0].obj_property == "kvar"
    assert cmd_list.root[0].val == "10.5"
    assert cmd_list.root[1].obj_name == "PVSystem.29"
    assert cmd_list.root[1].val == "-20.3"


def test_build_pv_commands_serialization():
    """CommandList should serialize to valid JSON."""
    pv_buses = ["7.1", "29.1"]
    kvar = np.array([15.0, -10.0])
    cmd_list = build_pv_commands(pv_buses, kvar)
    json_str = cmd_list.model_dump_json()
    parsed = json.loads(json_str)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[0]["obj_name"] == "PVSystem.7"


def test_build_pv_commands_no_phase_suffix():
    """Bus IDs without dots should be used directly."""
    cmd_list = build_pv_commands(["bus113"], np.array([5.0]))
    assert cmd_list.root[0].obj_name == "PVSystem.bus113"
