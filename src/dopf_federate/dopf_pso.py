"""PSO optimization for PV reactive power dispatch."""

import numpy as np


def compute_sensitivity_matrix(y_bus, bus_ids, base_voltages=None, base_power_mva=100.0):
    """Compute voltage sensitivity dV_pu/dQ_kVAR.

    When base_voltages is provided, converts Y-bus to per-unit before
    inverting so the sensitivity maps kVAR injections directly to
    per-unit voltage changes.

    Parameters
    ----------
    y_bus : np.ndarray
        Complex admittance matrix (n x n) in Siemens.
    bus_ids : list of str
        Bus identifiers corresponding to Y-bus rows/columns.
    base_voltages : np.ndarray or None
        Per-bus base voltage magnitudes in Volts. If None, uses raw Y-bus
        (legacy behavior, units will not be consistent with per-unit voltages).
    base_power_mva : float
        Base power in MVA (default 100).

    Returns
    -------
    np.ndarray
        Sensitivity matrix (n x n). When base_voltages is provided,
        maps kVAR to per-unit voltage change.
    """
    if base_voltages is not None:
        S_base_VA = base_power_mva * 1e6
        V = np.asarray(base_voltages, dtype=float)
        Y_pu = np.outer(V, V) * y_bus / S_base_VA
        Z_pu = np.linalg.inv(Y_pu)
        S_base_kVA = base_power_mva * 1000.0
        return np.imag(Z_pu) / S_base_kVA
    else:
        y_inv = np.linalg.inv(y_bus)
        return np.imag(y_inv)


def compute_pv_bounds(pv_capacity_kw, pv_active_kw=None):
   
    bounds = []
    for i, cap in enumerate(pv_capacity_kw):
        p = pv_active_kw[i] if pv_active_kw is not None else 0.0
        p = min(abs(p), cap)
        q_max = np.sqrt(max(cap**2 - p**2, 0.0))
        bounds.append((-q_max, q_max))
    return bounds


def pso_optimize(
    pv_buses,
    pv_capacity_kw,
    base_voltages_pu,
    sensitivity_matrix,
    pv_bus_indices,
    num_particles=30,
    max_iterations=30,
):
    """Run PSO to find optimal PV kVAR setpoints.

    Parameters
    ----------
    pv_buses : list of str
        PV system bus IDs.
    pv_capacity_kw : list of float
        Rated capacity per PV system.
    base_voltages_pu : np.ndarray
        Current bus voltages in per-unit (n_buses,).
    sensitivity_matrix : np.ndarray
        dV/dQ sensitivity matrix (n_buses x n_buses).
    pv_bus_indices : list of int
        Indices into sensitivity_matrix for each PV bus.
    num_particles : int
        Number of PSO particles.
    max_iterations : int
        Maximum PSO iterations.

    Returns
    -------
    np.ndarray
        Optimal kVAR values for each PV system.
    """
    n_pv = len(pv_buses)
    if n_pv == 0:
        return np.array([])

    bounds = compute_pv_bounds(pv_capacity_kw)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    inertia = 0.9
    cognitive = 1.5
    social = 1.5

    positions = np.zeros((num_particles, n_pv))
    for d in range(n_pv):
        positions[:, d] = np.random.uniform(lb[d], ub[d], num_particles)
    velocities = np.random.uniform(-1, 1, (num_particles, n_pv))

    best_positions = positions.copy()
    best_costs = np.array(
        [
            _evaluate(pos, base_voltages_pu, sensitivity_matrix, pv_bus_indices)
            for pos in positions
        ]
    )

    global_best_idx = np.argmin(best_costs)
    global_best_pos = best_positions[global_best_idx].copy()
    global_best_cost = best_costs[global_best_idx]

    for _ in range(max_iterations):
        for i in range(num_particles):
            r1 = np.random.rand(n_pv)
            r2 = np.random.rand(n_pv)
            velocities[i] = (
                inertia * velocities[i]
                + cognitive * r1 * (best_positions[i] - positions[i])
                + social * r2 * (global_best_pos - positions[i])
            )
            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], lb, ub)

            cost = _evaluate(
                positions[i], base_voltages_pu, sensitivity_matrix, pv_bus_indices
            )
            if cost < best_costs[i]:
                best_positions[i] = positions[i].copy()
                best_costs[i] = cost

            if best_costs[i] < global_best_cost:
                global_best_cost = best_costs[i]
                global_best_pos = best_positions[i].copy()

    return global_best_pos


def _evaluate(delta_q, base_voltages_pu, sensitivity_matrix, pv_bus_indices):
    """Evaluate cost for a candidate kVAR injection vector.

    Parameters
    ----------
    delta_q : np.ndarray
        kVAR injection at each PV bus (n_pv,).
    base_voltages_pu : np.ndarray
        Current bus voltages in per-unit (n_buses,).
    sensitivity_matrix : np.ndarray
        dV/dQ sensitivity matrix (n_buses x n_buses).
    pv_bus_indices : list of int
        Indices into sensitivity_matrix for each PV bus.

    Returns
    -------
    float
        Cost value.
    """
    n_buses = len(base_voltages_pu)
    full_dq = np.zeros(n_buses)
    for k, idx in enumerate(pv_bus_indices):
        if 0 <= idx < n_buses:
            full_dq[idx] = delta_q[k]

    dv = sensitivity_matrix @ full_dq
    v_est = base_voltages_pu + dv

    under = np.maximum(0.95 - v_est, 0.0)
    over = np.maximum(v_est - 1.05, 0.0)
    voltage_penalty = 1000.0 * float(np.sum(under**2 + over**2))

    return voltage_penalty + float(np.sum(np.abs(delta_q)))
