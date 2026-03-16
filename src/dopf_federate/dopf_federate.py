"""DOPF Federate."""

import json
import logging
from datetime import datetime

import helics as h
import numpy as np
from oedisi.types.common import BrokerConfig
from oedisi.types.data_types import (
    Command,
    CommandList,
    PowersImaginary,
    PowersReal,
    Topology,
    VoltagesMagnitude,
)

from .dopf_pso import compute_sensitivity_matrix, pso_optimize

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def build_pv_commands(pv_buses, kvar_values):
    """Build CommandList setting kvar on each PVSystem.

    Parameters
    ----------
    pv_buses : list of str
        PV bus identifiers.
    kvar_values : np.ndarray
        kVAR setpoint for each PV system.

    Returns
    -------
    CommandList
        Commands for the feeder.
    """
    commands = []
    for bus, kvar in zip(pv_buses, kvar_values):
        bus_name = bus.split(".")[0] if "." in bus else bus
        commands.append(
            Command(
                obj_name=f"PVSystem.{bus_name}",
                obj_property="kvar",
                val=str(float(kvar)),
            )
        )
    return CommandList(root=commands)


class DOPFFederate:

    def __init__(
        self,
        federate_name,
        input_mapping,
        broker_config: BrokerConfig,
        pv_buses=None,
        pv_capacity_kw=None,
        algorithm_parameters=None,
    ):
        self.pv_buses = pv_buses or []
        self.pv_capacity_kw = pv_capacity_kw or [150.0] * len(self.pv_buses)
        self.algorithm_parameters = algorithm_parameters or {}
        logger.info(f"PV buses: {self.pv_buses}")
        logger.info(f"PV capacity (kW): {self.pv_capacity_kw}")

        deltat = 1
        fedinfo = h.helicsCreateFederateInfo()
        h.helicsFederateInfoSetBroker(fedinfo, broker_config.broker_ip)
        h.helicsFederateInfoSetBrokerPort(fedinfo, broker_config.broker_port)
        fedinfo.core_name = federate_name
        fedinfo.core_type = h.HELICS_CORE_TYPE_ZMQ
        fedinfo.core_init = "--federates=1"
        h.helicsFederateInfoSetTimeProperty(
            fedinfo, h.helics_property_time_delta, deltat
        )

        self.vfed = h.helicsCreateValueFederate(federate_name, fedinfo)
        logger.info("Value federate created")

        self.sub_power_P = self.vfed.register_subscription(
            input_mapping["powers_real"], "W"
        )
        self.sub_power_Q = self.vfed.register_subscription(
            input_mapping["powers_imag"], "W"
        )
        self.sub_topology = self.vfed.register_subscription(
            input_mapping["topology"], ""
        )
        self.sub_voltages_mag = self.vfed.register_subscription(
            input_mapping["voltages_magnitude"], ""
        )

        self.pub_change_commands = self.vfed.register_publication(
            "change_commands", h.HELICS_DATA_TYPE_STRING, ""
        )

        self.sensitivity_matrix = None
        self.bus_ids = None
        self.pv_bus_indices = None

    def _build_sensitivity(self, topology):
        """Build voltage sensitivity matrix from topology Y-bus."""
        admittance = topology.admittance
        bus_ids = list(admittance.ids)
        n = len(bus_ids)

        y_bus = np.zeros((n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                entry = admittance.admittance_matrix[i][j]
                y_bus[i, j] = complex(entry[0], entry[1])

        self.bus_ids = [b.lower() for b in bus_ids]

        # Use per-bus base voltages for proper per-unit sensitivity
        base_voltages = None
        if hasattr(topology, "base_voltage_magnitudes"):
            base_voltages = np.array(topology.base_voltage_magnitudes.values)
            self._topology_base_voltages = base_voltages

        self.sensitivity_matrix = compute_sensitivity_matrix(
            y_bus, self.bus_ids, base_voltages=base_voltages
        )

        bus_index_map = {b: i for i, b in enumerate(self.bus_ids)}
        self.pv_bus_indices = []
        for pv_bus in self.pv_buses:
            pv_lower = pv_bus.lower()
            idx = bus_index_map.get(pv_lower, -1)
            self.pv_bus_indices.append(idx)
            if idx < 0:
                logger.warning(f"PV bus {pv_bus} not found in topology")

        logger.info(
            f"Built sensitivity matrix: {n} buses, "
            f"{sum(1 for i in self.pv_bus_indices if i >= 0)} PV buses mapped"
        )

    def run(self):
        """Main loop: receive data, run PSO, publish commands."""
        logger.info(f"Federate connected: {datetime.now()}")
        self.vfed.enter_executing_mode()
        granted_time = h.helicsFederateRequestTime(self.vfed, h.HELICS_TIME_MAXTIME)

        num_particles = self.algorithm_parameters.get("num_particles", 30)
        max_iterations = self.algorithm_parameters.get("max_iterations", 30)
        timestep_count = 0
        topology_built = False

        while granted_time < h.HELICS_TIME_MAXTIME:
            if not self.sub_power_P.is_updated():
                granted_time = h.helicsFederateRequestTime(
                    self.vfed, h.HELICS_TIME_MAXTIME
                )
                continue

            timestep_count += 1
            logger.info(f"TIMESTEP {timestep_count} | HELICS Time: {granted_time}")

            if not topology_built and self.sub_topology.is_updated():
                try:
                    topology = Topology.model_validate(self.sub_topology.json)
                    self._build_sensitivity(topology)
                    topology_built = True
                except Exception as e:
                    logger.warning(f"Could not build sensitivity from topology: {e}")

            base_voltages_pu = None
            if self.sub_voltages_mag.is_updated():
                try:
                    vmag = VoltagesMagnitude.model_validate(self.sub_voltages_mag.json)
                    base_volt_raw = np.array(vmag.values)
                    if (
                        topology_built
                        and hasattr(self, "_topology_base_voltages")
                        and self._topology_base_voltages is not None
                    ):
                        base_voltages_pu = base_volt_raw / self._topology_base_voltages
                    else:
                        v_base = np.median(base_volt_raw)
                        base_voltages_pu = (
                            base_volt_raw / v_base if v_base > 0 else None
                        )
                except Exception as e:
                    logger.warning(f"Could not parse voltages: {e}")

            if topology_built and self.sub_topology.is_updated():
                try:
                    topo = Topology.model_validate(self.sub_topology.json)
                    self._topology_base_voltages = np.array(
                        topo.base_voltage_magnitudes.values
                    )
                except Exception:
                    pass

            if self.sensitivity_matrix is None or base_voltages_pu is None:
                cmd_list = CommandList(root=[])
                self.pub_change_commands.publish(cmd_list.model_dump_json())
                granted_time = h.helicsFederateRequestTime(
                    self.vfed, h.HELICS_TIME_MAXTIME
                )
                continue

            kvar_opt = pso_optimize(
                self.pv_buses,
                self.pv_capacity_kw,
                base_voltages_pu,
                self.sensitivity_matrix,
                self.pv_bus_indices,
                num_particles=num_particles,
                max_iterations=max_iterations,
            )

            cmd_list = build_pv_commands(self.pv_buses, kvar_opt)
            self.pub_change_commands.publish(cmd_list.model_dump_json())
            logger.info(
                f"Published {len(cmd_list.root)} commands: "
                + ", ".join(f"{c.obj_name}={c.val}" for c in cmd_list.root)
            )

            granted_time = h.helicsFederateRequestTime(self.vfed, h.HELICS_TIME_MAXTIME)

        self.stop()

    def stop(self):
        h.helicsFederateDisconnect(self.vfed)
        h.helicsFederateFree(self.vfed)
        h.helicsCloseLibrary()
        logger.info(f"Federate disconnected: {datetime.now()}")


def run_simulator(broker_config: BrokerConfig):
    """Entry point called by server.py."""
    with open("static_inputs.json") as f:
        config = json.load(f)
        federate_name = config["name"]
        pv_buses = config.get("pv_buses", [])
        pv_capacity_kw = config.get("pv_capacity_kw", [150.0] * len(pv_buses))
        if isinstance(pv_capacity_kw, (int, float)):
            pv_capacity_kw = [float(pv_capacity_kw)] * len(pv_buses)
        algorithm_parameters = config.get("algorithm_parameters", {})

    with open("input_mapping.json") as f:
        input_mapping = json.load(f)

    try:
        sfed = DOPFFederate(
            federate_name,
            input_mapping,
            broker_config,
            pv_buses=pv_buses,
            pv_capacity_kw=pv_capacity_kw,
            algorithm_parameters=algorithm_parameters,
        )
        logger.info("DOPF federate created")
    except h.HelicsException as e:
        logger.error(f"Failed to create HELICS Value Federate: {str(e)}")
        return

    sfed.run()


if __name__ == "__main__":
    run_simulator(BrokerConfig(broker_ip="127.0.0.1"))
