from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cluster.fabric.evaluator import ManagementConfig, build_fabric_payloads, build_nmx_partition_lab_payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_vllm_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["concurrency", "total_token_throughput", "p99_ttft_ms"],
        )
        writer.writeheader()
        writer.writerow({"concurrency": 1, "total_token_throughput": 200.0, "p99_ttft_ms": 40.0})
        writer.writerow({"concurrency": 8, "total_token_throughput": 900.0, "p99_ttft_ms": 95.0})


def _seed_meta(
    structured: Path,
    run_id: str,
    label: str,
    *,
    topo: str,
    ibstat: str,
    rdma_link: str,
    ibv_devinfo: str,
    ethtool: str,
) -> None:
    _write_json(
        structured / f"{run_id}_{label}_meta.json",
        {
            "commands": {
                "nvidia_smi_topo": {"stdout": topo},
                "ibstat": {"stdout": ibstat},
                "rdma_link": {"stdout": rdma_link},
                "ibv_devinfo": {"stdout": ibv_devinfo},
                "ethtool": {"stdout": ethtool},
            }
        },
    )


def test_build_fabric_payloads_handles_localhost_nvlink_only(tmp_path: Path) -> None:
    run_id = "2026-03-16_fabric_localhost"
    label = "localhost"
    run_dir = tmp_path / "runs" / run_id
    structured = run_dir / "structured"
    structured.mkdir(parents=True, exist_ok=True)

    _seed_meta(
        structured,
        run_id,
        label,
        topo="GPU0\t X \t NV1",
        ibstat="CA 'mlx5_0'\n\tState: Active\n\tLink layer: Ethernet\n",
        rdma_link="",
        ibv_devinfo="",
        ethtool="Settings for eth0:\n\tSpeed: 25000Mb/s\n",
    )
    _write_json(structured / f"{run_id}_{label}_meta_nvlink_topology.json", {"summary": {"gpu_count": 1, "nvlink_pair_count": 1}})
    _write_json(structured / f"{run_id}_node1_nccl.json", {"results": [{"algbw_gbps": 2400.0, "busbw_gbps": 0.0}]})
    _write_vllm_csv(structured / f"{run_id}_{label}_vllm_serve_sweep.csv")

    payloads = build_fabric_payloads(
        run_id=run_id,
        run_dir=run_dir,
        primary_label=label,
        management=ManagementConfig(
            ib_mgmt_host=None,
            ib_mgmt_user=None,
            ib_mgmt_ssh_key=None,
            nmx_url=None,
            nmx_token=None,
            cumulus_hosts=(),
            cumulus_user=None,
            cumulus_ssh_key=None,
        ),
    )

    capabilities = payloads["fabric_capability_matrix"]["families"]
    verification = payloads["fabric_verification"]["families"]
    assert capabilities["nvlink"]["present"] is True
    assert capabilities["infiniband"]["present"] is False
    assert capabilities["spectrum-x"]["present"] is False
    assert verification["nvlink"]["completeness"] == "runtime_verified"
    assert verification["infiniband"]["completeness"] == "not_present"
    assert payloads["fabric_scorecard"]["status"] == "partial"


def test_build_fabric_payloads_marks_full_stack_when_management_and_runtime_are_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "2026-03-16_fabric_fullstack"
    label = "node1"
    run_dir = tmp_path / "runs" / run_id
    structured = run_dir / "structured"
    structured.mkdir(parents=True, exist_ok=True)

    _seed_meta(
        structured,
        run_id,
        label,
        topo="GPU0\t X \t NV4\tGPU1",
        ibstat="CA 'mlx5_0'\n\tState: Active\n\tLink layer: InfiniBand\n",
        rdma_link="link mlx5_0/1 state ACTIVE netdev eth0\n",
        ibv_devinfo="hca_id: mlx5_0\n",
        ethtool="Settings for eth0:\n\tSpeed: 400000Mb/s\n",
    )
    _write_json(structured / f"{run_id}_{label}_meta_nvlink_topology.json", {"summary": {"gpu_count": 2, "nvlink_pair_count": 2}})
    _write_json(structured / f"{run_id}_node1_nccl.json", {"results": [{"algbw_gbps": 2400.0, "busbw_gbps": 0.0}]})
    _write_json(structured / f"{run_id}_2nodes_nccl.json", {"results": [{"algbw_gbps": 450.0, "busbw_gbps": 430.0}]})
    _write_json(structured / f"{run_id}_2nodes_alltoall_nccl_alltoall.json", {"results": [{"algbw_gbps": 320.0, "busbw_gbps": 300.0}]})
    _write_json(structured / f"{run_id}_torchrun_connectivity_probe.json", {"world_size": 2})
    _write_vllm_csv(structured / f"{run_id}_{label}_vllm_serve_sweep.csv")

    def fake_runner(command: str, **_: object) -> dict[str, object]:
        if command.startswith("ibhosts"):
            stdout = "Ca : 0x0002 lid 11\nCa : 0x0003 lid 12"
        elif command.startswith("ibswitches"):
            stdout = "Switch : 0x0011 lid 21"
        else:
            stdout = "ok"
        return {"status": "ok", "command": command, "stdout": stdout, "stderr": "", "returncode": 0}

    domain_uuid = "domain-1"
    compute_nodes = [
        {"ID": "node-1", "Name": "node-1", "LocationInfo": {"ChassisID": 1, "SlotID": 1, "HostID": 1, "TrayIndex": 0}, "GpuIDList": ["gpu-1-1", "gpu-1-2"]},
        {"ID": "node-2", "Name": "node-2", "LocationInfo": {"ChassisID": 1, "SlotID": 2, "HostID": 1, "TrayIndex": 1}, "GpuIDList": ["gpu-2-1", "gpu-2-2"]},
        {"ID": "node-3", "Name": "node-3", "LocationInfo": {"ChassisID": 1, "SlotID": 3, "HostID": 1, "TrayIndex": 2}, "GpuIDList": ["gpu-3-1", "gpu-3-2"]},
        {"ID": "node-4", "Name": "node-4", "LocationInfo": {"ChassisID": 1, "SlotID": 4, "HostID": 1, "TrayIndex": 3}, "GpuIDList": ["gpu-4-1", "gpu-4-2"]},
    ]
    gpus = []
    for node_index in range(1, 5):
        for gpu_index in range(1, 3):
            gpus.append(
                {
                    "ID": f"gpu-{node_index}-{gpu_index}",
                    "SystemUID": f"sys-{node_index}",
                    "DeviceID": gpu_index,
                    "DomainUUID": domain_uuid,
                    "PartitionID": 32766 if not (node_index == 4 and gpu_index == 2) else None,
                    "PartitionName": "default" if not (node_index == 4 and gpu_index == 2) else "Unassigned",
                    "LocationInfo": {"ChassisID": 1, "SlotID": node_index, "HostID": 1, "TrayIndex": node_index - 1},
                    "PortIDList": [f"port-gpu-{node_index}-{gpu_index}-1", f"port-gpu-{node_index}-{gpu_index}-2"],
                }
            )
    switches = [
        {"ID": "sw-1a", "DeviceID": 1, "DomainUUID": domain_uuid, "LocationInfo": {"ChassisID": 1, "SlotID": 10, "HostID": 1, "TrayIndex": 20}, "PortIDList": ["p1", "p2"]},
        {"ID": "sw-1b", "DeviceID": 2, "DomainUUID": domain_uuid, "LocationInfo": {"ChassisID": 1, "SlotID": 10, "HostID": 1, "TrayIndex": 20}, "PortIDList": ["p3", "p4"]},
        {"ID": "sw-2a", "DeviceID": 1, "DomainUUID": domain_uuid, "LocationInfo": {"ChassisID": 1, "SlotID": 11, "HostID": 1, "TrayIndex": 21}, "PortIDList": ["p5", "p6"]},
        {"ID": "sw-2b", "DeviceID": 2, "DomainUUID": domain_uuid, "LocationInfo": {"ChassisID": 1, "SlotID": 11, "HostID": 1, "TrayIndex": 21}, "PortIDList": ["p7", "p8"]},
    ]
    switch_nodes = [
        {"ID": "switch-node-1", "DomainUUID": domain_uuid, "SwitchIDList": ["sw-1a", "sw-1b"], "LocationInfo": {"ChassisID": 1, "SlotID": 10, "HostID": 1, "TrayIndex": 20}},
        {"ID": "switch-node-2", "DomainUUID": domain_uuid, "SwitchIDList": ["sw-2a", "sw-2b"], "LocationInfo": {"ChassisID": 1, "SlotID": 11, "HostID": 1, "TrayIndex": 21}},
    ]
    chassis = [
        {"ID": "chassis-1", "Name": "GB200-Rack-1", "DomainUUID": domain_uuid, "Health": "HEALTHY", "LocationInfo": {"ChassisID": 1, "ChassisSerialNumber": "SN12345678901"}},
    ]
    ports = (
        [{"ID": f"gpu-port-{index}", "Type": "GPU", "BaseLID": 100 + index} for index in range(1, 33)]
        + [{"ID": f"switch-port-{index}", "Type": "SWITCH_ACCESS", "BaseLID": None} for index in range(1, 33)]
    )
    partitions = [
        {
            "ID": "partition-default",
            "PartitionID": 32766,
            "Name": "Default",
            "Type": "NVLINK",
            "DomainUUID": domain_uuid,
            "Members": {"locations": ["1.1.1.1", "1.1.1.2", "1.2.1.1", "1.2.1.2", "1.3.1.1", "1.3.1.2", "1.4.1.1"]},
        }
    ]

    def fake_http_json(url: str, token: str | None = None, timeout: int = 10) -> dict[str, object]:
        del token, timeout
        mapping = {
            "services": [{"ID": "service-1", "DomainUUID": domain_uuid}],
            "compute-nodes": compute_nodes,
            "gpus": gpus,
            "switches": switches,
            "switch-nodes": switch_nodes,
            "chassis": chassis,
            "ports": ports,
            "partitions": partitions,
        }
        for suffix, payload in mapping.items():
            if url.endswith(suffix):
                return {"status": "ok", "http_status": 200, "json": payload, "error": ""}
        return {"status": "error", "http_status": 404, "json": [], "error": f"unexpected url {url}"}

    def fake_http_text(url: str, token: str | None = None, timeout: int = 10) -> dict[str, object]:
        del token, timeout
        if url.endswith("/metrics"):
            text = "\n".join(
                [
                    "switch_temperature{switch=\"1\"} 51",
                    "switch_temperature{switch=\"2\"} 52",
                    "PortXmitDataExtended{port=\"1\"} 1000",
                    "PortRcvDataExtended{port=\"1\"} 900",
                    "PortLocalPhysicalErrors{port=\"1\"} 0",
                    "CableInfoTemperature{port=\"1\"} 31",
                    "CableInfoRxPower{port=\"1\"} 1.2",
                    "CableInfoTxPower{port=\"1\"} 1.4",
                ]
            )
            return {"status": "ok", "http_status": 200, "text": text, "error": ""}
        return {"status": "error", "http_status": 404, "text": "", "error": f"unexpected url {url}"}

    monkeypatch.setattr(
        "cluster.fabric.evaluator._http_json",
        fake_http_json,
    )
    monkeypatch.setattr("cluster.fabric.evaluator._http_text", fake_http_text)

    payloads = build_fabric_payloads(
        run_id=run_id,
        run_dir=run_dir,
        primary_label=label,
        management=ManagementConfig(
            ib_mgmt_host="ib-mgmt",
            ib_mgmt_user="ubuntu",
            ib_mgmt_ssh_key="/tmp/key",
            nmx_url="https://nmx.example/api",
            nmx_token="secret",
            cumulus_hosts=("leaf01",),
            cumulus_user="cumulus",
            cumulus_ssh_key="/tmp/key",
        ),
        runner=fake_runner,
    )

    verification = payloads["fabric_verification"]["families"]
    assert verification["nvlink"]["completeness"] == "full_stack_verified"
    assert verification["infiniband"]["completeness"] == "full_stack_verified"
    assert verification["spectrum-x"]["completeness"] == "full_stack_verified"
    assert payloads["fabric_scorecard"]["status"] == "ok"
    assert payloads["fabric_scorecard"]["summary"]["full_stack_verified_families"] == 3
    ib_summary = verification["infiniband"]["scenario_summary"]
    assert ib_summary["capacity_and_path_visibility_ready"] is True
    assert ib_summary["routing_and_counter_verification_ready"] is True
    assert ib_summary["visible_hca_count"] == 1
    assert ib_summary["visible_host_count"] == 2
    assert ib_summary["visible_switch_count"] == 1
    assert ib_summary["runtime_correlation_ready"] is True
    assert ib_summary["multi_to_single_nccl_ratio"] > 0
    spectrum_summary = verification["spectrum-x"]["scenario_summary"]
    assert spectrum_summary["fabric_readiness_ready"] is True
    assert spectrum_summary["switch_count_targeted"] == 1
    assert spectrum_summary["adaptive_routing_visible"] is True
    assert spectrum_summary["roce_qos_visible"] is True
    assert spectrum_summary["bgp_route_visibility"] is True
    assert spectrum_summary["runtime_correlation_ready"] is True
    nmx = verification["nvlink"]["control_plane"]["nmx"]
    assert nmx["topology"]["compute_node_count"] == 4
    assert nmx["topology"]["gpu_count"] == 8
    assert nmx["topology"]["switch_asic_count"] == 4
    assert nmx["topology"]["switch_tray_count"] == 2
    assert nmx["topology"]["chassis_count"] == 1
    assert nmx["topology"]["chassis_serial_numbers"] == ["SN12345678901"]
    assert nmx["topology"]["ports"]["gpu_facing_ports"] == 32
    assert nmx["topology"]["ports"]["switch_facing_ports"] == 32
    assert nmx["topology"]["ports"]["expected_total_ports"] == 64
    assert nmx["topology"]["ports"]["matches_expected_formula"] is True
    assert nmx["topology"]["sample_gpu"]["device_id"] == 1
    assert nmx["topology"]["sample_gpu"]["port_count"] == 2
    assert nmx["topology"]["sample_compute_node"]["gpu_id_count"] == 2
    assert nmx["topology"]["sample_switch"]["device_id"] == 1
    assert nmx["topology"]["sample_chassis"]["name"] == "GB200-Rack-1"
    assert nmx["topology"]["scenario_answers"]["can_support_two_concurrent_4gpu_workloads"] is True
    assert nmx["topology"]["scenario_answers"]["team_alpha_candidate"]["selected_gpu_count"] == 4
    assert nmx["topology"]["scenario_answers"]["team_beta_candidate"]["selected_gpu_count"] == 4
    assert nmx["topology"]["scenario_answers"]["node_gpu_grouping_field"] == "GpuIDList"
    assert nmx["topology"]["scenario_answers"]["switch_asic_distinguishing_field"] == "DeviceID"
    assert nmx["topology"]["scenario_answers"]["switch_tray_grouping_fields"] == [
        "LocationInfo.ChassisID",
        "LocationInfo.SlotID",
        "LocationInfo.HostID",
        "LocationInfo.TrayIndex",
    ]
    assert nmx["topology"]["scenario_answers"]["sample_switch_tray"]["switch_asic_count"] == 2
    assert nmx["partitions"]["partition_count"] == 1
    assert nmx["partitions"]["unassigned_gpu_count"] == 1
    assert nmx["partitions"]["unassigned_gpu_locations"] == ["1.4.1.2"]
    assert nmx["partitions"]["scenario_answers"]["ready_for_new_partition_create"] is True
    assert nmx["partitions"]["scenario_answers"]["default_partition_present"] is True
    assert nmx["partitions"]["scenario_answers"]["default_partition_member_count"] == 7
    assert nmx["telemetry"]["switch_temperature_series"] == 2
    assert nmx["telemetry"]["tx_throughput_series"] == 1
    assert nmx["telemetry"]["rx_throughput_series"] == 1
    assert nmx["telemetry"]["physical_error_series"] == 1


def test_build_nmx_partition_lab_payload_returns_lab_only_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain_uuid = "domain-1"
    compute_nodes = [
        {"ID": "node-1", "Name": "node-1", "LocationInfo": {"ChassisID": 1, "SlotID": 1, "HostID": 1, "TrayIndex": 0}, "GpuIDList": ["gpu-1-1", "gpu-1-2"]},
        {"ID": "node-2", "Name": "node-2", "LocationInfo": {"ChassisID": 1, "SlotID": 2, "HostID": 1, "TrayIndex": 1}, "GpuIDList": ["gpu-2-1", "gpu-2-2"]},
        {"ID": "node-3", "Name": "node-3", "LocationInfo": {"ChassisID": 1, "SlotID": 3, "HostID": 1, "TrayIndex": 2}, "GpuIDList": ["gpu-3-1", "gpu-3-2"]},
        {"ID": "node-4", "Name": "node-4", "LocationInfo": {"ChassisID": 1, "SlotID": 4, "HostID": 1, "TrayIndex": 3}, "GpuIDList": ["gpu-4-1", "gpu-4-2"]},
    ]
    gpus = []
    for node_index in range(1, 5):
        for gpu_index in range(1, 3):
            gpus.append(
                {
                    "ID": f"gpu-{node_index}-{gpu_index}",
                    "SystemUID": f"sys-{node_index}",
                    "DeviceID": gpu_index,
                    "DeviceUID": f"device-{node_index}-{gpu_index}",
                    "DomainUUID": domain_uuid,
                    "PartitionID": None,
                    "PartitionName": "Unassigned",
                    "Health": "HEALTHY",
                    "LocationInfo": {"ChassisID": 1, "SlotID": node_index, "HostID": 1, "TrayIndex": node_index - 1},
                    "PortIDList": [f"port-{node_index}-{gpu_index}-1", f"port-{node_index}-{gpu_index}-2"],
                }
            )
    switches = [
        {"ID": "sw-1a", "DeviceID": 1, "DomainUUID": domain_uuid, "Health": "HEALTHY", "LocationInfo": {"ChassisID": 1, "SlotID": 10, "HostID": 1, "TrayIndex": 20}, "PortIDList": ["p1", "p2"]},
    ]
    switch_nodes = [
        {"ID": "switch-node-1", "DomainUUID": domain_uuid, "SwitchIDList": ["sw-1a"], "LocationInfo": {"ChassisID": 1, "SlotID": 10, "HostID": 1, "TrayIndex": 20}},
    ]
    chassis = [
        {"ID": "chassis-1", "Name": "GB200-Rack-1", "DomainUUID": domain_uuid, "Health": "HEALTHY", "LocationInfo": {"ChassisID": 1, "ChassisSerialNumber": "SN12345678901"}},
    ]
    partitions = [
        {"ID": "partition-default", "PartitionID": 32766, "Name": "Default", "Type": "NVLINK", "DomainUUID": domain_uuid, "Members": {"locations": []}},
    ]

    def fake_http_json(url: str, token: str | None = None, timeout: int = 10) -> dict[str, object]:
        del token, timeout
        mapping = {
            "services": [{"ID": "service-1", "DomainUUID": domain_uuid}],
            "compute-nodes": compute_nodes,
            "gpus": gpus,
            "switches": switches,
            "switch-nodes": switch_nodes,
            "chassis": chassis,
            "ports": [],
            "partitions": partitions,
        }
        for suffix, payload in mapping.items():
            if url.endswith(suffix):
                return {"status": "ok", "http_status": 200, "json": payload, "error": ""}
        return {"status": "error", "http_status": 404, "json": [], "error": f"unexpected url {url}"}

    def fake_http_text(url: str, token: str | None = None, timeout: int = 10) -> dict[str, object]:
        del token, timeout
        if url.endswith("/metrics"):
            return {"status": "ok", "http_status": 200, "text": "", "error": ""}
        return {"status": "error", "http_status": 404, "text": "", "error": f"unexpected url {url}"}

    monkeypatch.setattr("cluster.fabric.evaluator._http_json", fake_http_json)
    monkeypatch.setattr("cluster.fabric.evaluator._http_text", fake_http_text)

    payload = build_nmx_partition_lab_payload(
        nmx_url="https://nmx.example",
        alpha_name="AlphaPartition",
        beta_name="BetaPartition",
        alpha_size=4,
        beta_size=4,
    )

    assert payload["lab_only"] is True
    assert payload["completeness"] == "lab_plan_ready"
    assert payload["topology"]["chassis_count"] == 1
    assert payload["topology"]["gpu_count"] == 8
    assert payload["topology"]["node_gpu_grouping_field"] == "GpuIDList"
    assert payload["topology"]["switch_asic_distinguishing_field"] == "DeviceID"
    assert payload["topology"]["team_alpha_candidate"]["selected_gpu_count"] == 4
    assert payload["partitions"]["ready_for_create"] is True
    assert payload["partitions"]["alpha_seed_locations"] == ["1.1.1.1", "1.1.1.2", "1.2.1.1", "1.2.1.2"]
    assert payload["partitions"]["beta_seed_locations"] == ["1.3.1.1", "1.3.1.2", "1.4.1.1", "1.4.1.2"]
    assert payload["commands"]["create_alpha"].startswith("curl -k -X POST https://nmx.example/nmx/v1/partitions")
    assert '"Name":"AlphaPartition"' in payload["commands"]["create_alpha"]
    assert payload["commands"]["update_alpha_after_borrow"].startswith("curl -k -X PUT https://nmx.example/nmx/v1/partitions/")
    assert payload["commands"]["delete_beta"].startswith("curl -k -X DELETE https://nmx.example/nmx/v1/partitions/")


def test_build_fabric_payloads_prefers_compute_node_gpu_id_list_over_system_uid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "2026-03-16_fabric_gpuidlist"
    label = "node1"
    run_dir = tmp_path / "runs" / run_id
    structured = run_dir / "structured"
    structured.mkdir(parents=True, exist_ok=True)

    _seed_meta(
        structured,
        run_id,
        label,
        topo="GPU0\t X \t NV4\tGPU1",
        ibstat="CA 'mlx5_0'\n\tState: Active\n\tLink layer: Ethernet\n",
        rdma_link="",
        ibv_devinfo="",
        ethtool="Settings for eth0:\n\tSpeed: 25000Mb/s\n",
    )
    _write_json(structured / f"{run_id}_{label}_meta_nvlink_topology.json", {"summary": {"gpu_count": 2, "nvlink_pair_count": 1}})
    _write_json(structured / f"{run_id}_node1_nccl.json", {"results": [{"algbw_gbps": 2200.0, "busbw_gbps": 0.0}]})
    _write_vllm_csv(structured / f"{run_id}_{label}_vllm_serve_sweep.csv")

    def fake_http_json(url: str, token: str | None = None, timeout: int = 10) -> dict[str, object]:
        del token, timeout
        mapping = {
            "services": [{"ID": "svc-1", "DomainUUID": "domain-1"}],
            "compute-nodes": [
                {"ID": "node-a", "Name": "node-a", "SystemUID": "sys-a", "GpuIDList": ["gpu-2", "gpu-1"], "LocationInfo": {"ChassisID": 1, "SlotID": 1, "HostID": 1, "TrayIndex": 0}},
                {"ID": "node-b", "Name": "node-b", "SystemUID": "sys-b", "GpuIDList": ["gpu-4", "gpu-3"], "LocationInfo": {"ChassisID": 1, "SlotID": 2, "HostID": 1, "TrayIndex": 1}},
            ],
            "gpus": [
                {"ID": "gpu-1", "SystemUID": "wrong-sys-1", "DeviceID": 1, "DomainUUID": "domain-1", "PartitionID": 32766, "PartitionName": "default", "LocationInfo": {"ChassisID": 1, "SlotID": 1, "HostID": 1, "TrayIndex": 0}},
                {"ID": "gpu-2", "SystemUID": "wrong-sys-2", "DeviceID": 2, "DomainUUID": "domain-1", "PartitionID": 32766, "PartitionName": "default", "LocationInfo": {"ChassisID": 1, "SlotID": 1, "HostID": 1, "TrayIndex": 0}},
                {"ID": "gpu-3", "SystemUID": "wrong-sys-3", "DeviceID": 1, "DomainUUID": "domain-1", "PartitionID": 32766, "PartitionName": "default", "LocationInfo": {"ChassisID": 1, "SlotID": 2, "HostID": 1, "TrayIndex": 1}},
                {"ID": "gpu-4", "SystemUID": "wrong-sys-4", "DeviceID": 2, "DomainUUID": "domain-1", "PartitionID": 32766, "PartitionName": "default", "LocationInfo": {"ChassisID": 1, "SlotID": 2, "HostID": 1, "TrayIndex": 1}},
            ],
            "switches": [],
            "switch-nodes": [],
            "chassis": [],
            "ports": [],
            "partitions": [],
        }
        for suffix, payload in mapping.items():
            if url.endswith(suffix):
                return {"status": "ok", "http_status": 200, "json": payload, "error": ""}
        return {"status": "error", "http_status": 404, "json": [], "error": f"unexpected url {url}"}

    def fake_http_text(url: str, token: str | None = None, timeout: int = 10) -> dict[str, object]:
        del token, timeout
        if url.endswith("/metrics"):
            return {"status": "ok", "http_status": 200, "text": "", "error": ""}
        return {"status": "error", "http_status": 404, "text": "", "error": f"unexpected url {url}"}

    monkeypatch.setattr("cluster.fabric.evaluator._http_json", fake_http_json)
    monkeypatch.setattr("cluster.fabric.evaluator._http_text", fake_http_text)

    payloads = build_fabric_payloads(
        run_id=run_id,
        run_dir=run_dir,
        primary_label=label,
        management=ManagementConfig(
            ib_mgmt_host=None,
            ib_mgmt_user=None,
            ib_mgmt_ssh_key=None,
            nmx_url="https://nmx.example",
            nmx_token=None,
            cumulus_hosts=(),
            cumulus_user=None,
            cumulus_ssh_key=None,
        ),
    )

    nmx = payloads["fabric_verification"]["families"]["nvlink"]["control_plane"]["nmx"]
    assert nmx["topology"]["scenario_answers"]["node_gpu_grouping_field"] == "GpuIDList"
    assert nmx["topology"]["compute_nodes"][0]["gpu_locations"] == ["1.1.1.2", "1.1.1.1"]
    assert nmx["topology"]["compute_nodes"][1]["gpu_locations"] == ["1.2.1.2", "1.2.1.1"]
    assert nmx["topology"]["scenario_answers"]["can_support_two_concurrent_4gpu_workloads"] is False
