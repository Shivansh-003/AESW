"""
Tests for AESW FastAPI Application and Simulation Endpoints
===========================================================
Validates API contracts, session management, graph streaming, step execution,
metrics retrieval, and error states using FastAPI TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from aesw.api.app import create_app
from aesw.api.simulation_manager import SimulationManager


@pytest.fixture
def client():
    manager = SimulationManager()
    app = create_app(manager=manager)
    return TestClient(app)


def test_api_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_get_algorithms(client):
    response = client.get("/api/algorithms")
    assert response.status_code == 200
    algos = response.json()
    assert len(algos) >= 7
    ids = [a["id"] for a in algos]
    assert "aesw" in ids
    assert "random_walk" in ids
    assert "ant_colony" in ids


def test_get_graph_types(client):
    response = client.get("/api/graph-types")
    assert response.status_code == 200
    types = response.json()
    ids = [t["id"] for t in types]
    assert "er" in ids
    assert "ba" in ids
    assert "ws" in ids
    assert "grid" in ids
    assert "rgg" in ids


def test_create_and_step_aesw_simulation(client):
    payload = {
        "algorithm": "aesw",
        "graph_type": "er",
        "num_nodes": 15,
        "graph_parameters": {"p": 0.3},
        "walker_count": 2,
        "seed": 42,
        "time_horizon": 50,
        "communication_mode": "PUSH_PULL",
    }
    create_res = client.post("/api/simulations", json=payload)
    assert create_res.status_code == 201
    sim_data = create_res.json()
    sim_id = sim_data["simulation_id"]
    assert sim_data["algorithm"] == "aesw"
    assert sim_data["is_aesw"] is True
    assert sim_data["time"] == 0

    # Test graph endpoint
    graph_res = client.get(f"/api/simulations/{sim_id}/graph")
    assert graph_res.status_code == 200
    g_data = graph_res.json()
    assert len(g_data["nodes"]) == 15
    assert g_data["total_edge_count"] > 0
    assert g_data["active_edge_count"] <= g_data["total_edge_count"]

    # Test initial metrics
    metrics_res = client.get(f"/api/simulations/{sim_id}/metrics")
    assert metrics_res.status_code == 200
    m_data = metrics_res.json()
    assert m_data["search_time"] == 0
    assert m_data["total_moves"] == 0

    # Advance 1 step
    step_res = client.post(f"/api/simulations/{sim_id}/step")
    assert step_res.status_code == 200
    step_data = step_res.json()
    assert step_data["time"] >= 1
    assert len(step_data["walkers"]) == 2

    # Check decisions
    dec_res = client.get(f"/api/simulations/{sim_id}/decisions")
    assert dec_res.status_code == 200
    decs = dec_res.json()
    assert len(decs) == 2
    for d in decs:
        assert d["mode"] in ["LOCAL", "LONG_JUMP"]
        assert d["estimated_lambda"] >= 0.0

    # Check evidence
    ev_res = client.get(f"/api/simulations/{sim_id}/evidence")
    assert ev_res.status_code == 200
    evs = ev_res.json()
    assert isinstance(evs, list)
    assert len(evs) >= 1  # Walkers store observations as evidence

    # Check churn
    churn_res = client.get(f"/api/simulations/{sim_id}/churn")
    assert churn_res.status_code == 200
    churns = churn_res.json()
    assert len(churns) == 2
    assert churns[0]["walker_id"] == 0

    # Check trajectory
    traj_res = client.get(f"/api/simulations/{sim_id}/trajectory")
    assert traj_res.status_code == 200
    traj_data = traj_res.json()
    assert len(traj_data["target_trajectory"]) >= 1
    assert "0" in traj_data["walker_trajectories"] or 0 in traj_data["walker_trajectories"]

    # Test reset
    reset_res = client.post(f"/api/simulations/{sim_id}/reset")
    assert reset_res.status_code == 200
    reset_data = reset_res.json()
    assert reset_data["time"] == 0
    assert reset_data["metrics"]["search_time"] == 0


def test_create_and_step_random_walk_baseline(client):
    payload = {
        "algorithm": "random_walk",
        "graph_type": "ba",
        "num_nodes": 20,
        "graph_parameters": {"m": 2},
        "walker_count": 2,
        "seed": 99,
    }
    create_res = client.post("/api/simulations", json=payload)
    assert create_res.status_code == 201
    sim_data = create_res.json()
    sim_id = sim_data["simulation_id"]
    assert sim_data["is_aesw"] is False

    # Advance 1 step
    step_res = client.post(f"/api/simulations/{sim_id}/step")
    assert step_res.status_code == 200
    step_data = step_res.json()
    assert step_data["time"] >= 0
    assert step_data["metrics"]["total_moves"] >= 1

    # Evidence and churn should be empty for baseline
    ev_res = client.get(f"/api/simulations/{sim_id}/evidence")
    assert ev_res.json() == []

    churn_res = client.get(f"/api/simulations/{sim_id}/churn")
    assert churn_res.json() == []


def test_simulation_not_found(client):
    res = client.get("/api/simulations/nonexistent_id")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_simulation_delete(client):
    payload = {"algorithm": "degree", "num_nodes": 10, "walker_count": 1}
    create_res = client.post("/api/simulations", json=payload)
    sim_id = create_res.json()["simulation_id"]

    del_res = client.delete(f"/api/simulations/{sim_id}")
    assert del_res.status_code == 200

    get_res = client.get(f"/api/simulations/{sim_id}")
    assert get_res.status_code == 404


def test_simulation_status_lifecycle_regression(client):
    """Regression test ensuring intermediate status is never prematurely BUDGET_EXHAUSTED.

    Tests:
    1. Initialized simulation (t = 0) reports status == 'INITIALIZED'
    2. Repeated requests without advancing remain 'INITIALIZED'
    3. Running simulation before budget reports status == 'RUNNING'
    4. Genuine budget exhaustion reports status == 'BUDGET_EXHAUSTED'
    5. Successful target acquisition reports status == 'SUCCESS' and success == True
    """
    # 1 & 2: Initialized simulation and repeated requests
    payload = {
        "algorithm": "random_walk",
        "graph_type": "er",
        "num_nodes": 15,
        "walker_count": 1,
        "seed": 999,
        "time_horizon": 5,
    }
    create_res = client.post("/api/simulations", json=payload)
    assert create_res.status_code == 201
    sim_data = create_res.json()
    sim_id = sim_data["simulation_id"]

    assert sim_data["status"] == "INITIALIZED"
    assert sim_data["metrics"]["status"] == "INITIALIZED"
    assert sim_data["termination_status"] is None

    # Repeated metrics request without stepping
    for _ in range(3):
        m_res = client.get(f"/api/simulations/{sim_id}/metrics")
        assert m_res.status_code == 200
        assert m_res.json()["status"] == "INITIALIZED"
        assert m_res.json()["success"] is False

    # 3: Running simulation before budget reached
    step1 = client.post(f"/api/simulations/{sim_id}/step")
    assert step1.status_code == 200
    s1_data = step1.json()
    if not s1_data["is_terminated"]:
        assert s1_data["metrics"]["status"] == "RUNNING"
        # Verify repeated query while running remains RUNNING
        m_running = client.get(f"/api/simulations/{sim_id}/metrics").json()
        assert m_running["status"] == "RUNNING"

    # Step until genuine termination
    final_step = s1_data
    while not final_step["is_terminated"]:
        res = client.post(f"/api/simulations/{sim_id}/step")
        assert res.status_code == 200
        final_step = res.json()

    assert final_step["is_terminated"] is True
    assert final_step["termination_status"] in ["SUCCESS", "BUDGET_EXHAUSTED", "DISCONNECTED"]
    assert final_step["metrics"]["status"] == final_step["termination_status"]

    # Verify persistent terminal status on repeated metrics requests
    m_term = client.get(f"/api/simulations/{sim_id}/metrics").json()
    assert m_term["status"] == final_step["termination_status"]

    # 4: Explicit test for genuine budget exhaustion
    # Use seed where walkers stay or target is far away with short horizon
    budget_payload = {
        "algorithm": "degree",
        "graph_type": "er",
        "num_nodes": 40,
        "walker_count": 1,
        "seed": 12345,
        "time_horizon": 5,
    }
    b_sim = client.post("/api/simulations", json=budget_payload).json()
    b_id = b_sim["simulation_id"]
    for _ in range(5):
        st = client.post(f"/api/simulations/{b_id}/step").json()
        if st["is_terminated"]:
            break

    b_final = client.get(f"/api/simulations/{b_id}").json()
    if b_final["time"] >= 5 and not b_final["metrics"]["success"]:
        assert b_final["termination_status"] == "BUDGET_EXHAUSTED"
        assert b_final["metrics"]["status"] == "BUDGET_EXHAUSTED"

    # 5: Explicit test for successful target acquisition (seed 42 ER with 3 walkers captures at step 3)
    success_payload = {
        "algorithm": "aesw",
        "graph_type": "er",
        "num_nodes": 30,
        "walker_count": 3,
        "seed": 42,
        "time_horizon": 100,
        "graph_parameters": {"p": 0.18},
        "p_move": 0.25,
    }
    succ_sim = client.post("/api/simulations", json=success_payload).json()
    succ_id = succ_sim["simulation_id"]
    for _ in range(3):
        succ_res = client.post(f"/api/simulations/{succ_id}/step").json()
    assert succ_res["is_terminated"] is True
    assert succ_res["termination_status"] == "SUCCESS"
    assert succ_res["metrics"]["status"] == "SUCCESS"
    assert succ_res["metrics"]["success"] is True

