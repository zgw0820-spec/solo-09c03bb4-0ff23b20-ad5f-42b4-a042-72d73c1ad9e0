from __future__ import annotations

import concurrent.futures
import copy
import json
import sqlite3
import sys
from pathlib import Path

from fastapi.testclient import TestClient

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from app.main import create_app  # noqa: E402


def _client(db: Path) -> TestClient:
    return TestClient(create_app(db))


def _topology() -> dict:
    return {
        "nodes": [
            {"id": "n1", "name": "one", "cpu_capacity": 8},
            {"id": "n2", "name": "two", "cpu_capacity": 8},
            {"id": "n3", "name": "three", "cpu_capacity": 8},
        ],
        "containers": [
            {
                "id": "c1",
                "service": "svc",
                "min_ready": 1,
                "node_id": "n1",
                "cpu_request": 2,
            },
            {
                "id": "c2",
                "service": "svc",
                "min_ready": 1,
                "node_id": "n2",
                "cpu_request": 3,
            },
        ],
    }


def _drain_with_pending(client: TestClient) -> tuple[dict, int]:
    """n1 draining, c1's replica starting on n3, plus a failed history row."""
    client.post("/api/topology", json=_topology())
    state = client.post("/api/nodes/n1/drain").json()
    migration = state["open_drains"][0]["pending_decisions"][0]
    return state, migration["id"]


def _backup(client: TestClient) -> dict:
    resp = client.get("/api/backup")
    assert resp.status_code == 200, resp.text
    assert "attachment" in resp.headers["content-disposition"]
    return resp.json()


def _state(client: TestClient) -> dict:
    return client.get("/api/state").json()


def test_backup_contains_full_state_including_reservations(tmp_path):
    with _client(tmp_path / "b.db") as client:
        _, migration_id = _drain_with_pending(client)

    with _client(tmp_path / "b.db") as client:
        payload = _backup(client)
        assert payload["format"] == "node-drain-backup"
        assert payload["version"] == 1
        state = payload["state"]
        assert {node["id"] for node in state["nodes"]} == {"n1", "n2", "n3"}
        assert {service["name"] for service in state["services"]} == {"svc"}
        assert len(state["containers"]) == 2
        assert len(state["drains"]) == 1
        # The unresolved replica's CPU reservation is in the file.
        starting = [m for m in state["migrations"] if m["state"] == "starting"]
        assert len(starting) == 1
        assert starting[0]["id"] == migration_id
        assert starting[0]["reserved_cpu"] == 2
        assert starting[0]["target_node_id"] is not None
        # Data and revision describe one committed state.
        assert isinstance(payload["revision"], int)


def test_backup_export_is_a_single_committed_snapshot(tmp_path):
    """Export matches a committed state, never an uncommitted in-flight write.

    A resolve is modelled as an open ``BEGIN IMMEDIATE`` transaction that has
    written the cutover but not committed. While it is open, the backup must be
    the pre-cutover state; after the commit it must be the post-cutover state,
    with no mixture.
    """
    db_path = tmp_path / "race.db"
    app = create_app(db_path)
    with TestClient(app) as client:
        state, migration_id = _drain_with_pending(client)
        database = app.state.db
        before = _backup(client)
        pre_revision = before["revision"]

        migration = next(
            m for m in before["state"]["migrations"] if m["state"] == "starting"
        )
        assert migration["id"] == migration_id

        # Write the cutover but hold the transaction open.
        with database.write() as conn:
            conn.execute(
                "UPDATE containers SET node_id = ? WHERE id = 'c1'",
                (migration["target_node_id"],),
            )
            conn.execute(
                "UPDATE migrations SET state = 'completed' WHERE id = ?",
                (migration_id,),
            )
            conn.execute(
                "UPDATE drains SET status = 'completed', finished_at = ? "
                "WHERE id = ?",
                ("2026-01-01T00:00:00+00:00", migration["drain_id"]),
            )
            conn.execute("UPDATE nodes SET status = 'drained' WHERE id = 'n1'")
            conn.execute("UPDATE meta SET value = value + 1 WHERE key = 'revision'")

            # Uncommitted changes are invisible to the export's snapshot.
            during = _backup(client)
            assert during["revision"] == pre_revision
            assert during["state"] == before["state"]
            assert next(
                c for c in during["state"]["containers"] if c["id"] == "c1"
            )["node_id"] == "n1"

        after = _backup(client)
        assert after["revision"] == pre_revision + 1
        assert next(
            c for c in after["state"]["containers"] if c["id"] == "c1"
        )["node_id"] == migration["target_node_id"]
        assert all(
            m["state"] != "starting" for m in after["state"]["migrations"]
        )
        assert after["state"]["drains"][0]["status"] == "completed"

        # Both exports are internally consistent on their own: the reservation
        # and placement agree in each.
        for snapshot in (during, after):
            containers = {c["id"]: c for c in snapshot["state"]["containers"]}
            nodes = {n["id"]: n for n in snapshot["state"]["nodes"]}
            for node in nodes.values():
                serving = sum(
                    c["cpu_request"]
                    for c in containers.values()
                    if c["node_id"] == node["id"]
                )
                reserved = sum(
                    m["reserved_cpu"]
                    for m in snapshot["state"]["migrations"]
                    if m["state"] == "starting" and m["target_node_id"] == node["id"]
                )
                assert serving + reserved <= node["cpu_capacity"]


def test_restore_round_trip_keeps_pending_decision_and_history(tmp_path):
    source_db = tmp_path / "source.db"
    dest_db = tmp_path / "dest.db"

    with _client(source_db) as client:
        state, migration_id = _drain_with_pending(client)
        # Add a failed attempt so history has more than one row.
        client.post(
            f"/api/migrations/{migration_id}/resolve",
            json={"outcome": "failed", "reason": "first try"},
        )
        client.post("/api/nodes/n1/resume")
        state = _state(client)
        pending = state["open_drains"][0]["pending_decisions"][0]
        pending_id = pending["id"]
        pending_target = pending["target_node_id"]
        backup = _backup(client)

    with _client(dest_db) as dest:
        # Give the destination a different, unrelated rehearsal first.
        dest.post(
            "/api/topology",
            json={
                "nodes": [{"id": "other", "cpu_capacity": 4}],
                "containers": [
                    {
                        "id": "x",
                        "service": "x",
                        "min_ready": 0,
                        "node_id": "other",
                        "cpu_request": 1,
                    }
                ],
            },
        )
        resp = dest.post("/api/backup/restore", json=backup)
        assert resp.status_code == 200, resp.text
        restored = resp.json()

        assert {n["id"] for n in restored["nodes"]} == {"n1", "n2", "n3"}
        assert restored["open_drains"][0]["id"] == backup["state"]["drains"][0]["id"]
        assert restored["open_drains"][0]["pending_decisions"][0]["id"] == pending_id
        assert (
            restored["open_drains"][0]["pending_decisions"][0]["target_node_id"]
            == pending_target
        )
        target = next(n for n in restored["nodes"] if n["id"] == pending_target)
        assert target["reserved_cpu"] == 2
        assert target["remaining_cpu"] == target["cpu_capacity"] - target["serving_cpu"] - 2
        assert len(restored["drains"][0]["attempts"]) == 2
        assert target["can_drain"] is False  # inbound reservation

        # The restored pending migration is still decidable and accounting
        # stays consistent through the cutover.
        resolved = dest.post(
            f"/api/migrations/{pending_id}/resolve", json={"outcome": "ready"}
        )
        assert resolved.status_code == 200, resolved.text
        body = resolved.json()
        n1 = next(n for n in body["nodes"] if n["id"] == "n1")
        assert n1["is_empty"] is True
        assert body["open_drains"] == []
        assert body["drains"][0]["status"] == "completed"
        assert body["drains"][0]["moved"] == 1

    # Restore again into a third instance and confirm state matches what the
    # source held at export time.
    with _client(tmp_path / "third.db") as third:
        resp = third.post("/api/backup/restore", json=backup)
        assert resp.status_code == 200, resp.text
        state = resp.json()
        assert state["unfinished_drain_count"] == 0
        assert state["active_drain_count"] == 1


def test_restore_does_not_reuse_ids_of_stale_tasks(tmp_path):
    """Ids in a restored file must not be reissued to *new* work.

    Otherwise a page opened before the restore could resolve a task that did
    not exist when its id was shown.
    """
    db = tmp_path / "ids.db"
    with _client(db) as client:
        _, old_migration_id = _drain_with_pending(client)
        backup = _backup(client)

    with _client(db) as client:
        # Take the old rehearsal to completion so the next drain ids differ.
        client.post(
            f"/api/migrations/{old_migration_id}/resolve", json={"outcome": "ready"}
        )
        client.post("/api/nodes/n1/restore")
        client.post("/api/nodes/n2/drain")
        state = _state(client)
        new_drain_ids = {drain["id"] for drain in state["drains"]}
        assert max(new_drain_ids) > backup["state"]["drains"][0]["id"]

        resp = client.post("/api/backup/restore", json=backup)
        assert resp.status_code == 200, resp.text

        # A new drain after the restore must not collide with restored ids.
        client.post("/api/nodes/n2/restore")
        client.post("/api/nodes/n1/restore")
        fresh = client.post("/api/nodes/n2/drain")
        assert fresh.status_code == 200, fresh.text
        body = fresh.json()
        restored_ids = {drain["id"] for drain in backup["state"]["drains"]}
        new_ids = {drain["id"] for drain in body["drains"]} - restored_ids
        assert new_ids
        assert all(new_id > max(restored_ids) for new_id in new_ids)


def test_stale_page_cannot_resolve_after_restore(tmp_path):
    db = tmp_path / "stale.db"
    with _client(db) as client:
        state, first_migration = _drain_with_pending(client)
        generation = state["generation"]
        backup = _backup(client)

        # Replace the dataset: import bumps the generation.
        client.post("/api/topology", json=_topology())
        client.post("/api/nodes/n1/drain")
        state = _state(client)
        assert state["generation"] != generation

        stale = client.post(
            f"/api/migrations/{first_migration}/resolve",
            json={"outcome": "ready"},
            params={"generation": generation},
        )
        assert stale.status_code == 409
        assert stale.json()["error"]["code"] == "STALE_STATE"
        # Nothing was resolved: the new pending task is untouched.
        assert _state(client)["open_drains"][0]["status"] == "active"

        # Same protection after a restore from a backup taken earlier.
        resp = client.post("/api/backup/restore", json=backup)
        assert resp.status_code == 200
        after_restore = resp.json()
        assert after_restore["generation"] != state["generation"]
        stale = client.post(
            f"/api/migrations/{first_migration}/resolve",
            json={"outcome": "failed"},
            params={"generation": generation},
        )
        assert stale.status_code == 409
        assert _state(client)["open_drains"][0]["status"] == "active"

        # A page that re-rendered after the restore can still decide.
        ok = client.post(
            f"/api/migrations/{first_migration}/resolve",
            json={"outcome": "ready"},
            params={"generation": after_restore["generation"]},
        )
        assert ok.status_code == 200, ok.text


def _tamper(backup: dict, mutate) -> dict:
    clone = copy.deepcopy(backup)
    mutate(clone["state"])
    return clone


def test_invalid_backup_leaves_state_unchanged(tmp_path):
    db = tmp_path / "invalid.db"
    with _client(db) as client:
        _, migration_id = _drain_with_pending(client)
        backup = _backup(client)
        before = _state(client)

        def drop_container(state: dict) -> None:
            state["containers"] = [c for c in state["containers"] if c["id"] != "c1"]

        def dangling_migration(state: dict) -> None:
            state["migrations"][0]["container_id"] = "nope"

        def dangling_drain(state: dict) -> None:
            state["drains"][0]["node_id"] = "nope"

        def broken_reference_target(state: dict) -> None:
            for migration in state["migrations"]:
                if migration["state"] == "starting":
                    migration["target_node_id"] = "nope"

        def over_capacity(state: dict) -> None:
            state["nodes"][0]["cpu_capacity"] = 1

        def reservation_over_capacity(state: dict) -> None:
            state["nodes"][0]["cpu_capacity"] = 2
            for migration in state["migrations"]:
                if migration["state"] == "starting":
                    migration["reserved_cpu"] = 2
                    migration["target_node_id"] = state["nodes"][0]["id"]

        def invalid_status(state: dict) -> None:
            state["nodes"][0]["status"] = "exploded"

        candidates = [
            ({"format": "wrong"}, "format"),
            ({"format": "node-drain-backup", "version": 99}, "version"),
            ({"not": "a backup"}, "shape"),
            (None, "null"),
        ]
        for payload, label in candidates:
            resp = client.post("/api/backup/restore", json=payload)
            assert resp.status_code in (409, 422), (label, resp.text)
            assert _state(client) == before, label

        for mutate, label in [
            (drop_container, "drop_container"),
            (dangling_migration, "dangling_migration"),
            (dangling_drain, "dangling_drain"),
            (broken_reference_target, "broken_target"),
            (over_capacity, "over_capacity"),
            (reservation_over_capacity, "reservation_over_capacity"),
            (invalid_status, "invalid_status"),
        ]:
            resp = client.post("/api/backup/restore", json=_tamper(backup, mutate))
            assert resp.status_code == 422, (label, resp.text)
            assert resp.json()["error"]["code"] == "INVALID_BACKUP", label
            assert _state(client) == before, label

        # Malformed JSON and empty bodies are rejected too.
        resp = client.post(
            "/api/backup/restore",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422
        assert _state(client) == before
        resp = client.post(
            "/api/backup/restore",
            content=b"",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422
        assert _state(client) == before

        # The valid file still restores, proving the refusals were not caused
        # by an unrelated breakage.
        assert client.post("/api/backup/restore", json=backup).status_code == 200


def test_restore_failure_mid_write_rolls_back(tmp_path, monkeypatch):
    from app import service

    db = tmp_path / "rollback.db"
    with _client(db) as client:
        _, migration_id = _drain_with_pending(client)
        backup = _backup(client)
        before = _state(client)

        original = service._mark_replaced

        def explode(conn):
            raise sqlite3.OperationalError("simulated disk failure")

        monkeypatch.setattr(service, "_mark_replaced", explode)
        resp = client.post("/api/backup/restore", json=backup)
        assert resp.status_code == 500
        monkeypatch.setattr(service, "_mark_replaced", original)

        after = _state(client)
        assert after == before
        # The pending task from before is still resolvable.
        ok = client.post(
            f"/api/migrations/{migration_id}/resolve", json={"outcome": "ready"}
        )
        assert ok.status_code == 200


def test_restore_and_drain_apply_in_commit_order(tmp_path):
    """Concurrent restore/drain serialise on the write lock.

    Either the drain commits first and the restore then wipes it (final state
    equals the backup), or the restore commits first and the drain acts on the
    restored topology (one completed drain, n1 empty). Never a mix.
    """
    db = tmp_path / "concurrent.db"
    with _client(db) as client:
        client.post("/api/topology", json=_topology())
        backup = _backup(client)

    def drain() -> str:
        with _client(db) as other:
            resp = other.post("/api/nodes/n1/drain")
            assert resp.status_code == 200, resp.text
            # Complete it so the outcome is visible in the final state.
            for migration in resp.json()["open_drains"][0]["pending_decisions"]:
                other.post(
                    f"/api/migrations/{migration['id']}/resolve",
                    json={"outcome": "ready"},
                )
            return "drain"

    def restore() -> str:
        with _client(db) as other:
            resp = other.post("/api/backup/restore", json=backup)
            assert resp.status_code == 200, resp.text
            return "restore"

    backup_state = backup["state"]
    backup_drains = {drain["id"] for drain in backup_state["drains"]}

    # ``drain``/``restore`` are shadowed inside the loop below, so keep the
    # callables here.
    workers = [drain, restore]

    for _ in range(5):
        with _client(db) as reset:
            reset.post("/api/backup/restore", json=backup)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda fn: fn(), workers))
        assert sorted(results) == ["drain", "restore"]

        with _client(db) as final:
            state = final.get("/api/state").json()
            for node in state["nodes"]:
                assert node["used_cpu"] == node["serving_cpu"] + node["reserved_cpu"]
                assert node["used_cpu"] <= node["cpu_capacity"]
            assert state["open_drains"] == []
            drain_ids = {drain["id"] for drain in state["drains"]}
            new_drains = drain_ids - backup_drains
            if not new_drains:
                # Restore committed last: byte-for-byte the backup's state.
                assert state["containers"] == [
                    {
                        "id": c["id"],
                        "name": c["name"],
                        "service": c["service"],
                        "cpu_request": c["cpu_request"],
                        "node_id": c["node_id"],
                        "status": c["status"],
                        "migration": None,
                        "blocked_attempt": None,
                    }
                    for c in backup_state["containers"]
                ]
            else:
                # Drain committed last: it ran against the restored topology
                # and completed cleanly.
                assert len(new_drains) == 1
                (drain_id,) = new_drains
                drain_row = next(d for d in state["drains"] if d["id"] == drain_id)
                assert drain_row["status"] == "completed"
                assert drain_row["moved"] == drain_row["total"] == 1
                n1 = next(n for n in state["nodes"] if n["id"] == "n1")
                assert n1["is_empty"] is True
                assert all(n["reserved_cpu"] == 0 for n in state["nodes"])


def test_backup_a_backup_is_stable(tmp_path):
    """Download -> restore -> download yields the same state and revision stays monotonic."""
    with _client(tmp_path / "stable.db") as client:
        _drain_with_pending(client)
        first = _backup(client)
        client.post("/api/backup/restore", json=first)
        second = _backup(client)
    assert first["state"] == second["state"]
    assert second["revision"] > first["revision"]


def test_backup_with_history_and_new_open_drain_round_trips(tmp_path):
    """A node can have an old completed drain plus a newer open one."""
    with _client(tmp_path / "history.db") as client:
        _, migration_id = _drain_with_pending(client)
        client.post(f"/api/migrations/{migration_id}/resolve", json={"outcome": "ready"})
        state = _state(client)
        n1 = next(n for n in state["nodes"] if n["id"] == "n1")
        assert n1["is_empty"] is True
        client.post("/api/nodes/n1/restore")
        # Move a replica back to n1 so it can be drained a second time.
        state = client.post("/api/nodes/n2/drain").json()
        assert state["open_drains"], state
        backup = _backup(client)
        assert len(backup["state"]["drains"]) == 2
        drains_by_status = {d["status"] for d in backup["state"]["drains"]}
        assert drains_by_status == {"completed", "active"}

    with _client(tmp_path / "history2.db") as restored:
        resp = restored.post("/api/backup/restore", json=backup)
        assert resp.status_code == 200, resp.text
        state = resp.json()
        assert state["active_drain_count"] == 1
        assert len(state["drains"]) == 2


def test_backup_endpoints_are_wired_into_ui(tmp_path):
    static = (BASE / "app" / "static" / "app.js").read_text(encoding="utf-8")
    page = (BASE / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert "/api/backup" in static
    assert "/api/backup/restore" in static
    assert "backup-download-btn" in page
    assert "backup-file-input" in page
