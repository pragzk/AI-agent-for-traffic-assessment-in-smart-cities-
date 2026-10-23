"""
tests/test_api.py
Comprehensive pytest tests for the Traffic Risk Assessment API.
Run from the project root:  pytest tests/test_api.py -v
"""
import sys, os, json, time, tempfile, shutil
# Ensure the project root is on the path so app.py can import its siblings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_LEVELS = {"GREEN", "AMBER", "RED"}


def _level_for_score(score: float) -> str:
    if score >= 0.60:
        return "RED"
    if score >= 0.40:
        return "AMBER"
    return "GREEN"


# ---------------------------------------------------------------------------
# 1. GET /api/status
# ---------------------------------------------------------------------------

class TestStatus:
    @pytest.fixture(autouse=True)
    def client(self):
        from app import app
        self.client = TestClient(app)

    def test_returns_200(self):
        r = self.client.get("/api/status")
        assert r.status_code == 200

    def test_required_keys_present(self):
        r = self.client.get("/api/status")
        data = r.json()
        for key in ("hour", "trained", "weights", "threshold"):
            assert key in data, f"Missing key: {key}"

    def test_types(self):
        data = self.client.get("/api/status").json()
        assert isinstance(data["hour"], int)
        assert isinstance(data["trained"], bool)
        assert isinstance(data["weights"], dict)
        assert isinstance(data["threshold"], float)


# ---------------------------------------------------------------------------
# 2. POST /api/reset  +  24x POST /api/step
# ---------------------------------------------------------------------------

class TestResetAndStep:
    @pytest.fixture(autouse=True)
    def setup(self):
        from app import app, S
        self.client = TestClient(app)
        self.S = S
        r = self.client.post("/api/reset")
        assert r.status_code == 200
        assert r.json().get("ok") is True

    def test_24_hours_zones_count_and_hour_progression(self):
        hours_seen = []
        for _ in range(24):
            r = self.client.post("/api/step")
            assert r.status_code == 200
            data = r.json()
            hours_seen.append(data["hour"])
            assert len(data["zones"]) == 10, "Expected exactly 10 zones"
        assert hours_seen == list(range(24)), "Hours should progress 0..23"

    def test_next_hour_wraps_to_0_after_24_steps(self):
        r = None
        for _ in range(24):
            r = self.client.post("/api/step")
        assert r.json()["next_hour"] == 0, "next_hour must wrap to 0 after 24 steps"

    def test_risk_scores_in_range(self):
        for _ in range(24):
            r = self.client.post("/api/step")
            for zone in r.json()["zones"]:
                score = zone["risk_score"]
                assert 0.0 <= score <= 1.0, f"risk_score {score} out of range"

    def test_level_is_valid(self):
        for _ in range(24):
            r = self.client.post("/api/step")
            for zone in r.json()["zones"]:
                assert zone["level"] in VALID_LEVELS, f"Invalid level: {zone['level']}"

    def test_level_matches_score(self):
        for _ in range(24):
            r = self.client.post("/api/step")
            for zone in r.json()["zones"]:
                score = zone["risk_score"]
                expected = _level_for_score(score)
                assert zone["level"] == expected, (
                    f"Zone {zone['zone_id']}: score={score}, "
                    f"expected level={expected}, got {zone['level']}"
                )

    def test_rl_alert_is_null_before_training(self):
        """Before training, rl_alert must be null (None) for all zones."""
        original_trained = self.S.trained
        self.S.trained = False
        try:
            for _ in range(3):
                r = self.client.post("/api/step")
                for zone in r.json()["zones"]:
                    assert zone["rl_alert"] is None, (
                        f"Expected rl_alert=null before training, got {zone['rl_alert']}"
                    )
        finally:
            self.S.trained = original_trained


# ---------------------------------------------------------------------------
# 3. POST /api/train  (background) + GET /api/train/progress
# ---------------------------------------------------------------------------

class TestTrain:
    @pytest.fixture(autouse=True)
    def setup(self):
        from app import app, S, _progress, _progress_lock
        self.client = TestClient(app)
        self.S = S
        self.progress = _progress
        self.progress_lock = _progress_lock
        self.client.post("/api/reset")
        # Wait for any leftover training run to finish
        for _ in range(60):
            p = self.client.get("/api/train/progress").json()
            if not p["running"]:
                break
            time.sleep(0.5)

    def test_train_starts_and_returns_started(self):
        r = self.client.post("/api/train", json={"episodes": 30})
        assert r.status_code == 200
        data = r.json()
        assert data.get("started") is True
        assert data.get("episodes") == 30
        # wait for completion
        for _ in range(120):
            p = self.client.get("/api/train/progress").json()
            if not p["running"]:
                break
            time.sleep(0.5)

    def test_409_on_double_start(self):
        """Second POST while training is running must return 409."""
        # Start a longish run so we can catch the overlap
        self.client.post("/api/train", json={"episodes": 200})
        # Give the thread time to flip running=True
        time.sleep(0.2)
        r2 = self.client.post("/api/train", json={"episodes": 50})
        assert r2.status_code == 409, f"Expected 409, got {r2.status_code}"
        # wait for training to finish before leaving
        for _ in range(120):
            p = self.client.get("/api/train/progress").json()
            if not p["running"]:
                break
            time.sleep(0.5)

    def test_progress_shape(self):
        """GET /api/train/progress must have the five required keys."""
        p = self.client.get("/api/train/progress").json()
        for key in ("running", "episode", "total", "reward_history_so_far", "q_states"):
            assert key in p, f"Missing progress key: {key}"

    def test_progress_updates_during_training(self):
        """Episode counter should advance while training."""
        self.client.post("/api/train", json={"episodes": 100})
        time.sleep(0.3)
        p = self.client.get("/api/train/progress").json()
        # Either still running or already done — either way shape must be correct
        for key in ("running", "episode", "total", "reward_history_so_far", "q_states"):
            assert key in p
        # Wait for finish
        for _ in range(180):
            p = self.client.get("/api/train/progress").json()
            if not p["running"]:
                break
            time.sleep(0.5)
        assert p["episode"] == 100

    def test_reward_history_length_matches_episodes(self):
        """After training, reward_history_so_far length == episodes."""
        eps = 40
        self.client.post("/api/train", json={"episodes": eps})
        for _ in range(120):
            p = self.client.get("/api/train/progress").json()
            if not p["running"]:
                break
            time.sleep(0.5)
        assert len(p["reward_history_so_far"]) == eps

    def test_train_returns_positive_q_states(self):
        self.client.post("/api/train", json={"episodes": 40})
        for _ in range(120):
            p = self.client.get("/api/train/progress").json()
            if not p["running"]:
                break
            time.sleep(0.5)
        assert p["q_states"] > 0, "q_states must be > 0 after training"

    def test_rl_alert_is_0_or_1_after_training(self):
        eps = 40
        self.client.post("/api/train", json={"episodes": eps})
        for _ in range(120):
            p = self.client.get("/api/train/progress").json()
            if not p["running"]:
                break
            time.sleep(0.5)
        self.client.post("/api/reset")
        for _ in range(5):
            r = self.client.post("/api/step")
            for zone in r.json()["zones"]:
                assert zone["rl_alert"] in (0, 1), (
                    f"After training, rl_alert must be 0 or 1, got {zone['rl_alert']}"
                )


# ---------------------------------------------------------------------------
# 4. qtable.json round-trip
# ---------------------------------------------------------------------------

class TestQTablePersistence:
    """Train a tiny model, verify qtable.json is written and can be reloaded."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        import app as app_module
        # Redirect QTABLE_PATH to a temp location so tests don't pollute project dir
        self.tmp_qtable = str(tmp_path / "qtable.json")
        monkeypatch.setattr(app_module, "QTABLE_PATH", self.tmp_qtable)
        self.app_module = app_module
        from app import app
        self.client = TestClient(app)
        # Wait for any ongoing training to stop
        for _ in range(60):
            p = self.client.get("/api/train/progress").json()
            if not p["running"]:
                break
            time.sleep(0.5)

    def _wait_done(self, timeout=180):
        for _ in range(timeout * 2):
            p = self.client.get("/api/train/progress").json()
            if not p["running"]:
                return p
            time.sleep(0.5)
        return self.client.get("/api/train/progress").json()

    def test_qtable_json_written_after_training(self):
        self.client.post("/api/train", json={"episodes": 30})
        self._wait_done()
        assert os.path.exists(self.tmp_qtable), "qtable.json was not created"

    def test_qtable_json_structure(self):
        self.client.post("/api/train", json={"episodes": 30})
        self._wait_done()
        with open(self.tmp_qtable) as f:
            payload = json.load(f)
        for key in ("q_table", "reward_history", "alert_history", "training_summary"):
            assert key in payload, f"Missing key in qtable.json: {key}"
        assert len(payload["reward_history"]) == 30
        assert len(payload["q_table"]) > 0

    def test_qtable_json_round_trip(self):
        """Keys are stored as strings; _load_qtable must reconstruct tuple keys."""
        self.client.post("/api/train", json={"episodes": 30})
        self._wait_done()
        # Reload into a fresh agent
        from rl_agent import RLAgent
        agent2 = RLAgent()
        result = self.app_module._load_qtable(agent2)
        assert result is not None, "_load_qtable returned None"
        assert len(agent2.q_table) > 0, "q_table empty after round-trip"
        # All restored keys must be tuples
        for k in agent2.q_table:
            assert isinstance(k, tuple), f"Restored key is not a tuple: {k!r}"
        assert agent2.epsilon == 0.0, "epsilon should be 0 after load"

    def test_startup_restores_trained_flag(self):
        """If qtable.json exists at import time, S.trained must be True."""
        self.client.post("/api/train", json={"episodes": 30})
        self._wait_done()
        # Now call _load_qtable on the shared agent to simulate startup
        result = self.app_module._load_qtable(self.app_module.S.agent)
        assert result is not None
        # Also verify S.trained is already True (set by training background thread)
        assert self.app_module.S.trained is True


# ---------------------------------------------------------------------------
# 5. POST /api/evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    @pytest.fixture(autouse=True)
    def setup(self):
        from app import app, S
        self.client = TestClient(app)
        self.S = S

    def _wait_done(self, timeout=180):
        for _ in range(timeout * 2):
            p = self.client.get("/api/train/progress").json()
            if not p["running"]:
                return
            time.sleep(0.5)

    def test_evaluate_before_training_returns_error(self):
        original_trained = self.S.trained
        self.S.trained = False
        try:
            r = self.client.post("/api/evaluate", json={"episodes": 1})
            assert r.status_code == 200
            data = r.json()
            assert "error" in data, "Should return an error dict when not trained"
        finally:
            self.S.trained = original_trained

    def test_evaluate_after_training_structure(self):
        self.client.post("/api/train", json={"episodes": 40})
        self._wait_done()
        r = self.client.post("/api/evaluate", json={"episodes": 3})
        assert r.status_code == 200
        data = r.json()
        for key in ("rule", "rl", "blind_spot"):
            assert key in data, f"Missing top-level key: {key}"

    def test_evaluate_metrics_in_range(self):
        self.client.post("/api/train", json={"episodes": 40})
        self._wait_done()
        r = self.client.post("/api/evaluate", json={"episodes": 3})
        data = r.json()
        for agent_key in ("rule", "rl"):
            m = data[agent_key]
            for metric in ("precision", "recall", "f1"):
                val = m[metric]
                assert 0.0 <= val <= 100.0, (
                    f"{agent_key}.{metric}={val} not in [0, 100]"
                )

    def test_blind_spot_totals(self):
        self.client.post("/api/train", json={"episodes": 40})
        self._wait_done()
        r = self.client.post("/api/evaluate", json={"episodes": 3})
        data = r.json()
        blind = data["blind_spot"]
        assert blind["total"] >= blind["rule_caught"], (
            "blind_spot.total must be >= blind_spot.rule_caught"
        )
