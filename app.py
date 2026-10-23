"""Backend for the Traffic Risk Agent site.
Place this file next to city_sim.py, perception.py, decision.py, action.py, rl_agent.py, main.py
Run:  pip install fastapi uvicorn numpy   then   uvicorn app:app --reload
"""
import os, sys, json, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from city_sim import CitySimulator
from perception import PerceptionModule
from decision import DecisionModule
from action import ActionModule
from rl_agent import RLAgent
from main import ALERT_THRESHOLD

app = FastAPI(title="Agent-Based Traffic Risk Assessment")
lock = threading.Lock()
HERE = os.path.dirname(os.path.abspath(__file__))
QTABLE_PATH = os.path.join(HERE, "qtable.json")


# ---------------------------------------------------------------------------
# Shared training-progress state (written by background thread, read by API)
# ---------------------------------------------------------------------------
_progress: dict = {
    "running": False,
    "episode": 0,
    "total": 0,
    "reward_history_so_far": [],
    "q_states": 0,
}
_progress_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_qtable(agent: RLAgent, training_summary: dict) -> None:
    """Persist q_table (tuple keys → string), reward_history, alert_history."""
    payload = {
        "q_table": {str(k): v for k, v in agent.q_table.items()},
        "reward_history": agent.reward_history,
        "alert_history": agent.alert_history,
        "training_summary": training_summary,
    }
    with open(QTABLE_PATH, "w") as f:
        json.dump(payload, f)


def _load_qtable(agent: RLAgent) -> dict | None:
    """Load q_table from qtable.json into agent; return training_summary or None."""
    if not os.path.exists(QTABLE_PATH):
        return None
    try:
        with open(QTABLE_PATH) as f:
            payload = json.load(f)
        # Restore q_table: string keys → tuple keys
        agent.q_table = {
            tuple(int(x) for x in k.strip("()").split(", ")) if "," in k
            else tuple(int(x) for x in k.strip("()").split(",")): v
            for k, v in payload.get("q_table", {}).items()
        }
        agent.reward_history = payload.get("reward_history", [])
        agent.alert_history = payload.get("alert_history", [])
        agent.epsilon = 0.0
        return payload.get("training_summary")
    except Exception as e:
        print(f"[app] Warning: could not load qtable.json: {e}")
        return None


# ---------------------------------------------------------------------------
# Shared session state
# ---------------------------------------------------------------------------

class S:  # shared session state
    sim = CitySimulator(10)
    perception = PerceptionModule()
    decision = DecisionModule()
    action = ActionModule()
    agent = RLAgent()
    trained = False
    training = None  # last training summary


# Restore persisted agent on startup
_saved = _load_qtable(S.agent)
if _saved is not None:
    S.trained = True
    S.training = _saved
    print(f"[app] Loaded qtable.json – {len(S.agent.q_table)} Q-states restored.")


# ---------------------------------------------------------------------------
# Background training loop (mirrors main.train_agent)
# ---------------------------------------------------------------------------

def _train_background(episodes: int) -> None:
    """Run the training loop in a background thread, updating _progress each ep."""
    sim = CitySimulator(n_zones=10)
    perception = PerceptionModule()
    decision = DecisionModule()
    agent = RLAgent()

    with _progress_lock:
        _progress["running"] = True
        _progress["episode"] = 0
        _progress["total"] = episodes
        _progress["reward_history_so_far"] = []
        _progress["q_states"] = 0

    for ep in range(episodes):
        sim.current_hour = 0
        sim.step_count = 0
        ep_reward = 0

        raw = sim.step()
        percepts = perception.get_all_percepts(raw, add_noise=True)
        decisions = decision.evaluate_all_zones(percepts)

        for hour in range(24):
            done = (hour == 23)

            if not done:
                next_raw = sim.step()
                next_percepts = perception.get_all_percepts(next_raw, add_noise=True)
                next_decisions = decision.evaluate_all_zones(next_percepts)
            else:
                next_percepts = percepts
                next_decisions = decisions

            for i, p in enumerate(percepts):
                pv = p["percept_vector"]
                next_pv = next_percepts[i]["percept_vector"]
                risk_score = decisions[i]["risk_score"]
                true_danger = p["true_danger"]

                action = agent.choose_action(pv)
                reward = agent.compute_reward(action, true_danger, risk_score)
                agent.update_q_table(pv, action, reward, next_pv, done=done)
                ep_reward += reward

            percepts = next_percepts
            decisions = next_decisions

        agent.reward_history.append(ep_reward)
        agent.alert_history.append(0)  # alert count not tracked here
        agent.decay_epsilon()

        with _progress_lock:
            _progress["episode"] = ep + 1
            _progress["reward_history_so_far"] = list(agent.reward_history)
            _progress["q_states"] = len(agent.q_table)

    # Training complete — commit to shared state
    agent.epsilon = 0.0
    summary = {
        "episodes": episodes,
        "epsilon_final": max(0.01, 0.993 ** episodes),
        "q_states": len(agent.q_table),
        "reward_history": agent.reward_history,
        "alert_history": agent.alert_history,
    }

    _save_qtable(agent, summary)

    with lock:
        S.agent = agent
        S.trained = True
        S.training = summary

    with _progress_lock:
        _progress["running"] = False
        _progress["episode"] = episodes
        _progress["q_states"] = len(agent.q_table)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/status")
def status():
    return {"hour": S.sim.current_hour, "trained": S.trained, "training": S.training,
            "weights": S.decision.weights, "threshold": ALERT_THRESHOLD}


@app.post("/api/reset")
def reset():
    with lock:
        S.sim.reset()
        S.action = ActionModule()
    return {"ok": True}


@app.post("/api/step")
def step():
    """Advance the simulation one hour and return all zone decisions."""
    with lock:
        hour = S.sim.current_hour
        if hour == 0:
            S.action.new_episode()
        raw = S.sim.step()
        percepts = S.perception.get_all_percepts(raw, add_noise=False)
        decisions = S.decision.evaluate_all_zones(percepts)
        results = S.action.process_all_zones(decisions)
        zones = []
        for p, d, r in zip(percepts, decisions, results):
            rl = int(S.agent.choose_action(p["percept_vector"])) if S.trained else None
            zones.append({
                "zone_id": r["zone_id"], "risk_score": r["risk_score"],
                "level": r["risk_level"], "rule_alert": bool(r["alert"]),
                "rl_alert": rl, "weather": r["weather"], "density": d["density"],
                "is_night": r["is_night"], "time_flag": d["time_flag"],
                "history": d["hist_risk"], "true_danger": int(p["true_danger"]),
                "breakdown": r["breakdown"],
            })
        return {"hour": hour, "next_hour": S.sim.current_hour, "zones": zones,
                "alert_log": S.action.alert_log[-40:][::-1],
                "summary": S.action.get_alert_summary()}


# ---------------------------------------------------------------------------
# Training endpoints
# ---------------------------------------------------------------------------

class TrainReq(BaseModel):
    episodes: int = 200


@app.post("/api/train", status_code=200)
def train(req: TrainReq):
    """Start background training. Returns 409 if already running."""
    with _progress_lock:
        if _progress["running"]:
            raise HTTPException(status_code=409, detail="Training is already running.")

    eps = max(10, min(req.episodes, 2000))
    t = threading.Thread(target=_train_background, args=(eps,), daemon=True)
    t.start()
    return {"started": True, "episodes": eps}


@app.get("/api/train/progress")
def train_progress():
    """Poll training progress. Safe to call at any time."""
    with _progress_lock:
        return dict(_progress)


# ---------------------------------------------------------------------------
# Evaluation endpoint
# ---------------------------------------------------------------------------

class EvalReq(BaseModel):
    episodes: int = 10


@app.post("/api/evaluate")
def evaluate(req: EvalReq):
    """Rule-based vs RL on fresh episodes (no noise, epsilon=0), plus the 0.45-0.60 blind spot."""
    if not S.trained:
        return {"error": "Train the agent first."}
    with lock:
        sim = CitySimulator(10)
        m = {k: dict(tp=0, fp=0, fn=0, alerts=0) for k in ("rule", "rl")}
        blind = {"total": 0, "rule_caught": 0, "rl_caught": 0}
        total = danger = 0
        S.agent.epsilon = 0.0
        for _ in range(max(1, min(req.episodes, 100))):
            sim.reset()
            for _h in range(24):
                ps = S.perception.get_all_percepts(sim.step(), add_noise=False)
                ds = S.decision.evaluate_all_zones(ps)
                for p, d in zip(ps, ds):
                    td, R = p["true_danger"], d["risk_score"]
                    acts = {"rule": int(R > ALERT_THRESHOLD),
                            "rl": int(S.agent.choose_action(p["percept_vector"]))}
                    for k, a in acts.items():
                        m[k]["alerts"] += a
                        if a and td: m[k]["tp"] += 1
                        elif a: m[k]["fp"] += 1
                        elif td: m[k]["fn"] += 1
                    total += 1; danger += td
                    if td and 0.45 < R <= ALERT_THRESHOLD:
                        blind["total"] += 1
                        blind["rule_caught"] += acts["rule"]
                        blind["rl_caught"] += acts["rl"]
        for k, v in m.items():
            pr = v["tp"] / v["alerts"] * 100 if v["alerts"] else 0
            rc = v["tp"] / (v["tp"] + v["fn"]) * 100 if (v["tp"] + v["fn"]) else 0
            v.update(precision=pr, recall=rc, f1=(2 * pr * rc / (pr + rc)) if pr + rc else 0)
        return {"decisions": total, "dangerous": danger, "rule": m["rule"], "rl": m["rl"], "blind_spot": blind}
