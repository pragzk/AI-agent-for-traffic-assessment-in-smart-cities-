import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from city_sim   import CitySimulator
from perception import PerceptionModule
from decision   import DecisionModule
from action     import ActionModule
from rl_agent   import RLAgent

ALERT_THRESHOLD = 0.60   # rule-based threshold
# true_danger uses 0.45 internally in city_sim — agent must learn this


def train_agent(episodes=700, verbose=True):
    sim        = CitySimulator(n_zones=10)
    perception = PerceptionModule()
    decision   = DecisionModule()
    agent      = RLAgent()

    if verbose:
        print("=" * 65)
        print("         PHASE 1 — RL AGENT TRAINING")
        print("=" * 65)

    for ep in range(episodes):
        sim.current_hour = 0
        sim.step_count   = 0
        ep_reward = 0
        ep_alerts = 0

        raw      = sim.step()
        percepts  = perception.get_all_percepts(raw, add_noise=True)
        decisions = decision.evaluate_all_zones(percepts)

        for hour in range(24):
            done = (hour == 23)

            if not done:
                next_raw       = sim.step()
                next_percepts  = perception.get_all_percepts(next_raw, add_noise=True)
                next_decisions = decision.evaluate_all_zones(next_percepts)
            else:
                next_percepts  = percepts
                next_decisions = decisions

            for i, p in enumerate(percepts):
                pv          = p['percept_vector']
                next_pv     = next_percepts[i]['percept_vector']
                risk_score  = decisions[i]['risk_score']
                true_danger = p['true_danger']

                action = agent.choose_action(pv)
                reward = agent.compute_reward(action, true_danger, risk_score)
                agent.update_q_table(pv, action, reward, next_pv, done=done)

                ep_reward += reward
                if action == 1:
                    ep_alerts += 1

            percepts  = next_percepts
            decisions = next_decisions

        agent.reward_history.append(ep_reward)
        agent.alert_history.append(ep_alerts)
        agent.decay_epsilon()

        if verbose and (ep + 1) % 50 == 0:
            avg_r = np.mean(agent.reward_history[-50:])
            print(f"Episode {ep+1:>4} | AvgReward(50): {avg_r:>8.1f} | "
                  f"Alerts: {ep_alerts:>4} | Epsilon: {agent.epsilon:.3f} | "
                  f"Q-states: {len(agent.q_table)}")

    return agent


def run_agent(agent, episodes=10, verbose=True):
    sim        = CitySimulator(n_zones=10)
    perception = PerceptionModule()
    decision   = DecisionModule()
    action_mod = ActionModule()

    agent.epsilon = 0.0

    rule_alerts = rule_correct = rule_false = rule_missed = 0
    rl_alerts   = rl_correct   = rl_false   = rl_missed   = 0
    total_decisions = 0

    for ep in range(episodes):
        sim.current_hour = 0
        sim.step_count   = 0
        action_mod.new_episode()

        for hour in range(24):
            raw_states = sim.step()
            percepts   = perception.get_all_percepts(raw_states, add_noise=False)
            decisions  = decision.evaluate_all_zones(percepts)

            for i, p in enumerate(percepts):
                pv          = p['percept_vector']
                risk_score  = decisions[i]['risk_score']
                true_danger = p['true_danger']

                # Rule-based: fires when risk_score > 0.60
                if risk_score > ALERT_THRESHOLD:
                    rule_alerts += 1
                    if true_danger:
                        rule_correct += 1
                    else:
                        rule_false += 1
                elif true_danger:
                    rule_missed += 1

                # RL agent
                rl_action = agent.choose_action(pv)
                if rl_action == 1:
                    rl_alerts += 1
                    if true_danger:
                        rl_correct += 1
                    else:
                        rl_false += 1
                elif true_danger:
                    rl_missed += 1

                total_decisions += 1

    def pct(a, b):
        return (a / b * 100) if b > 0 else 0

    def f1(p, r):
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0

    rp = pct(rule_correct, rule_alerts)
    rr = pct(rule_correct, rule_correct + rule_missed)
    ap = pct(rl_correct,   rl_alerts)
    ar = pct(rl_correct,   rl_correct + rl_missed)

    total_danger = rl_correct + rl_missed
    print("\n" + "=" * 55)
    print("FINAL RESULTS")
    print("=" * 55)
    print(f"  Truly dangerous zones  : {total_danger} / {total_decisions} "
          f"({pct(total_danger, total_decisions):.1f}%)")
    print("=" * 55)
    print(f"{'Metric':<22} {'Rule-Based':>14} {'RL Agent':>14}")
    print("-" * 55)
    print(f"{'Total Alerts':<22} {rule_alerts:>14} {rl_alerts:>14}")
    print(f"{'Correct (TP)':<22} {rule_correct:>14} {rl_correct:>14}")
    print(f"{'False Alarms (FP)':<22} {rule_false:>14} {rl_false:>14}")
    print(f"{'Missed (FN)':<22} {rule_missed:>14} {rl_missed:>14}")
    print(f"{'Precision':<22} {rp:>13.2f}% {ap:>13.2f}%")
    print(f"{'Recall':<22} {rr:>13.2f}% {ar:>13.2f}%")
    print(f"{'F1 Score':<22} {f1(rp,rr):>13.2f}% {f1(ap,ar):>13.2f}%")
    print("=" * 55)
    print(f"Total decisions evaluated: {total_decisions}")
    print(f"Agent Q-table size: {len(agent.q_table)} states")


if __name__ == "__main__":
    print("TRAINING...")
    agent = train_agent(episodes=700)
    print("\nEVALUATING...")
    run_agent(agent)

