import numpy as np
import random

LEARNING_RATE   = 0.30   # high LR — deterministic env converges fast
DISCOUNT_FACTOR = 0.80
EPSILON_START   = 1.0
EPSILON_END     = 0.01
EPSILON_DECAY   = 0.993
N_BINS          = 6      # 6^5 = 7776 possible states — enough resolution

class RLAgent:
    def __init__(self):
        self.epsilon        = EPSILON_START
        self.lr             = LEARNING_RATE
        self.gamma          = DISCOUNT_FACTOR
        self.n_bins         = N_BINS
        self.q_table        = {}
        self.episode        = 0
        self.reward_history = []
        self.alert_history  = []

    def discretise(self, percept_vector):
        bins    = np.linspace(0, 1, self.n_bins + 1)
        clipped = np.clip(percept_vector, 0.0, 1.0)
        return tuple(int(np.digitize(v, bins) - 1) for v in clipped)

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
        return self.q_table[state]

    def choose_action(self, percept_vector):
        state = self.discretise(percept_vector)
        if random.random() < self.epsilon:
            return random.choice([0, 1])
        return int(np.argmax(self.get_q_values(state)))

    def compute_reward(self, action, true_danger, risk_score):
        """
        Now that true_danger is deterministic, Q-values will converge cleanly.
        Danger rate ~28% so safe zones appear ~2.6x more often.
        
        Verified math:
          Selective policy: 0.28*+15 + 0.72*+2  = +5.6 per step
          Alert-all:        0.28*+15 + 0.72*-8  = -1.6 per step
          Alert-none:       0.28*-15 + 0.72*+2  = -2.8 per step
        Only selective alerting is profitable.
        """
        if true_danger:
            return +15 if action == 1 else -15
        else:
            return -8  if action == 1 else +2

    def update_q_table(self, percept_vector, action, reward,
                       next_percept_vector, done=False):
        state      = self.discretise(percept_vector)
        next_state = self.discretise(next_percept_vector)
        q_vals      = self.get_q_values(state)
        next_q_vals = self.get_q_values(next_state)
        future      = 0.0 if done else self.gamma * max(next_q_vals)
        q_vals[action] += self.lr * (reward + future - q_vals[action])

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        self.episode += 1

    def get_stats(self):
        return {
            'episodes'          : self.episode,
            'epsilon'           : round(self.epsilon, 4),
            'q_table_size'      : len(self.q_table),
            'avg_reward_last10' : round(
                np.mean(self.reward_history[-10:])
                if len(self.reward_history) >= 10 else 0, 3),
        }



