import random
import numpy as np

PEAK_HOURS = [10, 11, 12, 17, 18]

WEATHER_CONDITIONS = {
    'Clear':         0.0,
    'Overcast':      0.3,
    'Rain':          0.6,
    'Freezing Rain': 0.8,
    'Fog':           1.0,
}
WEATHER_PROBS    = [0.50, 0.25, 0.15, 0.05, 0.05]
SEVERITY_WEIGHTS = [0, 0.001, 0.593, 0.405, 0.001]

# ✅ True danger threshold — LOWER than agent's alert threshold (0.60)
# This creates zones the rule-based system MISSES but RL can learn to catch
DANGER_THRESHOLD = 0.45


class Zone:
    def __init__(self, zone_id):
        self.zone_id         = zone_id
        self.historical_risk = round(zone_id / 10, 1)

    def get_state(self, hour):
        if hour in PEAK_HOURS:
            density = round(random.uniform(0.6, 1.0), 2)
        else:
            density = round(random.uniform(0.1, 0.6), 2)

        weather_name    = random.choices(list(WEATHER_CONDITIONS.keys()), weights=WEATHER_PROBS)[0]
        weather_encoded = WEATHER_CONDITIONS[weather_name]
        time_flag       = 1.0 if hour in PEAK_HOURS else 0.0
        is_night        = 1.0 if (hour >= 20 or hour <= 6) else 0.0

        # ✅ DETERMINISTIC true_danger based on risk score
        # Same features -> same danger label every time
        # Q-learning can now converge properly
        risk = (0.35 * density +
                0.25 * weather_encoded +
                0.20 * time_flag +
                0.10 * self.historical_risk +
                0.10 * is_night)

        true_danger = int(risk > DANGER_THRESHOLD)

        return {
            'zone_id':         self.zone_id,
            'hour':            hour,
            'density':         density,
            'weather':         weather_name,
            'weather_encoded': weather_encoded,
            'time_flag':       time_flag,
            'is_night':        is_night,
            'historical_risk': self.historical_risk,
            'true_danger':     true_danger,
        }


class CitySimulator:
    def __init__(self, n_zones=10):
        self.n_zones      = n_zones
        self.zones        = [Zone(i + 1) for i in range(n_zones)]
        self.current_hour = 0
        self.step_count   = 0

    def step(self):
        states            = [zone.get_state(self.current_hour) for zone in self.zones]
        self.current_hour = (self.current_hour + 1) % 24
        self.step_count  += 1
        return states

    def reset(self):
        self.current_hour = 0
        self.step_count   = 0

    def get_info(self):
        return {'current_hour': self.current_hour,
                'step_count':   self.step_count,
                'n_zones':      self.n_zones}


