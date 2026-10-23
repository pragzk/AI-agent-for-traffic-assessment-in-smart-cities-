# ── Weights derived from Hu et al. [3] + EDA findings ────────────────
# w1 = traffic density   → dominant risk factor       (0.35)
# w2 = weather encoding  → second highest             (0.25)
# w3 = time of day flag  → peak hour elevation        (0.20)
# w4 = historical risk   → zone baseline              (0.10)
# w5 = night flag        → night severity from EDA    (0.10)
WEIGHTS = {
    'w1_density': 0.35,
    'w2_weather': 0.25,
    'w3_time':    0.20,
    'w4_history': 0.10,
    'w5_night':   0.10,
}

ALERT_THRESHOLD = 0.60


class DecisionModule:
    """
    Computes risk score R for each zone using:
    R = w1*d + w2*w + w3*t + w4*h + w5*n
    Raises alert if R > threshold θ = 0.60.
    """

    def __init__(self):
        self.weights   = WEIGHTS
        self.threshold = ALERT_THRESHOLD

    def compute_risk(self, percept):
        d = percept['density']
        w = percept['weather_encoded']
        t = percept['time_flag']
        h = percept['historical_risk']
        n = percept.get('is_night', 0.0)

        # ✅ FIX: use 'weather_name' — the key perception actually outputs
        weather_name = percept.get('weather_name', 'Unknown')

        R = (self.weights['w1_density'] * d +
             self.weights['w2_weather'] * w +
             self.weights['w3_time']    * t +
             self.weights['w4_history'] * h +
             self.weights['w5_night']   * n)

        R     = round(R, 3)
        alert = R > self.threshold

        return {
            'zone_id':     percept['zone_id'],
            'hour':        percept['hour'],
            # ✅ FIX: store as 'weather' so action.py can read decision['weather']
            'weather':     weather_name,
            'density':     d,
            'weather_enc': w,
            'time_flag':   t,
            'hist_risk':   h,
            'is_night':    n,
            'risk_score':  R,
            'alert':       alert,
            'breakdown': {
                'density_contrib': round(self.weights['w1_density'] * d, 3),
                'weather_contrib': round(self.weights['w2_weather'] * w, 3),
                'time_contrib':    round(self.weights['w3_time']    * t, 3),
                'history_contrib': round(self.weights['w4_history'] * h, 3),
                'night_contrib':   round(self.weights['w5_night']   * n, 3),
            }
        }

    def evaluate_all_zones(self, percepts):
        return [self.compute_risk(p) for p in percepts]


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from city_sim   import CitySimulator
    from perception import PerceptionModule

    sim        = CitySimulator(n_zones=10)
    perception = PerceptionModule()
    decision   = DecisionModule()

    print("=" * 65)
    print("        DECISION MODULE TEST")
    print("=" * 65)

    total_alerts = 0
    for hour in range(24):
        raw_states = sim.step()
        percepts   = perception.get_all_percepts(raw_states)
        decisions  = decision.evaluate_all_zones(percepts)
        alerts     = [d for d in decisions if d['alert']]

        if alerts:
            label = ("PEAK"  if hour in [10,11,12,17,18] else
                     "NIGHT" if hour >= 20 or hour <= 6  else "OFF-PEAK")
            print(f"\n── Hour {hour:02d}:00 ({label}) ──")
            for d in decisions:
                flag  = " ⚠ ALERT" if d['alert'] else ""
                night = "🌙" if d['is_night'] == 1.0 else "  "
                print(
                    f"  {night} Zone {d['zone_id']:2d} | "
                    f"R={d['risk_score']:.3f} | "
                    f"D:{d['density']:.2f} "
                    f"W:{d['weather_enc']:.1f} "
                    f"T:{d['time_flag']:.1f} "
                    f"N:{d['is_night']:.1f} | "
                    f"{d['weather']:15s}{flag}"
                )
            total_alerts += len(alerts)

    print(f"\n  Total alerts in 24h: {total_alerts}")
    print("=" * 65)
