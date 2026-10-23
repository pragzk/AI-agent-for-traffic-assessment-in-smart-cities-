import numpy as np

class PerceptionModule:
    """
    Converts raw zone state into a normalised percept vector.
    Noise kept very small (0.03) — just enough to prevent exact overfitting.
    """

    def get_percept(self, zone_state, add_noise=False):
        d = zone_state['density']
        w = zone_state['weather_encoded']
        t = zone_state['time_flag']
        h = zone_state['historical_risk']
        n = zone_state.get('is_night', 0.0)

        vector = np.array([d, w, t, h, n], dtype=float)

        if add_noise:
            noise  = np.random.normal(0, 0.03, size=vector.shape)
            vector = np.clip(vector + noise, 0.0, 1.0)

        return {
            'zone_id':         zone_state['zone_id'],
            'hour':            zone_state['hour'],
            'weather_name':    zone_state['weather'],
            'density':         round(d, 2),
            'weather_encoded': round(w, 2),
            'time_flag':       t,
            'historical_risk': round(h, 2),
            'is_night':        n,
            'true_danger':     zone_state.get('true_danger', 0),
            'percept_vector':  vector.tolist(),
        }

    def get_all_percepts(self, all_zone_states, add_noise=False):
        return [self.get_percept(z, add_noise=add_noise) for z in all_zone_states]


