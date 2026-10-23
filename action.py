import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datetime


class ActionModule:

    def __init__(self):
        self.alert_log = []
        self.episode   = 0

    def get_risk_level(self, risk_score):
        if risk_score >= 0.60:
            return 'RED'
        elif risk_score >= 0.40:
            return 'AMBER'
        else:
            return 'GREEN'

    def process(self, decision):
        risk_level = self.get_risk_level(decision['risk_score'])
        result = {
            'zone_id':    decision['zone_id'],
            'hour':       decision['hour'],
            'risk_score': decision['risk_score'],
            'risk_level': risk_level,
            'alert':      decision['alert'],
            'weather':    decision['weather'],
            'is_night':   decision['is_night'],
            'breakdown':  decision['breakdown'],
        }
        if decision['alert']:
            self.alert_log.append({
                'episode':    self.episode,
                'zone_id':    decision['zone_id'],
                'hour':       decision['hour'],
                'risk_score': decision['risk_score'],
                'risk_level': risk_level,
                'weather':    decision['weather'],
                'density':    decision['density'],
                'is_night':   decision['is_night'],
                'breakdown':  decision['breakdown'],
                'timestamp':  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
        return result

    def process_all_zones(self, decisions):
        return [self.process(d) for d in decisions]

    def get_dashboard_state(self, results):
        dashboard = {}
        for r in results:
            dashboard[f"Zone {r['zone_id']}"] = {
                'risk_score': r['risk_score'],
                'risk_level': r['risk_level'],
                'alert':      r['alert'],
                'weather':    r['weather'],
                'is_night':   r['is_night'],
            }
        return dashboard

    def get_alert_summary(self):
        if not self.alert_log:
            return "No alerts raised yet."
        total   = len(self.alert_log)
        by_zone = {}
        by_hour = {}
        for entry in self.alert_log:
            by_zone[entry['zone_id']] = by_zone.get(entry['zone_id'], 0) + 1
            by_hour[entry['hour']]    = by_hour.get(entry['hour'], 0) + 1
        night_alerts = sum(1 for e in self.alert_log if e['is_night'] == 1.0)
        return {
            'total_alerts':      total,
            'alerts_by_zone':    by_zone,
            'alerts_by_hour':    by_hour,
            'most_alerted_zone': max(by_zone, key=by_zone.get),
            'most_alerted_hour': max(by_hour, key=by_hour.get),
            'night_alerts':      night_alerts,
            'day_alerts':        total - night_alerts,
        }

    def new_episode(self):
        self.episode += 1


if __name__ == "__main__":
    from city_sim   import CitySimulator
    from perception import PerceptionModule
    from decision   import DecisionModule

    sim        = CitySimulator(n_zones=10)
    perception = PerceptionModule()
    decision   = DecisionModule()
    action     = ActionModule()

    print("=" * 65)
    print("            ACTION MODULE TEST")
    print("=" * 65)

    for hour in range(24):
        raw_states = sim.step()
        percepts   = perception.get_all_percepts(raw_states)
        decisions  = decision.evaluate_all_zones(percepts)
        results    = action.process_all_zones(decisions)

        if hour in [10, 11, 12, 17, 18] or hour in [0, 3, 22]:
            label = "PEAK" if hour in [10,11,12,17,18] else "NIGHT"
            print(f"\n── Dashboard Hour {hour:02d}:00 ({label}) ──")
            print(f"  {'Zone':<8} {'Score':<8} {'Level':<10} {'Alert':<8} {'Weather'}")
            print(f"  {'-'*55}")
            for r in results:
                night = "🌙" if r['is_night'] else "  "
                alert = "⚠ YES" if r['alert'] else "no"
                level_display = {
                    'RED':   '🔴 RED  ',
                    'AMBER': '🟡 AMBER',
                    'GREEN': '🟢 GREEN',
                }[r['risk_level']]
                print(
                    f"  {night} Zone {r['zone_id']:<3} "
                    f"{r['risk_score']:<8.3f}"
                    f"{level_display}  "
                    f"{alert:<8}"
                    f"{r['weather']}"
                )

    print("\n" + "=" * 65)
    print("              ALERT SUMMARY")
    print("=" * 65)
    summary = action.get_alert_summary()
    print(f"\n  Total alerts raised    : {summary['total_alerts']}")
    print(f"  Day alerts             : {summary['day_alerts']}")
    print(f"  Night alerts           : {summary['night_alerts']}")
    print(f"  Most alerted zone      : Zone {summary['most_alerted_zone']}")
    print(f"  Most alerted hour      : {summary['most_alerted_hour']:02d}:00")
    print(f"\n  Alerts per zone:")
    for zone, count in sorted(summary['alerts_by_zone'].items()):
        print(f"    Zone {zone:2d} : {'█' * count} ({count})")
    print(f"\n  Alerts per hour:")
    for h, count in sorted(summary['alerts_by_hour'].items()):
        label = "PEAK" if h in [10,11,12,17,18] else (
                "NIGHT" if h >= 20 or h <= 6 else "OFF-PEAK")
        print(f"    {h:02d}:00 ({label:<8}) : {'█' * count} ({count})")
    print("=" * 65)