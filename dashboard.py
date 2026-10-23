# dashboard.py — Intelligent Traffic Risk Assessment Dashboard
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import time

from city_sim   import CitySimulator
from perception import PerceptionModule
from decision   import DecisionModule
from action     import ActionModule
from rl_agent   import RLAgent

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEXUS — Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS — dark industrial theme ────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

  /* Global reset */
  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0A0E1A;
    color: #E8EAF0;
  }

  /* Main background */
  .stApp {
    background: linear-gradient(135deg, #0A0E1A 0%, #0D1220 50%, #080C18 100%);
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F1525 0%, #0A0E1A 100%);
    border-right: 1px solid #1E2640;
  }
  [data-testid="stSidebar"] * { color: #C8D0E0 !important; }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {
    color: #7EB8F7 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
  }

  /* Title area */
  h1 { 
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.4rem !important;
    background: linear-gradient(90deg, #7EB8F7, #A78BFA, #F472B6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
  }
  h2 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: #7EB8F7 !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    border-left: 3px solid #7EB8F7;
    padding-left: 10px;
  }
  h3 {
    font-family: 'Space Mono', monospace !important;
    color: #A78BFA !important;
    font-size: 0.85rem !important;
  }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: linear-gradient(135deg, #111827 0%, #0F1520 100%);
    border: 1px solid #1E2A40;
    border-radius: 8px;
    padding: 16px 20px !important;
    position: relative;
    overflow: hidden;
  }
  [data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #7EB8F7, #A78BFA);
  }
  [data-testid="stMetricLabel"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    color: #6B7A99 !important;
    text-transform: uppercase !important;
  }
  [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1.6rem !important;
    color: #E8EAF0 !important;
  }
  [data-testid="stMetricDelta"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
  }

  /* Dataframe */
  [data-testid="stDataFrame"] {
    border: 1px solid #1E2A40 !important;
    border-radius: 8px !important;
    overflow: hidden;
  }
  .dataframe { color: #C8D0E0 !important; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #2563EB, #7C3AED) !important;
    color: white !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 12px 28px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 20px rgba(37, 99, 235, 0.3) !important;
  }
  .stButton > button:hover {
    box-shadow: 0 0 30px rgba(124, 58, 237, 0.5) !important;
    transform: translateY(-1px) !important;
  }

  /* Sliders */
  .stSlider [data-testid="stTickBarMin"],
  .stSlider [data-testid="stTickBarMax"] {
    color: #6B7A99 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.65rem !important;
  }

  /* Info boxes */
  .stAlert {
    background: #111827 !important;
    border: 1px solid #1E2A40 !important;
    border-radius: 8px !important;
    color: #C8D0E0 !important;
  }

  /* Divider */
  hr { border-color: #1E2A40 !important; }

  /* Spinner */
  .stSpinner > div { border-top-color: #7EB8F7 !important; }

  /* Status badge */
  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }
  .badge-red   { background: rgba(239,68,68,0.15); color: #F87171; border: 1px solid #F87171; }
  .badge-amber { background: rgba(251,191,36,0.15); color: #FCD34D; border: 1px solid #FCD34D; }
  .badge-green { background: rgba(52,211,153,0.15); color: #34D399; border: 1px solid #34D399; }
  .badge-blue  { background: rgba(126,184,247,0.15); color: #7EB8F7; border: 1px solid #7EB8F7; }

  /* Zone card grid */
  .zone-card {
    background: linear-gradient(135deg, #111827 0%, #0F1520 100%);
    border: 1px solid #1E2A40;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 8px;
    position: relative;
    overflow: hidden;
  }
  .zone-card-red    { border-top: 3px solid #F87171; }
  .zone-card-amber  { border-top: 3px solid #FCD34D; }
  .zone-card-green  { border-top: 3px solid #34D399; }

  /* Comparison table header */
  .compare-header {
    background: linear-gradient(135deg, #111827, #0F1520);
    border: 1px solid #1E2A40;
    border-radius: 8px 8px 0 0;
    padding: 12px 16px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7EB8F7;
    margin-bottom: 0;
  }

  /* Progress bars */
  .risk-bar-container {
    background: #1A2235;
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
    margin-top: 4px;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: #0A0E1A; }
  ::-webkit-scrollbar-thumb { background: #1E2A40; border-radius: 2px; }

  /* Caption text */
  .stCaption, small { color: #6B7A99 !important; font-family: 'Space Mono', monospace !important; font-size: 0.65rem !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1E2A40;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #6B7A99 !important;
    border-radius: 6px !important;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1E3A5F, #2D1B69) !important;
    color: #7EB8F7 !important;
  }
</style>
""", unsafe_allow_html=True)

# ── matplotlib global style ───────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#0D1220',
    'axes.facecolor':    '#111827',
    'axes.edgecolor':    '#1E2A40',
    'axes.labelcolor':   '#8892AA',
    'axes.titlecolor':   '#C8D0E0',
    'xtick.color':       '#6B7A99',
    'ytick.color':       '#6B7A99',
    'grid.color':        '#1A2235',
    'text.color':        '#C8D0E0',
    'legend.facecolor':  '#111827',
    'legend.edgecolor':  '#1E2A40',
    'font.family':       'monospace',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

ACCENT_BLUE   = '#7EB8F7'
ACCENT_PURPLE = '#A78BFA'
ACCENT_PINK   = '#F472B6'
COLOR_RED     = '#F87171'
COLOR_AMBER   = '#FCD34D'
COLOR_GREEN   = '#34D399'


# ── Header ────────────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("# NEXUS // Traffic Intelligence")
    st.markdown(
        '<p style="font-family:\'Space Mono\',monospace;font-size:0.75rem;'
        'color:#6B7A99;letter-spacing:0.12em;margin-top:-10px;">'
        'ZONE-WISE RISK MONITORING — RL-ENHANCED ALERTING SYSTEM</p>',
        unsafe_allow_html=True
    )
with col_status:
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(
        '<span class="badge badge-blue">● SYSTEM ONLINE</span>',
        unsafe_allow_html=True
    )

st.markdown('<hr style="margin:6px 0 20px 0;">', unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚙ Controls")
st.sidebar.markdown('<hr style="margin:4px 0 12px 0;">', unsafe_allow_html=True)

n_train_episodes = st.sidebar.slider(
    "Training Episodes", 100, 700, 400, step=100,
    help="More episodes = better RL agent (but slower to train)"
)
selected_hour = st.sidebar.slider(
    "Simulation Hour", 0, 23, 12,
    format="%02d:00"
)
run_button = st.sidebar.button("▶  LAUNCH SIMULATION", type="primary", use_container_width=True)

st.sidebar.markdown('<hr style="margin:16px 0 12px 0;">', unsafe_allow_html=True)
st.sidebar.markdown("## ◈ Legend")
st.sidebar.markdown(
    '<div style="font-family:\'Space Mono\',monospace;font-size:0.7rem;line-height:2;">'
    '<span style="color:#F87171;">■</span> RED &nbsp;&nbsp; R > 0.60 — ALERT<br>'
    '<span style="color:#FCD34D;">■</span> AMBER  R 0.40–0.60 — CAUTION<br>'
    '<span style="color:#34D399;">■</span> GREEN  R < 0.40 — SAFE'
    '</div>',
    unsafe_allow_html=True
)
st.sidebar.markdown('<hr style="margin:12px 0;">', unsafe_allow_html=True)
st.sidebar.markdown("## ◈ Thresholds")
st.sidebar.markdown(
    '<div style="font-family:\'Space Mono\',monospace;font-size:0.7rem;line-height:2;color:#8892AA;">'
    'Rule alert threshold: <span style="color:#F87171;">0.60</span><br>'
    'True danger threshold: <span style="color:#FCD34D;">0.45</span><br>'
    'RL learns the gap between them'
    '</div>',
    unsafe_allow_html=True
)


# ── Helpers ───────────────────────────────────────────────────────────
def zone_color(risk_level):
    return {'RED': COLOR_RED, 'AMBER': COLOR_AMBER, 'GREEN': COLOR_GREEN}[risk_level]

def zone_badge_class(risk_level):
    return {'RED': 'badge-red', 'AMBER': 'badge-amber', 'GREEN': 'badge-green'}[risk_level]


# ── Cached training ───────────────────────────────────────────────────
@st.cache_resource
def get_trained_agent(episodes):
    """Train the RL agent with the fixed sliding-window loop."""
    sim        = CitySimulator(n_zones=10)
    perception = PerceptionModule()
    decision   = DecisionModule()
    agent      = RLAgent()

    for ep in range(episodes):
        sim.current_hour = 0
        sim.step_count   = 0

        raw      = sim.step()
        percepts = perception.get_all_percepts(raw, add_noise=True)
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

            percepts  = next_percepts
            decisions = next_decisions

        agent.reward_history.append(0)
        agent.decay_epsilon()

    agent.epsilon = 0.0
    return agent


# ── Main simulation ───────────────────────────────────────────────────
if run_button:

    with st.spinner("Initialising neural pathways..."):
        trained_agent = get_trained_agent(n_train_episodes)

        sim        = CitySimulator(n_zones=10)
        perception = PerceptionModule()
        decision   = DecisionModule()
        action_mod = ActionModule()

        all_results = []
        all_percepts = []
        all_decisions = []

        for h in range(selected_hour + 1):
            raw_states = sim.step()
            percepts   = perception.get_all_percepts(raw_states, add_noise=False)
            decisions  = decision.evaluate_all_zones(percepts)
            results    = action_mod.process_all_zones(decisions)
            if h == selected_hour:
                all_results   = results
                all_percepts  = percepts
                all_decisions = decisions

    # Compute RL decisions
    rl_actions = []
    for p in all_percepts:
        rl_actions.append(trained_agent.choose_action(p['percept_vector']))

    # ── KPI bar ───────────────────────────────────────────────────────
    danger_zones    = sum(1 for p in all_percepts if p['true_danger'])
    rule_alerts_now = sum(1 for r in all_results if r['alert'])
    rl_alerts_now   = sum(rl_actions)
    rl_correct_now  = sum(1 for i, a in enumerate(rl_actions) if a == 1 and all_percepts[i]['true_danger'])
    hour_label      = "PEAK" if selected_hour in [10,11,12,17,18] else (
                      "NIGHT" if selected_hour >= 20 or selected_hour <= 6 else "OFF-PEAK")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("SIMULATION HOUR", f"{selected_hour:02d}:00", hour_label)
    k2.metric("TRAINING EPISODES", f"{n_train_episodes:,}")
    k3.metric("TRULY DANGEROUS", f"{danger_zones} / 10")
    k4.metric("RULE ALERTS", rule_alerts_now)
    k5.metric("RL ALERTS", rl_alerts_now, f"{rl_correct_now} correct")

    st.markdown('<hr style="margin:20px 0;">', unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📍  Zone Status", "⚔  Rule vs RL", "🗺  Heatmap", "📈  Learning Curve"
    ])

    # ════════════════════════════════════════
    # TAB 1 — Zone Status
    # ════════════════════════════════════════
    with tab1:
        st.markdown("## Zone Risk Map")
        st.markdown(
            f'<p style="font-family:\'Space Mono\',monospace;font-size:0.72rem;color:#6B7A99;">'
            f'Hour {selected_hour:02d}:00 — All 10 zones — True danger threshold R* > 0.45</p>',
            unsafe_allow_html=True
        )

        cols = st.columns(5)
        for i, r in enumerate(all_results):
            pv         = all_percepts[i]['percept_vector']
            true_danger = all_percepts[i]['true_danger']
            rl_alert   = rl_actions[i]
            col        = cols[i % 5]

            with col:
                color    = zone_color(r['risk_level'])
                icon     = "🔴" if r['risk_level']=="RED" else ("🟡" if r['risk_level']=="AMBER" else "🟢")
                night_ic = "🌙" if r['is_night'] else "☀️"
                danger_label = "⚠ DANGER" if true_danger else "SAFE"
                danger_color = "#F87171" if true_danger else "#34D399"
                rl_label     = "🤖 ALERT" if rl_alert else "🤖 SILENT"

                st.markdown(f"""
                <div class="zone-card zone-card-{'red' if r['risk_level']=='RED' else 'amber' if r['risk_level']=='AMBER' else 'green'}">
                  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#6B7A99;letter-spacing:0.1em;margin-bottom:4px;">ZONE {r['zone_id']:02d}</div>
                  <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.8rem;color:{color};line-height:1;">{r['risk_score']:.3f}</div>
                  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:{color};margin-top:2px;">{icon} {r['risk_level']}</div>
                  <div class="risk-bar-container"><div style="background:{color};height:6px;width:{int(r['risk_score']*100)}%;border-radius:4px;"></div></div>
                  <div style="margin-top:8px;font-family:'Space Mono',monospace;font-size:0.6rem;color:#6B7A99;">
                    {night_ic} {r['weather'][:8]}<br>
                    <span style="color:{danger_color};">{danger_label}</span> &nbsp;·&nbsp; {rl_label}
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<hr style="margin:20px 0;">', unsafe_allow_html=True)

        # Bar chart
        st.markdown("## Risk Score Distribution")
        fig, ax = plt.subplots(figsize=(13, 3.5))

        zones  = [f"Z{r['zone_id']}" for r in all_results]
        scores = [r['risk_score'] for r in all_results]
        colors = [zone_color(r['risk_level']) for r in all_results]
        danger = [all_percepts[i]['true_danger'] for i in range(10)]

        bars = ax.bar(zones, scores, color=colors, width=0.6,
                      edgecolor='#0D1220', linewidth=1.5, zorder=3)

        # Danger marker dots
        for i, (z, d) in enumerate(zip(zones, danger)):
            if d:
                ax.scatter(i, scores[i] + 0.03, marker='▼', s=80,
                           color='#F87171', zorder=5)

        ax.axhline(0.60, color=COLOR_RED,   linestyle='--', linewidth=1.2,
                   alpha=0.7, label='Rule alert threshold (0.60)', zorder=4)
        ax.axhline(0.45, color=COLOR_AMBER, linestyle=':', linewidth=1.2,
                   alpha=0.7, label='True danger threshold (0.45)', zorder=4)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Risk Score R', fontsize=8)
        ax.set_title(f'Zone Risk Scores — Hour {selected_hour:02d}:00  (▼ = truly dangerous)',
                     fontsize=9, pad=10)
        ax.legend(fontsize=7.5, loc='upper left')
        ax.grid(axis='y', alpha=0.3, zorder=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ════════════════════════════════════════
    # TAB 2 — Rule vs RL
    # ════════════════════════════════════════
    with tab2:
        st.markdown("## Rule-Based  vs  RL Agent — Head to Head")

        # Summary metrics
        s1, s2, s3 = st.columns(3)
        rule_tp = sum(1 for i,r in enumerate(all_results) if r['alert'] and all_percepts[i]['true_danger'])
        rule_fp = sum(1 for i,r in enumerate(all_results) if r['alert'] and not all_percepts[i]['true_danger'])
        rule_fn = sum(1 for i,r in enumerate(all_results) if not r['alert'] and all_percepts[i]['true_danger'])
        rl_tp   = sum(1 for i,a in enumerate(rl_actions) if a==1 and all_percepts[i]['true_danger'])
        rl_fp   = sum(1 for i,a in enumerate(rl_actions) if a==1 and not all_percepts[i]['true_danger'])
        rl_fn   = sum(1 for i,a in enumerate(rl_actions) if a==0 and all_percepts[i]['true_danger'])

        with s1:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#111827,#0F1520);border:1px solid #1E2A40;border-radius:8px;padding:16px;">
              <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#6B7A99;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">Truly Dangerous Zones</div>
              <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:2rem;color:#7EB8F7;">{danger_zones}</div>
              <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#6B7A99;">of 10 zones this hour</div>
            </div>
            """, unsafe_allow_html=True)
        with s2:
            rule_recall = f"{rule_tp/max(danger_zones,1)*100:.0f}%"
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#111827,#0F1520);border:1px solid #1E2A40;border-radius:8px;padding:16px;">
              <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#6B7A99;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">Rule-Based Recall</div>
              <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:2rem;color:#FCD34D;">{rule_recall}</div>
              <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#6B7A99;">TP={rule_tp} FP={rule_fp} FN={rule_fn}</div>
            </div>
            """, unsafe_allow_html=True)
        with s3:
            rl_recall_pct = f"{rl_tp/max(danger_zones,1)*100:.0f}%"
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#111827,#0F1520);border:1px solid #1E2A40;border-radius:8px;padding:16px;">
              <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#6B7A99;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">RL Agent Recall</div>
              <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:2rem;color:#34D399;">{rl_recall_pct}</div>
              <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#6B7A99;">TP={rl_tp} FP={rl_fp} FN={rl_fn}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)

        # Comparison table
        col1, col2 = st.columns(2)
        table_rows = []
        for i, r in enumerate(all_results):
            true_d   = all_percepts[i]['true_danger']
            rl_alert = rl_actions[i]
            rule_a   = r['alert']
            table_rows.append({
                'Zone':        f"Zone {r['zone_id']}",
                'Risk Score':  f"{r['risk_score']:.3f}",
                'True Danger': '⚠ YES' if true_d else 'No',
                'Weather':     r['weather'],
                'Rule Alert':  '🔔 ALERT' if rule_a else '— quiet',
                'RL Alert':    '🔔 ALERT' if rl_alert else '— quiet',
                'RL Correct':  '✅' if (rl_alert == true_d) else '❌',
            })

        df = pd.DataFrame(table_rows)

        with col1:
            st.markdown('<div class="compare-header">RULE-BASED SYSTEM</div>', unsafe_allow_html=True)
            st.dataframe(
                df[['Zone','Risk Score','True Danger','Rule Alert']],
                use_container_width=True, hide_index=True
            )

        with col2:
            st.markdown('<div class="compare-header">RL AGENT</div>', unsafe_allow_html=True)
            st.dataframe(
                df[['Zone','Risk Score','True Danger','RL Alert','RL Correct']],
                use_container_width=True, hide_index=True
            )

        st.markdown('<br>', unsafe_allow_html=True)

        # Side by side bar comparison
        st.markdown("## Alert Comparison — This Hour")
        fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))

        for ax_i, (ax, label, alerts, color_alert) in enumerate(zip(
            axes,
            ['Rule-Based System', 'RL Agent'],
            [[r['alert'] for r in all_results], rl_actions],
            [COLOR_RED, ACCENT_PURPLE]
        )):
            zone_labels = [f"Z{r['zone_id']}" for r in all_results]
            scores_vals = [r['risk_score'] for r in all_results]
            bar_colors  = []
            for j, alert in enumerate(alerts):
                true_d = all_percepts[j]['true_danger']
                if alert and true_d:       bar_colors.append(COLOR_GREEN)    # TP
                elif alert and not true_d: bar_colors.append(COLOR_RED)      # FP
                elif not alert and true_d: bar_colors.append(COLOR_AMBER)    # FN
                else:                      bar_colors.append('#1E2A40')       # TN

            ax.bar(zone_labels, scores_vals, color=bar_colors,
                   edgecolor='#0D1220', linewidth=1.2, width=0.6)
            ax.axhline(0.60, color=COLOR_RED,   linestyle='--', linewidth=1, alpha=0.6)
            ax.axhline(0.45, color=COLOR_AMBER, linestyle=':', linewidth=1, alpha=0.6)
            ax.set_ylim(0, 1.05)
            ax.set_title(label, fontsize=9, pad=8, color='#C8D0E0')
            ax.set_ylabel('Risk Score', fontsize=7.5)
            ax.grid(axis='y', alpha=0.25)

        tp_patch = mpatches.Patch(color=COLOR_GREEN, label='True Positive')
        fp_patch = mpatches.Patch(color=COLOR_RED,   label='False Positive')
        fn_patch = mpatches.Patch(color=COLOR_AMBER, label='False Negative')
        tn_patch = mpatches.Patch(color='#1E2A40',   label='True Negative')
        axes[1].legend(handles=[tp_patch,fp_patch,fn_patch,tn_patch],
                       fontsize=7, loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ════════════════════════════════════════
    # TAB 3 — Heatmap
    # ════════════════════════════════════════
    with tab3:
        st.markdown("## 24-Hour Risk Heatmap")
        st.markdown(
            '<p style="font-family:\'Space Mono\',monospace;font-size:0.72rem;color:#6B7A99;">'
            'Risk score R across all zones and all hours. Darker red = higher risk.</p>',
            unsafe_allow_html=True
        )

        sim2        = CitySimulator(n_zones=10)
        perception2 = PerceptionModule()
        decision2   = DecisionModule()

        heatmap_data   = np.zeros((10, 24))
        danger_data    = np.zeros((10, 24), dtype=int)
        rl_alert_data  = np.zeros((10, 24), dtype=int)
        rule_alert_data= np.zeros((10, 24), dtype=int)

        for h in range(24):
            raw  = sim2.step()
            pcs  = perception2.get_all_percepts(raw, add_noise=False)
            decs = decision2.evaluate_all_zones(pcs)
            for d_i, (pc, dc) in enumerate(zip(pcs, decs)):
                heatmap_data[d_i][h]    = dc['risk_score']
                danger_data[d_i][h]     = pc['true_danger']
                rl_alert_data[d_i][h]   = trained_agent.choose_action(pc['percept_vector'])
                rule_alert_data[d_i][h] = int(dc['risk_score'] > 0.60)

        # Custom dark-to-red colormap
        cmap_custom = LinearSegmentedColormap.from_list(
            'nexus', ['#0D1F35', '#1A3A5C', '#FCD34D', '#F87171', '#DC2626'], N=256
        )

        fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5),
                                   gridspec_kw={'width_ratios': [6, 1, 1], 'wspace': 0.08})

        # Main heatmap
        im = axes2[0].imshow(heatmap_data, aspect='auto',
                             cmap=cmap_custom, vmin=0, vmax=1, interpolation='nearest')
        axes2[0].set_xticks(range(24))
        axes2[0].set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=6.5)
        axes2[0].set_yticks(range(10))
        axes2[0].set_yticklabels([f"Zone {i+1}" for i in range(10)], fontsize=7.5)
        axes2[0].set_xlabel("Hour of Day", fontsize=8)
        axes2[0].set_title("Risk Score R  (0–1)", fontsize=9, pad=8)

        # Mark selected hour
        axes2[0].axvline(x=selected_hour, color=ACCENT_BLUE, linewidth=2,
                         alpha=0.8, linestyle='-')
        axes2[0].text(selected_hour + 0.3, -0.8, f"▲ {selected_hour:02d}h",
                      color=ACCENT_BLUE, fontsize=6.5)

        # Peak hour markers
        for ph in [10, 11, 12, 17, 18]:
            axes2[0].axvline(x=ph, color=COLOR_AMBER, linewidth=0.5, alpha=0.3)

        plt.colorbar(im, ax=axes2[0], fraction=0.02, pad=0.02, label='Risk Score')

        # True danger heatmap
        axes2[1].imshow(danger_data, aspect='auto',
                        cmap=LinearSegmentedColormap.from_list('d', ['#111827','#F87171']),
                        vmin=0, vmax=1, interpolation='nearest')
        axes2[1].set_xticks([])
        axes2[1].set_yticks(range(10))
        axes2[1].set_yticklabels([])
        axes2[1].set_title("True\nDanger", fontsize=7.5, pad=8)

        # RL alert heatmap
        axes2[2].imshow(rl_alert_data, aspect='auto',
                        cmap=LinearSegmentedColormap.from_list('rl', ['#111827','#A78BFA']),
                        vmin=0, vmax=1, interpolation='nearest')
        axes2[2].set_xticks([])
        axes2[2].set_yticks(range(10))
        axes2[2].set_yticklabels([])
        axes2[2].set_title("RL\nAlerts", fontsize=7.5, pad=8)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # Hourly danger summary bar
        st.markdown("## Danger Frequency by Hour")
        fig3, ax3 = plt.subplots(figsize=(13, 2.5))
        hour_danger_counts = danger_data.sum(axis=0)
        hour_rl_counts     = rl_alert_data.sum(axis=0)

        x = np.arange(24)
        ax3.bar(x, hour_danger_counts, color=COLOR_RED,   alpha=0.5,
                width=0.8, label='True Danger Zones')
        ax3.bar(x, hour_rl_counts,     color=ACCENT_PURPLE, alpha=0.4,
                width=0.4, label='RL Alerts', align='edge')

        peak_hrs = [10, 11, 12, 17, 18]
        for ph in peak_hrs:
            ax3.axvspan(ph - 0.5, ph + 0.5, alpha=0.1, color=COLOR_AMBER)

        ax3.set_xticks(range(24))
        ax3.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
        ax3.set_ylabel('Count (zones)', fontsize=7.5)
        ax3.set_title('Zones in True Danger vs RL Alerts per Hour  (amber = peak hours)',
                      fontsize=8.5, pad=8)
        ax3.legend(fontsize=7.5)
        ax3.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    # ════════════════════════════════════════
    # TAB 4 — Learning Curve
    # ════════════════════════════════════════
    with tab4:
        st.markdown("## RL Agent Learning Curve")
        st.markdown(
            '<p style="font-family:\'Space Mono\',monospace;font-size:0.72rem;color:#6B7A99;">'
            'Training reward over episodes. Positive = agent making correct decisions.</p>',
            unsafe_allow_html=True
        )

        with st.spinner("Running training curve..."):
            agent_c   = RLAgent()
            sim_c     = CitySimulator(n_zones=10)
            perc_c    = PerceptionModule()
            dec_c     = DecisionModule()
            curve_eps = min(n_train_episodes, 400)
            rewards   = []
            alert_counts = []

            for ep in range(curve_eps):
                sim_c.current_hour = 0
                sim_c.step_count   = 0
                ep_reward = 0
                ep_alerts = 0

                raw = sim_c.step()
                ps  = perc_c.get_all_percepts(raw, add_noise=True)
                ds  = dec_c.evaluate_all_zones(ps)

                for hour in range(24):
                    done = (hour == 23)
                    if not done:
                        nraw = sim_c.step()
                        nps  = perc_c.get_all_percepts(nraw, add_noise=True)
                        nds  = dec_c.evaluate_all_zones(nps)
                    else:
                        nps = ps; nds = ds

                    for i, p in enumerate(ps):
                        pv = p['percept_vector']
                        td = p['true_danger']
                        rs = ds[i]['risk_score']
                        a  = agent_c.choose_action(pv)
                        r  = agent_c.compute_reward(a, td, rs)
                        agent_c.update_q_table(pv, a, r, nps[i]['percept_vector'], done=done)
                        ep_reward += r
                        if a == 1: ep_alerts += 1

                    ps = nps; ds = nds

                rewards.append(ep_reward)
                alert_counts.append(ep_alerts)
                agent_c.decay_epsilon()

        fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

        # Reward plot
        ax4a.plot(rewards, color=ACCENT_BLUE, alpha=0.25, linewidth=0.8)
        window = 15
        smoothed = pd.Series(rewards).rolling(window).mean()
        ax4a.plot(smoothed, color=ACCENT_BLUE, linewidth=2.5, label=f'{window}-ep moving avg')
        ax4a.axhline(0, color='#4A5568', linewidth=0.8, linestyle='--')
        ax4a.fill_between(range(len(rewards)), rewards, 0,
                          where=[r > 0 for r in rewards],
                          alpha=0.08, color=COLOR_GREEN)
        ax4a.fill_between(range(len(rewards)), rewards, 0,
                          where=[r < 0 for r in rewards],
                          alpha=0.08, color=COLOR_RED)
        ax4a.set_ylabel('Episode Reward', fontsize=8)
        ax4a.set_title('Training Reward — Convergence to Positive Territory', fontsize=9, pad=8)
        ax4a.legend(fontsize=7.5)
        ax4a.grid(alpha=0.2)

        # Alert count plot
        ax4b.plot(alert_counts, color=ACCENT_PURPLE, alpha=0.25, linewidth=0.8)
        smoothed_a = pd.Series(alert_counts).rolling(window).mean()
        ax4b.plot(smoothed_a, color=ACCENT_PURPLE, linewidth=2.5, label=f'{window}-ep moving avg')
        # Reference line: expected dangerous zones ~240 * 0.275 = 66
        ax4b.axhline(66, color=COLOR_AMBER, linewidth=1, linestyle='--',
                     alpha=0.6, label='Expected danger zones (~66)')
        ax4b.set_xlabel('Training Episode', fontsize=8)
        ax4b.set_ylabel('Alerts per Episode', fontsize=8)
        ax4b.set_title('Alert Frequency — Converging to Optimal ~66 Alerts/Episode', fontsize=9, pad=8)
        ax4b.legend(fontsize=7.5)
        ax4b.grid(alpha=0.2)

        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

        # Final stats
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown("## Training Summary")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Final Episode Reward",    f"{rewards[-1]:.0f}")
        f2.metric("Avg Reward (last 50)",    f"{np.mean(rewards[-50:]):.0f}")
        f3.metric("Episodes to Positive",
                  next((i for i, r in enumerate(rewards) if r > 0), curve_eps))
        f4.metric("Final Alerts/Episode",    f"{alert_counts[-1]}")

else:
    # ── Landing screen ─────────────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:60px 40px;background:linear-gradient(135deg,#111827,#0F1520);
                border:1px solid #1E2A40;border-radius:12px;max-width:700px;margin:0 auto;">
      <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#6B7A99;
                  letter-spacing:0.2em;text-transform:uppercase;margin-bottom:12px;">
        NEXUS Traffic Intelligence System
      </div>
      <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:2rem;
                  background:linear-gradient(90deg,#7EB8F7,#A78BFA,#F472B6);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  margin-bottom:16px;">
        Reinforcement Learning<br>Traffic Safety Monitor
      </div>
      <div style="font-family:'Syne',sans-serif;font-size:0.95rem;color:#8892AA;
                  line-height:1.7;max-width:500px;margin:0 auto 24px;">
        A Q-learning agent trained to detect dangerous traffic zones — 
        achieving <span style="color:#34D399;font-weight:700;">96.96% F1</span> 
        vs <span style="color:#FCD34D;">48.05%</span> for rule-based systems.
      </div>
      <div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-bottom:8px;">
        <span class="badge badge-blue">10 City Zones</span>
        <span class="badge badge-green">95.06% Precision</span>
        <span class="badge badge-green">98.94% Recall</span>
        <span class="badge badge-blue">Q-Learning Agent</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in zip(
        [c1, c2, c3],
        ["📍", "⚔", "🗺"],
        ["Zone Risk Map", "Rule vs RL", "24h Heatmap"],
        ["Live risk scores and alert status for all 10 zones at the selected hour",
         "Side-by-side comparison of rule-based vs RL agent decisions",
         "Full danger landscape across all zones and hours of the day"]
    ):
        with col:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#111827,#0F1520);border:1px solid #1E2A40;
                        border-radius:10px;padding:20px;text-align:center;height:140px;">
              <div style="font-size:1.8rem;margin-bottom:8px;">{icon}</div>
              <div style="font-family:'Syne',sans-serif;font-weight:700;color:#7EB8F7;
                          font-size:0.9rem;margin-bottom:6px;">{title}</div>
              <div style="font-family:'Syne',sans-serif;font-size:0.78rem;color:#6B7A99;
                          line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<br><br>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;font-family:\'Space Mono\',monospace;font-size:0.65rem;'
        'color:#3A4560;letter-spacing:0.15em;">SET PARAMETERS IN SIDEBAR → CLICK LAUNCH SIMULATION</p>',
        unsafe_allow_html=True
    )
