"""
STEP 2 - The web app itself.

Run with (in Anaconda Prompt or Spyder's terminal, NOT with F5):
    streamlit run app.py

Reads dashboard_data.csv (created by train_and_save_dashboard_data.py)
and displays it as an interactive dashboard.
"""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# -----------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------
DATA_PATH = "C:/Users/monic/Desktop/project_practica/dashboard_data.csv"
RISK_HIGH = 0.7   # above this = red
RISK_MEDIUM = 0.4  # above this = yellow, below = green

st.set_page_config(page_title="Long COVID Monitoring", layout="wide", page_icon="💙")

# -----------------------------------------------------------------
# Simple styling - soft cards, rounded corners, calm color palette
# -----------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #F7F9FC; }
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        border: 1px solid #EEF1F6;
    }
    .metric-label {
        color: #8A94A6;
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        margin-top: 4px;
    }
    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 14px;
    }
    .risk-low { background: #E6F7EE; color: #1B9E5A; }
    .risk-medium { background: #FFF4E0; color: #C97F00; }
    .risk-high { background: #FDE8E8; color: #D92D2D; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------
# Load data
# -----------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["received_date"])
    return df

df = load_data()

# -----------------------------------------------------------------
# Sidebar - patient selector
# -----------------------------------------------------------------
with st.sidebar:
    st.markdown("## 💙 Long COVID Monitor")
    st.caption("Personalized digital phenotyping dashboard")
    st.markdown("---")
    patient_ids = sorted(df["user_id"].unique())
    selected_patient = st.selectbox("Select patient", patient_ids)
    st.markdown("---")
    st.caption("This tool supports clinical decisions. "
               "It is not an autonomous diagnostic system.")

patient_df = df[df["user_id"] == selected_patient].sort_values("received_date")
latest = patient_df.iloc[-1]

# -----------------------------------------------------------------
# Header
# -----------------------------------------------------------------
st.markdown(f"# Patient #{selected_patient}")
st.caption(f"Last update: {latest['received_date']}")

# -----------------------------------------------------------------
# Top row: key metrics
# -----------------------------------------------------------------
risk_score = latest["risk_score"]
uncertainty = latest["uncertainty"]

if risk_score >= RISK_HIGH:
    risk_label, risk_class = "High risk", "risk-high"
elif risk_score >= RISK_MEDIUM:
    risk_label, risk_class = "Medium risk", "risk-medium"
else:
    risk_label, risk_class = "Low risk", "risk-low"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Risk (next 12h)</div>
        <div class="metric-value">{risk_score*100:.0f}%</div>
        <span class="risk-badge {risk_class}">{risk_label}</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Model confidence</div>
        <div class="metric-value">{(1-uncertainty)*100:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Heart Rate</div>
        <div class="metric-value">{latest['heart_rate']:.0f} <span style="font-size:16px;color:#8A94A6;">bpm</span></div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">SpO2</div>
        <div class="metric-value">{latest['spo2']:.0f}<span style="font-size:16px;color:#8A94A6;">%</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if risk_score >= RISK_HIGH:
    st.error("⚠️ Elevated risk detected. Consider reducing activity and "
             "monitoring symptoms closely over the next 12 hours.")

# -----------------------------------------------------------------
# Risk trend chart
# -----------------------------------------------------------------
st.markdown("### Risk score over time")
fig_risk = go.Figure()
fig_risk.add_trace(go.Scatter(
    x=patient_df["received_date"], y=patient_df["risk_score"],
    mode="lines", fill="tozeroy",
    line=dict(color="#5B7FFF", width=2),
    fillcolor="rgba(91,127,255,0.1)",
    name="Risk score",
))
fig_risk.add_hline(y=RISK_HIGH, line_dash="dot", line_color="#D92D2D",
                    annotation_text="High risk threshold")
fig_risk.update_layout(
    height=280, margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="white", paper_bgcolor="white",
    yaxis=dict(range=[0, 1], title="Risk"),
    xaxis=dict(title=""),
)
st.plotly_chart(fig_risk, use_container_width=True)

# -----------------------------------------------------------------
# Raw signals chart
# -----------------------------------------------------------------
st.markdown("### Vitals over time")
signal_choice = st.multiselect(
    "Signals to display",
    ["heart_rate", "spo2", "body_battery", "heart_rate_variability", "steps"],
    default=["heart_rate", "body_battery"],
)

if signal_choice:
    fig_signals = go.Figure()
    colors = ["#5B7FFF", "#1B9E5A", "#C97F00", "#D92D2D", "#8A5CF6"]
    for i, sig in enumerate(signal_choice):
        fig_signals.add_trace(go.Scatter(
            x=patient_df["received_date"], y=patient_df[sig],
            mode="lines", name=sig, line=dict(color=colors[i % len(colors)], width=1.8),
        ))
    fig_signals.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_signals, use_container_width=True)

# -----------------------------------------------------------------
# Share with doctor (simple demo version)
# -----------------------------------------------------------------
with st.expander("📤 Share with doctor"):
    st.write(f"**Patient:** #{selected_patient}")
    st.write(f"**Current risk (next 12h):** {risk_score*100:.0f}% ({risk_label})")
    st.write(f"**Model confidence:** {(1-uncertainty)*100:.0f}%")
    st.write(f"**Last heart rate:** {latest['heart_rate']:.0f} bpm")
    st.write(f"**Last SpO2:** {latest['spo2']:.0f}%")
    st.caption("In a full version, this would generate a secure link or "
               "PDF for the treating physician.")
