import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from fpdf import FPDF
from io import BytesIO

# -----------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------
DATA_PATH = "dashboard_data.csv"
RISK_HIGH = 0.7   # above this = red
RISK_MEDIUM = 0.4  # above this = yellow, below = green

st.set_page_config(page_title="Long COVID Monitoring", layout="wide", page_icon="💙")

# -----------------------------------------------------------------
# Simple styling - soft cards, rounded corners, calm color palette
# -----------------------------------------------------------------
st.markdown("""
<style>
    .stApp, .main { background-color: var(--background-color); }
    .metric-card {
        background: white;
        color: #1A1D29;
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        border: 1px solid #EEF1F6;
        height: 160px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        gap: 8px;
    }
    .info-card {
        background: white;
        color: #1A1D29;
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        border: 1px solid #EEF1F6;
        text-align: left;
    }
    .metric-label {
        color: #8A94A6;
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .metric-value {
        color: #1A1D29;
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
        width: fit-content;
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
# PDF report generator - builds a one-page summary in memory,
# no file is saved to disk, it's streamed straight to the browser.
# -----------------------------------------------------------------
def generate_patient_pdf(patient_id, latest_row, risk_label, main_factors):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Long COVID Monitor - Patient Report", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 8, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"Patient #{patient_id}", ln=True)

    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Last update: {latest_row['received_date']}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Risk (next 12h): {latest_row['risk_score']*100:.0f}%  ({risk_label})", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Model confidence: {(1 - latest_row['uncertainty'])*100:.0f}%", ln=True)
    pdf.cell(0, 8, f"Heart rate: {latest_row['heart_rate']:.0f} bpm", ln=True)
    pdf.cell(0, 8, f"SpO2: {latest_row['spo2']:.0f}%", ln=True)
    pdf.ln(4)

    if main_factors:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Main contributing signals:", ln=True)
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 8, ", ".join(main_factors), ln=True)
        pdf.ln(4)

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 6, "This tool supports clinical decisions. "
                         "It is not an autonomous diagnostic system.")

    # fpdf2 returns a bytearray - BytesIO wraps it for st.download_button
    return BytesIO(pdf.output())


# -----------------------------------------------------------------
# Sidebar - patient selector
# -----------------------------------------------------------------
with st.sidebar:
    st.markdown("## Long COVID Monitor")
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
    st.error(" Elevated risk detected. Consider reducing activity and "
             "monitoring symptoms closely over the next 12 hours.")

# -----------------------------------------------------------------
# NEW - Why this score (explainability)
# Shows which signals are driving the current prediction, so this
# reads as a decision-support tool rather than an unexplained number.
# -----------------------------------------------------------------
factor_cols = ["top_factor_1", "top_factor_2", "top_factor_3"]
if all(col in latest.index for col in factor_cols):
    st.markdown("### Why this score?")
    factors = [latest[c] for c in factor_cols if pd.notna(latest[c])]
    if factors:
        badges = " ".join(
            f'<span class="risk-badge risk-medium" style="margin-right:8px;">{f}</span>'
            for f in factors
        )
        st.markdown(
            f'<div class="info-card">This risk estimate is mainly driven by: '
            f'{badges}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Ranked by how unusual each signal is for this patient right now, "
                   "weighted by how much the model relies on that signal overall.")
    st.markdown("<br>", unsafe_allow_html=True)

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
                    annotation_text="High risk threshold",
                    annotation_font_color="#1A1D29")
fig_risk.update_layout(
    height=280, margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(color="#1A1D29"),
    yaxis=dict(range=[0, 1], title="Risk", color="#1A1D29",
               tickfont=dict(color="#1A1D29")),
    xaxis=dict(title="", color="#1A1D29",
               tickfont=dict(color="#1A1D29")),
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
        font=dict(color="#1A1D29"),
        xaxis=dict(color="#1A1D29", tickfont=dict(color="#1A1D29")),
        yaxis=dict(color="#1A1D29", tickfont=dict(color="#1A1D29")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    font=dict(color="#1A1D29")),
    )
    st.plotly_chart(fig_signals, use_container_width=True)

# -----------------------------------------------------------------
# Share with doctor (simple demo version)
# -----------------------------------------------------------------
with st.expander(" Share with doctor"):
    st.write(f"**Patient:** #{selected_patient}")
    st.write(f"**Current risk (next 12h):** {risk_score*100:.0f}% ({risk_label})")
    st.write(f"**Model confidence:** {(1-uncertainty)*100:.0f}%")
    main_factors = []
    if all(col in latest.index for col in factor_cols):
        main_factors = [latest[c] for c in factor_cols if pd.notna(latest[c])]
        if main_factors:
            st.write(f"**Main contributing signals:** {', '.join(main_factors)}")
    st.write(f"**Last heart rate:** {latest['heart_rate']:.0f} bpm")
    st.write(f"**Last SpO2:** {latest['spo2']:.0f}%")

    pdf_bytes = generate_patient_pdf(selected_patient, latest, risk_label, main_factors)
    st.download_button(
        label="Download PDF report",
        data=pdf_bytes,
        file_name=f"patient_{selected_patient}_report.pdf",
        mime="application/pdf",
    )
    st.caption("In a full version, this would also generate a secure "
               "shareable link for the treating physician.")