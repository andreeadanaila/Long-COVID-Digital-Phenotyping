import os
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from fpdf import FPDF
from io import BytesIO

# -----------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------
DATA_PATH = "dashboard_data.csv"
ALERTS_LOG_PATH = "alerts_log.csv"
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
# Alert logging - appends a row to alerts_log.csv whenever the
# current risk score for a patient crosses RISK_HIGH, so we can
# later show "caught X alerts, Y hours before self-report"
# -----------------------------------------------------------------
def log_alert_if_high_risk(patient_id, latest_row):
    if latest_row["risk_score"] < RISK_HIGH:
        return
    alert_row = pd.DataFrame([{
        "user_id": patient_id,
        "alert_time": latest_row["received_date"],
        "risk_score": latest_row["risk_score"],
        "uncertainty": latest_row["uncertainty"],
    }])
    if os.path.exists(ALERTS_LOG_PATH):
        existing = pd.read_csv(ALERTS_LOG_PATH, parse_dates=["alert_time"])
        already_logged = (
            (existing["user_id"] == patient_id)
            & (existing["alert_time"] == latest_row["received_date"])
        ).any()
        if already_logged:
            return
        alert_row.to_csv(ALERTS_LOG_PATH, mode="a", header=False, index=False)
    else:
        alert_row.to_csv(ALERTS_LOG_PATH, mode="w", header=True, index=False)


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
    view_mode = st.radio("View", ["Patient view", "Doctor view"], horizontal=True)
    is_doctor_view = view_mode == "Doctor view"
    st.markdown("---")
    st.caption("This tool supports clinical decisions. "
               "It is not an autonomous diagnostic system.")

patient_df = df[df["user_id"] == selected_patient].sort_values("received_date")

# Guard: no data for this patient
if patient_df.empty:
    st.warning(f"No data available for patient #{selected_patient}.")
    st.stop()

# Guard: required columns missing
required_cols = ["risk_score", "uncertainty", "heart_rate", "spo2"]
missing_cols = [c for c in required_cols if c not in patient_df.columns]
if missing_cols:
    st.warning(f"Missing expected columns for patient #{selected_patient}: {', '.join(missing_cols)}")
    st.stop()

latest = patient_df.iloc[-1]

# Guard: latest row has NaN in required fields
if latest[required_cols].isna().any():
    st.warning(f"Latest reading for patient #{selected_patient} is incomplete, showing last available data.")
    patient_df_complete = patient_df.dropna(subset=required_cols)
    if patient_df_complete.empty:
        st.stop()
    latest = patient_df_complete.iloc[-1]

# -----------------------------------------------------------------
# Date range filter - with 10k+ rows per patient, an all-time chart
# is unreadable, so default to the last 7 days
# -----------------------------------------------------------------
min_date = patient_df["received_date"].min()
max_date = patient_df["received_date"].max()
default_start = max(min_date, max_date - pd.Timedelta(days=7))

date_range = st.sidebar.date_input(
    "Date range",
    value=(default_start.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (patient_df["received_date"].dt.date >= start_date) & (patient_df["received_date"].dt.date <= end_date)
    patient_df_view = patient_df[mask]
else:
    patient_df_view = patient_df

if patient_df_view.empty:
    st.warning("No data in the selected date range.")
    st.stop()

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

log_alert_if_high_risk(selected_patient, latest)

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

all_signals = ["heart_rate", "spo2", "body_battery", "heart_rate_variability", "steps"]

# -----------------------------------------------------------------
# Doctor view only - alert history and raw signal stats
# -----------------------------------------------------------------
if is_doctor_view:
    st.markdown("### Alert history")
    if os.path.exists(ALERTS_LOG_PATH):
        alerts_log = pd.read_csv(ALERTS_LOG_PATH, parse_dates=["alert_time"])
        patient_alerts = alerts_log[alerts_log["user_id"] == selected_patient].sort_values(
            "alert_time", ascending=False
        )
        if patient_alerts.empty:
            st.caption("No high-risk episodes detected in the monitored period for this patient.")
        else:
            st.dataframe(patient_alerts, use_container_width=True, hide_index=True)
    else:
        st.caption("No high-risk episodes detected in the monitored period for this patient.")

    st.markdown("### Signal stats (personal baseline)")
    stats_rows = []
    for sig in all_signals:
        if sig in patient_df.columns:
            stats_rows.append({
                "signal": sig,
                "mean": round(patient_df[sig].mean(), 2),
                "std": round(patient_df[sig].std(), 2),
                "last value": round(patient_df[sig].iloc[-1], 2),
            })
    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)
    st.caption(f"Model confidence (1 - uncertainty): {(1-uncertainty)*100:.1f}%  |  "
               f"raw uncertainty: {uncertainty:.3f}")
    st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------
# Risk trend chart - with uncertainty band (risk +/- uncertainty)
# -----------------------------------------------------------------
st.markdown("### Risk score over time")

if "uncertainty" in patient_df_view.columns:
    upper = (patient_df_view["risk_score"] + patient_df_view["uncertainty"]).clip(upper=1)
    lower = (patient_df_view["risk_score"] - patient_df_view["uncertainty"]).clip(lower=0)
else:
    upper = lower = None

fig_risk = go.Figure()

if upper is not None:
    fig_risk.add_trace(go.Scatter(
        x=pd.concat([patient_df_view["received_date"], patient_df_view["received_date"][::-1]]),
        y=pd.concat([upper, lower[::-1]]),
        fill="toself",
        fillcolor="rgba(91,127,255,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False,
        name="Uncertainty band",
    ))

fig_risk.add_trace(go.Scatter(
    x=patient_df_view["received_date"], y=patient_df_view["risk_score"],
    mode="lines", fill="tozeroy",
    line=dict(color="#5B7FFF", width=2),
    fillcolor="rgba(91,127,255,0.1)",
    name="Risk score",
))
fig_risk.add_hline(y=RISK_HIGH, line_dash="dot", line_color="#D92D2D",
                    annotation_text="High risk threshold",
                    annotation_font_color="#1A1D29")

# device_not_worn shading - flag intervals where the wearable wasn't
# on so the chart doesn't read as if all data were real measurements
if "device_not_worn" in patient_df_view.columns:
    not_worn = patient_df_view[patient_df_view["device_not_worn"] == True]
    if not not_worn.empty:
        gaps = (not_worn["received_date"].diff() > pd.Timedelta(minutes=1)).cumsum()
        for _, seg in not_worn.groupby(gaps):
            fig_risk.add_vrect(
                x0=seg["received_date"].min(), x1=seg["received_date"].max(),
                fillcolor="rgba(138,148,166,0.2)", line_width=0,
                annotation_text="device not worn", annotation_position="top left",
                annotation_font_size=10, annotation_font_color="#8A94A6",
            )

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
# Raw signals chart - with personal baseline band and optional
# normalization when signals on very different scales are compared
# -----------------------------------------------------------------
st.markdown("### Vitals over time")

col_a, col_b = st.columns([3, 1])
with col_a:
    signal_choice = st.multiselect(
        "Signals to display",
        all_signals,
        default=all_signals if is_doctor_view else ["heart_rate", "body_battery"],
    )
with col_b:
    normalize = st.checkbox("Normalize (0-1)", value=len(signal_choice) > 2)

if signal_choice:
    fig_signals = go.Figure()
    colors = ["#5B7FFF", "#1B9E5A", "#C97F00", "#D92D2D", "#8A5CF6"]

    # personal baseline: mean/std computed from this patient's full history,
    # not just the filtered view, so it reflects their stable range
    baseline_stats = {
        sig: (patient_df[sig].mean(), patient_df[sig].std())
        for sig in signal_choice
    }

    for i, sig in enumerate(signal_choice):
        series = patient_df_view[sig]
        mean_val, std_val = baseline_stats[sig]

        if normalize:
            sig_min, sig_max = patient_df[sig].min(), patient_df[sig].max()
            span = (sig_max - sig_min) or 1
            plot_series = (series - sig_min) / span
            band_low = (mean_val - std_val - sig_min) / span
            band_high = (mean_val + std_val - sig_min) / span
        else:
            plot_series = series
            band_low = mean_val - std_val
            band_high = mean_val + std_val

        color = colors[i % len(colors)]

        # personal baseline band (mean +/- std)
        fig_signals.add_trace(go.Scatter(
            x=pd.concat([patient_df_view["received_date"], patient_df_view["received_date"][::-1]]),
            y=pd.Series([band_high] * len(patient_df_view) + [band_low] * len(patient_df_view)),
            fill="toself",
            fillcolor="rgba(0,0,0,0.04)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ))

        fig_signals.add_trace(go.Scatter(
            x=patient_df_view["received_date"], y=plot_series,
            mode="lines", name=sig, line=dict(color=color, width=1.8),
        ))

    fig_signals.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="#1A1D29"),
        xaxis=dict(color="#1A1D29", tickfont=dict(color="#1A1D29")),
        yaxis=dict(color="#1A1D29", tickfont=dict(color="#1A1D29"),
                   title="Normalized (0-1)" if normalize else ""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    font=dict(color="#1A1D29")),
    )
    st.plotly_chart(fig_signals, use_container_width=True)
    st.caption("Shaded band = this patient's personal baseline range (mean ± 1 std). "
               "Values outside the band are unusual for them specifically, "
               "not against a general population threshold.")

# -----------------------------------------------------------------
# Share with doctor (simple demo version) - patient-facing action,
# doesn't make sense to show when the doctor is already viewing
# -----------------------------------------------------------------
if not is_doctor_view:
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