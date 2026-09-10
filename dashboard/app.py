import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="ATO Anomaly Detection", page_icon="🚨", layout="wide")

FEATURES = Path("data/processed/session_features.csv")
SCORES = Path("reports/combined_scores.csv")

st.title("🚨 Account Takeover Behaviour Anomaly Detection")
st.caption("Synthetic banking-session investigation dashboard")

if not FEATURES.exists():
    st.error("Run the data and training pipeline first.")
    st.code("""
python src/data_generation.py
python src/sessionization.py
python src/feature_engineering.py
python src/train_isolation_forest.py
python src/train_autoencoder.py
python src/train_lstm_autoencoder.py
python src/evaluate.py
""")
    st.stop()

df = pd.read_csv(FEATURES)

if SCORES.exists():
    scores = pd.read_csv(SCORES)
    df = df.merge(scores, on=["session_id", "user_id", "label", "attack_type"], how="left")

model_cols = [c for c in ["Isolation Forest", "Autoencoder", "LSTM Autoencoder"] if c in df.columns]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sessions", f"{len(df):,}")
c2.metric("Known anomalous sessions", f"{int(df.label.sum()):,}")
c3.metric("Anomaly rate", f"{df.label.mean()*100:.1f}%")
c4.metric("Models available", len(model_cols))

st.subheader("Session risk distribution")
if model_cols:
    selected_model = st.selectbox("Scoring model", model_cols)
    fig = px.histogram(df, x=selected_model, color=df["label"].map({0: "Normal", 1: "Anomalous"}),
                       nbins=40, barmode="overlay",
                       labels={"color": "Ground truth"})
    st.plotly_chart(fig, width="stretch")

st.subheader("Investigate a session")
scored_df = df.dropna(subset=model_cols, how="all")

sid = st.selectbox("Session ID", scored_df.session_id.tolist())
row = scored_df[scored_df.session_id == sid].iloc[0]

left, right = st.columns(2)
with left:
    st.write("### Session details")
    st.write({
        "User": row.user_id,
        "Ground truth": "Anomalous" if row.label else "Normal",
        "Attack type": row.attack_type,
        "Events": int(row.event_count),
        "Transfers": int(row.transfer_count),
        "Failed logins": int(row.failed_logins),
        "New device": int(row.new_device),
        "New location": int(row.new_location),
        "Impossible travel": int(row.impossible_travel),
    })

with right:
    st.write("### Model scores")
    for col in model_cols:
        st.metric(col.replace("_score", "").replace("_", " ").title(),
                  f"{row[col]:.4f}")

st.write("### Behavioural explanation")
reasons = []
if row.new_device:
    reasons.append("⚠ New/unseen device")
if row.new_location:
    reasons.append("⚠ New geographic location")
if row.impossible_travel:
    reasons.append("⚠ Impossible-travel pattern")
if row.failed_logins >= 2:
    reasons.append(f"⚠ {int(row.failed_logins)} failed logins")
if row.beneficiary_adds:
    reasons.append("⚠ New beneficiary")
if row.password_changes:
    reasons.append("⚠ Password change")
if row.security_changes:
    reasons.append("⚠ Security settings changed")
if row.transaction_velocity_per_min > 0.12:
    reasons.append("⚠ High transaction velocity")
if row.max_transfer_amount > 25000:
    reasons.append("⚠ Unusually large transfer")

if reasons:
    for reason in reasons:
        st.write(reason)
else:
    st.success("No major rule-based behavioural deviations identified.")

st.info(
    "The explanation layer identifies behavioural signals associated with the "
    "session. It should be interpreted as investigation support, not proof of fraud."
)
