from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Project setup
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import InferenceEngine


FEATURES_PATH = ROOT / "data" / "processed" / "session_features.csv"
SCORES_PATH = ROOT / "reports" / "combined_scores.csv"
SEQUENCES_PATH = ROOT / "data" / "processed" / "sequences.npz"
SEQUENCE_META_PATH = ROOT / "data" / "processed" / "sequence_meta.json"


# ------------------------------------------------------------
# Page
# ------------------------------------------------------------

st.set_page_config(
    page_title="ATO Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: #080c11;
    }

    [data-testid="stSidebar"] {
        background: #0b1016;
        border-right: 1px solid #202a35;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    .hero {
        border: 1px solid #26323f;
        border-radius: 16px;
        padding: 1.35rem 1.5rem;
        background: linear-gradient(135deg, #111923, #0b1016);
        margin-bottom: 1.1rem;
    }

    .eyebrow {
        color: #708191;
        text-transform: uppercase;
        letter-spacing: .14em;
        font-size: .70rem;
        font-weight: 800;
    }

    .hero-title {
        font-size: 2.25rem;
        font-weight: 850;
        letter-spacing: -.045em;
        margin-top: .1rem;
    }

    .hero-subtitle {
        color: #91a0af;
        margin-top: .2rem;
    }

    .panel {
        border: 1px solid #26323f;
        border-radius: 13px;
        padding: 1rem;
        background: #0e151d;
    }

    .risk {
        border-radius: 14px;
        padding: 1.25rem;
        text-align: center;
        margin: .35rem 0 1rem;
    }

    .risk-high {
        background: #230f12;
        border: 1px solid #71343b;
    }

    .risk-suspicious {
        background: #211b0b;
        border: 1px solid #6c5a20;
    }

    .risk-normal {
        background: #0d1d14;
        border: 1px solid #28583e;
    }

    .risk-title {
        font-size: 1.8rem;
        font-weight: 850;
    }

    .model-box {
        border: 1px solid #26323f;
        background: #0e151d;
        border-radius: 12px;
        padding: 1rem;
        min-height: 158px;
    }

    .model-name {
        font-weight: 750;
        font-size: .95rem;
    }

    .model-score {
        font-size: 1.45rem;
        font-weight: 850;
        margin-top: .45rem;
    }

    .muted {
        color: #81909e;
        font-size: .82rem;
    }

    .event {
        border-left: 3px solid #3a8a58;
        background: #0e151d;
        border-radius: 0 9px 9px 0;
        padding: .65rem .8rem;
        margin: .35rem 0;
    }

    .event-risk {
        border-left-color: #d9534f;
    }

    .event-warning {
        border-left-color: #d2a52e;
    }

    .signal {
        border: 1px solid #283440;
        background: #0e151d;
        border-radius: 9px;
        padding: .65rem .8rem;
        margin: .35rem 0;
    }

    .scenario-card {
        border: 1px solid #26323f;
        border-radius: 12px;
        padding: .9rem 1rem;
        background: #0e151d;
        margin-bottom: .8rem;
    }

    div[data-testid="stMetric"] {
        background: #0e151d;
        border: 1px solid #26323f;
        padding: .65rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Data / model loading
# ------------------------------------------------------------

@st.cache_resource
def load_engine():
    return InferenceEngine()


@st.cache_data
def load_data():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FEATURES_PATH}. Run the data/feature pipeline first."
        )

    features = pd.read_csv(FEATURES_PATH)

    scores = None
    if SCORES_PATH.exists():
        scores = pd.read_csv(SCORES_PATH)

    sequence_events = {}

    if SEQUENCES_PATH.exists() and SEQUENCE_META_PATH.exists():
        z = np.load(SEQUENCES_PATH)
        sequences = z["sequences"]

        with open(SEQUENCE_META_PATH, "r") as f:
            meta = json.load(f)

        id_to_event = {
            int(k): v for k, v in meta["id_to_event"].items()
        }

        session_ids = meta.get("session_ids", [])

        for i, sid in enumerate(session_ids):
            if i >= len(sequences):
                break

            sequence_events[str(sid)] = [
                id_to_event[int(event_id)]
                for event_id in sequences[i]
                if int(event_id) != 0
            ]

    return features, scores, sequence_events


try:
    engine = load_engine()
    df, scores, sequence_events = load_data()
except Exception as exc:
    st.error("ATO Sentinel could not initialize.")
    st.exception(exc)
    st.stop()


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Security Analytics Platform</div>
        <div class="hero-title">🛡️ ATO SENTINEL</div>
        <div class="hero-subtitle">
            Account Takeover & Suspicious Session Behaviour Detection
            · Synthetic banking environment · Live ML inference
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Shared definitions
# ------------------------------------------------------------

EVENT_OPTIONS = [
    "LOGIN",
    "FAILED_LOGIN",
    "ACCOUNT_VIEW",
    "BALANCE_CHECK",
    "BENEFICIARY_VIEW",
    "BENEFICIARY_ADD",
    "PASSWORD_CHANGE",
    "SECURITY_CHANGE",
    "DEVICE_REGISTER",
    "TRANSFER",
    "LOGOUT",
]

DANGEROUS_EVENTS = {
    "FAILED_LOGIN",
    "BENEFICIARY_ADD",
    "PASSWORD_CHANGE",
    "SECURITY_CHANGE",
    "DEVICE_REGISTER",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def render_events(events):
    descriptions = {
        "LOGIN": ("🔐", "Authentication successful"),
        "FAILED_LOGIN": ("🔴", "Authentication failure"),
        "ACCOUNT_VIEW": ("✓", "Account accessed"),
        "BALANCE_CHECK": ("✓", "Balance viewed"),
        "BENEFICIARY_VIEW": ("✓", "Beneficiary list viewed"),
        "BENEFICIARY_ADD": ("⚠️", "New beneficiary created"),
        "PASSWORD_CHANGE": ("⚠️", "Credential modification"),
        "SECURITY_CHANGE": ("⚠️", "Security configuration modified"),
        "DEVICE_REGISTER": ("⚠️", "New device registered"),
        "TRANSFER": ("💸", "Funds transfer initiated"),
        "LOGOUT": ("↪️", "Session terminated"),
    }

    for i, event in enumerate(events, 1):
        icon, description = descriptions.get(
            event,
            ("•", "Account activity"),
        )

        if event in DANGEROUS_EVENTS:
            cls = "event event-risk"
        elif event == "TRANSFER":
            cls = "event event-warning"
        else:
            cls = "event"

        st.markdown(
            f"""
            <div class="{cls}">
                <strong>{i:02d}</strong>
                &nbsp;&nbsp; {icon} <strong>{event}</strong>
                <div class="muted" style="margin-left:2.2rem;">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_model_results(result):
    models = [
        ("Isolation Forest", result["Isolation Forest"]),
        ("Dense Autoencoder", result["Autoencoder"]),
        ("LSTM Autoencoder", result["LSTM Autoencoder"]),
    ]

    columns = st.columns(3)

    for column, (name, data) in zip(columns, models):
        with column:
            status = "🚨 ANOMALY" if data["is_anomaly"] else "✓ NORMAL"
            score = data["score"]
            threshold = data["threshold"]
            ratio = score / threshold if threshold else 0.0

            st.markdown(
                f"""
                <div class="model-box">
                    <div class="model-name">{name}</div>
                    <div class="model-score">{score:.4f}</div>
                    <div class="muted">
                        Learned threshold: {threshold:.4f}
                    </div>
                    <br>
                    <strong>{status}</strong>
                    <div class="muted">
                        Score / threshold: {ratio:.2f}×
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(
                min(max(ratio / 2.0, 0.0), 1.0),
                text=f"{ratio:.2f}× threshold",
            )


def render_verdict(result):
    verdict = result["verdict"]
    votes = result["votes"]

    if verdict == "HIGH RISK":
        css = "risk risk-high"
        icon = "🔴"
        description = "Potential account takeover behaviour detected"
        action = "INVESTIGATE"
    elif verdict == "SUSPICIOUS":
        css = "risk risk-suspicious"
        icon = "🟠"
        description = "Anomalous behaviour requires investigation"
        action = "REVIEW"
    else:
        css = "risk risk-normal"
        icon = "🟢"
        description = "No significant anomaly detected"
        action = "ALLOW"

    st.markdown(
        f"""
        <div class="{css}">
            <div class="risk-title">{icon} {verdict}</div>
            <div>{description}</div>
            <div class="muted">Model consensus: {votes} / 3</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3)

    with a:
        st.metric("Model Consensus", f"{votes} / 3")
    with b:
        st.metric("Models Flagging", votes)
    with c:
        st.metric("Recommended Action", action)


def behavioural_signals(events, flags=None, row=None):
    signals = []

    flags = flags or {}

    if flags.get("new_device") or (row is not None and row.get("new_device", 0) == 1):
        signals.append("📱 New device detected")

    if flags.get("new_location") or (row is not None and row.get("new_location", 0) == 1):
        signals.append("🌍 New geographic location")

    if flags.get("impossible_travel") or (
        row is not None and row.get("impossible_travel", 0) == 1
    ):
        signals.append("✈️ Impossible travel pattern")

    failed = events.count("FAILED_LOGIN")
    if failed >= 2 or (row is not None and row.get("failed_logins", 0) >= 2):
        count = failed if failed else int(row["failed_logins"])
        signals.append(f"🔐 {count} failed login attempts")

    if "PASSWORD_CHANGE" in events or (
        row is not None and row.get("password_changes", 0) >= 1
    ):
        signals.append("🔑 Password changed during session")

    if "SECURITY_CHANGE" in events or (
        row is not None and row.get("security_changes", 0) >= 1
    ):
        signals.append("🛡️ Security settings changed")

    if "DEVICE_REGISTER" in events or (
        row is not None and row.get("device_registers", 0) >= 1
    ):
        signals.append("📱 Device registration occurred")

    if "BENEFICIARY_ADD" in events or (
        row is not None and row.get("beneficiary_adds", 0) >= 1
    ):
        signals.append("👤 New beneficiary added")

    if flags.get("large_transfer") or (
        row is not None and row.get("amount_zscore", 0) >= 3
    ):
        signals.append("💸 Unusually large transfer")

    if flags.get("transfer_velocity"):
        signals.append("⚡ High transaction velocity")

    return signals


def make_simulated_features(events, flags):
    """
    Use a real normal feature row as the schema/template, then
    replace behavioural values with the current synthetic scenario.
    This guarantees that inference receives the exact feature names
    expected by the trained models.
    """
    normal_rows = df[df["label"] == 0]

    if normal_rows.empty:
        base = df.iloc[[0]].copy()
    else:
        base = normal_rows.iloc[[0]].copy()

    event_count = len(events)
    failed = events.count("FAILED_LOGIN")
    beneficiary_adds = events.count("BENEFICIARY_ADD")
    password_changes = events.count("PASSWORD_CHANGE")
    security_changes = events.count("SECURITY_CHANGE")
    device_registers = events.count("DEVICE_REGISTER")
    transfer_count = events.count("TRANSFER")

    transfer_amount = 10000.0 if flags["large_transfer"] else 500.0
    transfer_sum = transfer_count * transfer_amount
    transfer_max = transfer_amount if transfer_count else 0.0
    transfer_mean = transfer_amount if transfer_count else 0.0

    duration_sec = max(60.0, event_count * 45.0)

    values = {
        "event_count": event_count,
        "session_duration_sec": duration_sec,
        "failed_logins": failed,
        "beneficiary_adds": beneficiary_adds,
        "password_changes": password_changes,
        "security_changes": security_changes,
        "device_registers": device_registers,
        "transfer_count": transfer_count,
        "total_transfer_amount": transfer_sum,
        "max_transfer_amount": transfer_max,
        "avg_transfer_amount": transfer_mean,
        "transaction_velocity_per_min": (
            transfer_count / max(duration_sec / 60.0, 1.0)
        ),
        "new_device": int(flags["new_device"]),
        "new_location": int(flags["new_location"]),
        "time_since_prev_session_hr": (
            0.25 if flags["impossible_travel"] else 1.0
        ),
        "impossible_travel": int(flags["impossible_travel"]),
        "amount_zscore": 4.0 if flags["large_transfer"] else 0.0,
        "login_hour": 2.0 if flags["transfer_velocity"] else 14.0,
    }

    for column, value in values.items():
        if column in base.columns:
            base.loc[base.index[0], column] = value

    return base


def performance_table():
    return pd.DataFrame(
        [
            {
                "Model": "Isolation Forest",
                "PR-AUC": 0.848022,
                "Precision": 0.690909,
                "Recall": 0.867580,
                "F1": 0.769231,
                "Recall @ 1% FPR": 0.552511,
            },
            {
                "Model": "Dense Autoencoder",
                "PR-AUC": 0.971104,
                "Precision": 0.927928,
                "Recall": 0.940639,
                "F1": 0.934240,
                "Recall @ 1% FPR": 0.963470,
            },
            {
                "Model": "LSTM Autoencoder",
                "PR-AUC": 0.763862,
                "Precision": 1.000000,
                "Recall": 0.607306,
                "F1": 0.755682,
                "Recall @ 1% FPR": 0.657534,
            },
        ]
    )


# ------------------------------------------------------------
# Navigation
# ------------------------------------------------------------

mode = st.radio(
    "Operating mode",
    [
        "🧪 Live Simulator",
        "🔍 Session Investigation",
        "📊 Model Performance",
    ],
    horizontal=True,
    label_visibility="collapsed",
)


# ============================================================
# LIVE SIMULATOR
# ============================================================

if mode == "🧪 Live Simulator":

    SCENARIOS = {
        "🟢 Normal Banking Session": {
            "events": [
                "LOGIN",
                "ACCOUNT_VIEW",
                "BALANCE_CHECK",
                "TRANSFER",
                "LOGOUT",
            ],
            "flags": {
                "new_device": False,
                "new_location": False,
                "impossible_travel": False,
                "large_transfer": False,
                "transfer_velocity": False,
            },
            "description": "Typical account activity within an expected session pattern.",
        },
        "🟡 New Device Login": {
            "events": [
                "LOGIN",
                "ACCOUNT_VIEW",
                "BALANCE_CHECK",
                "LOGOUT",
            ],
            "flags": {
                "new_device": True,
                "new_location": False,
                "impossible_travel": False,
                "large_transfer": False,
                "transfer_velocity": False,
            },
            "description": "A legitimate-looking session originating from an unfamiliar device.",
        },
        "🟠 Credential Attack": {
            "events": [
                "FAILED_LOGIN",
                "FAILED_LOGIN",
                "FAILED_LOGIN",
                "LOGIN",
                "ACCOUNT_VIEW",
                "BALANCE_CHECK",
                "LOGOUT",
            ],
            "flags": {
                "new_device": False,
                "new_location": False,
                "impossible_travel": False,
                "large_transfer": False,
                "transfer_velocity": False,
            },
            "description": "Repeated authentication failures followed by a successful login.",
        },
        "🔴 Account Takeover": {
            "events": [
                "FAILED_LOGIN",
                "FAILED_LOGIN",
                "FAILED_LOGIN",
                "LOGIN",
                "PASSWORD_CHANGE",
                "SECURITY_CHANGE",
                "DEVICE_REGISTER",
                "BENEFICIARY_ADD",
                "LOGOUT",
            ],
            "flags": {
                "new_device": True,
                "new_location": True,
                "impossible_travel": True,
                "large_transfer": False,
                "transfer_velocity": False,
            },
            "description": "Credential compromise followed by account and security changes.",
        },
        "🔴 ATO + Fund Transfer": {
            "events": [
                "FAILED_LOGIN",
                "FAILED_LOGIN",
                "FAILED_LOGIN",
                "LOGIN",
                "PASSWORD_CHANGE",
                "SECURITY_CHANGE",
                "DEVICE_REGISTER",
                "BENEFICIARY_ADD",
                "TRANSFER",
                "LOGOUT",
            ],
            "flags": {
                "new_device": True,
                "new_location": True,
                "impossible_travel": True,
                "large_transfer": True,
                "transfer_velocity": True,
            },
            "description": "Full takeover pattern followed by suspicious fund movement.",
        },
    }

    if "sim_events" not in st.session_state:
        st.session_state.sim_events = [
            "LOGIN",
            "ACCOUNT_VIEW",
            "BALANCE_CHECK",
        ]

    if "sim_flags" not in st.session_state:
        st.session_state.sim_flags = {
            "new_device": False,
            "new_location": False,
            "impossible_travel": False,
            "large_transfer": False,
            "transfer_velocity": False,
        }

    if "sim_name" not in st.session_state:
        st.session_state.sim_name = "Custom Session"

    if "sim_result" not in st.session_state:
        st.session_state.sim_result = None

    with st.sidebar:
        st.markdown("### 🎯 Scenario Simulator")
        st.caption(
            "Generate synthetic banking behaviour and send it through "
            "the trained anomaly-detection models."
        )

        selected_scenario = st.selectbox(
            "Scenario",
            list(SCENARIOS.keys()),
        )

        if st.button(
            "⚡ LOAD SCENARIO",
            type="primary",
            use_container_width=True,
        ):
            selected = SCENARIOS[selected_scenario]
            st.session_state.sim_events = selected["events"].copy()
            st.session_state.sim_flags = selected["flags"].copy()
            st.session_state.sim_name = selected_scenario
            st.session_state.sim_result = None

        st.divider()
        st.markdown("### 🔧 Custom Controls")

        flags = st.session_state.sim_flags

        flags["new_device"] = st.checkbox(
            "📱 New device",
            value=flags["new_device"],
        )

        flags["new_location"] = st.checkbox(
            "🌍 New location",
            value=flags["new_location"],
        )

        flags["impossible_travel"] = st.checkbox(
            "✈️ Impossible travel",
            value=flags["impossible_travel"],
        )

        flags["large_transfer"] = st.checkbox(
            "💸 Large transfer",
            value=flags["large_transfer"],
        )

        flags["transfer_velocity"] = st.checkbox(
            "⚡ High transaction velocity",
            value=flags["transfer_velocity"],
        )

        st.session_state.sim_flags = flags

        st.divider()
        st.markdown("### ➕ Add Event")

        selected_event = st.selectbox(
            "Event type",
            EVENT_OPTIONS,
        )

        if st.button("＋ Add Event", use_container_width=True):
            st.session_state.sim_events.append(selected_event)
            st.session_state.sim_name = "Custom Session"
            st.session_state.sim_result = None

        if st.button("↺ Reset", use_container_width=True):
            st.session_state.sim_events = [
                "LOGIN",
                "ACCOUNT_VIEW",
                "BALANCE_CHECK",
            ]
            st.session_state.sim_flags = {
                "new_device": False,
                "new_location": False,
                "impossible_travel": False,
                "large_transfer": False,
                "transfer_velocity": False,
            }
            st.session_state.sim_name = "Custom Session"
            st.session_state.sim_result = None

        st.divider()
        st.caption(
            "All simulator sessions are synthetic. "
            "No real banking information is used."
        )

    features = make_simulated_features(
        st.session_state.sim_events,
        st.session_state.sim_flags,
    )

    left, right = st.columns([1, 1.08])

    with left:
        st.markdown('<div class="eyebrow">LIVE SESSION</div>', unsafe_allow_html=True)
        st.markdown(f"### {st.session_state.sim_name}")

        selected_description = SCENARIOS.get(
            st.session_state.sim_name,
            {},
        ).get(
            "description",
            "Custom synthetic session constructed for model testing.",
        )

        st.caption(selected_description)

        st.markdown("#### 📡 Event Timeline")
        st.caption(
            f"{len(st.session_state.sim_events)} events in chronological order"
        )
        render_events(st.session_state.sim_events)

        st.markdown("#### 🧬 Extracted Feature Vector")

        expected_features = engine.isolation_features

        feature_table = pd.DataFrame(
            {
                "Feature": expected_features,
                "Value": [
                    features.iloc[0][feature]
                    for feature in expected_features
                ],
            }
        )

        st.dataframe(
            feature_table,
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.markdown('<div class="eyebrow">DETECTION ENGINE</div>', unsafe_allow_html=True)
        st.markdown("### 🤖 ML Detection Console")

        st.caption(
            "Every detection run performs inference against the actual "
            "trained model artifacts."
        )

        if st.button(
            "▶ RUN ML DETECTION",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner(
                "Running Isolation Forest + Autoencoder + LSTM Autoencoder..."
            ):
                st.session_state.sim_result = engine.predict(
                    features,
                    st.session_state.sim_events,
                )

        if st.session_state.sim_result is None:
            st.info(
                "Load a scenario and click **RUN ML DETECTION** "
                "to analyse the session."
            )
        else:
            result = st.session_state.sim_result

            render_verdict(result)

            st.divider()
            st.markdown("#### 🧠 Model Analysis")
            render_model_results(result)

            st.divider()
            st.markdown("#### 🔎 Behavioural Risk Signals")

            signals = behavioural_signals(
                st.session_state.sim_events,
                st.session_state.sim_flags,
            )

            if signals:
                for signal in signals:
                    st.markdown(
                        f'<div class="signal">{signal}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.success(
                    "No major behavioural risk signals detected."
                )

            st.divider()
            st.markdown("#### 📋 Investigation Summary")

            if result["verdict"] == "HIGH RISK":
                st.error(
                    "Multiple independent anomaly detectors agree that "
                    "this synthetic session warrants investigation for "
                    "potential account takeover."
                )
            elif result["verdict"] == "SUSPICIOUS":
                st.warning(
                    "At least one anomaly detector identified unusual "
                    "behaviour. Additional verification is recommended."
                )
            else:
                st.success(
                    "The session remains within the learned normal "
                    "behavioural range."
                )

            st.caption(
                "Anomaly scores are not probabilities. "
                "The simulator uses a transparent 2-of-3 model consensus "
                "for a HIGH RISK verdict."
            )


# ============================================================
# HISTORICAL SESSION INVESTIGATION
# ============================================================

elif mode == "🔍 Session Investigation":

    st.markdown('<div class="eyebrow">HISTORICAL INVESTIGATION</div>', unsafe_allow_html=True)
    st.markdown("### 🔍 Session Investigation Console")
    st.caption(
        "Investigate sessions from the 12,000-session synthetic dataset."
    )

    if scores is None:
        st.error(
            "Historical investigation requires "
            "reports/combined_scores.csv."
        )
        st.stop()

    merged = df.merge(
        scores,
        on=["session_id", "user_id", "label", "attack_type"],
        how="left",
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        total_sessions = len(merged)
        st.metric("Sessions", f"{total_sessions:,}")

    with c2:
        anomaly_count = int(merged["label"].sum())
        st.metric("Known Anomalies", f"{anomaly_count:,}")

    with c3:
        st.metric(
            "Anomaly Rate",
            f"{100 * merged['label'].mean():.1f}%",
        )

    st.divider()

    filter_mode = st.radio(
        "Session filter",
        ["All sessions", "Anomalous sessions only", "Normal sessions only"],
        horizontal=True,
    )

    if filter_mode == "Anomalous sessions only":
        filtered = merged[merged["label"] == 1]
    elif filter_mode == "Normal sessions only":
        filtered = merged[merged["label"] == 0]
    else:
        filtered = merged

    search_text = st.text_input(
        "Search session or user",
        placeholder="e.g. U0000_D2",
    )

    if search_text.strip():
        query = search_text.strip().lower()
        mask = (
            filtered["session_id"].astype(str).str.lower().str.contains(query)
            | filtered["user_id"].astype(str).str.lower().str.contains(query)
        )
        filtered = filtered[mask]

    if filtered.empty:
        st.warning("No sessions match the current filters.")
        st.stop()

    session_options = filtered["session_id"].astype(str).tolist()

    selected_sid = st.selectbox(
        "Select session",
        session_options,
    )

    row = filtered[
        filtered["session_id"].astype(str) == selected_sid
    ].iloc[0]

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Session ID", str(row["session_id"]))

    with c2:
        st.metric("User ID", str(row["user_id"]))

    with c3:
        st.metric(
            "Ground Truth",
            "ANOMALY" if int(row["label"]) == 1 else "NORMAL",
        )

    with c4:
        st.metric(
            "Attack Type",
            str(row["attack_type"]),
        )

    st.markdown("#### 📡 Observed Event Sequence")

    events = sequence_events.get(
        selected_sid,
        [],
    )

    if events:
        render_events(events)
    else:
        st.info(
            "Sequence metadata is unavailable for this session."
        )

    st.markdown("#### 🧠 Stored Model Scores")

    score_columns = [
        ("Isolation Forest", "Isolation Forest"),
        ("Dense Autoencoder", "Autoencoder"),
        ("LSTM Autoencoder", "LSTM Autoencoder"),
    ]

    score_cards = st.columns(3)

    for column, (name, source_column) in zip(
        score_cards,
        score_columns,
    ):
        with column:
            value = row[source_column]

            if pd.notna(value):
                st.metric(name, f"{float(value):.4f}")
            else:
                st.metric(name, "N/A")

    st.markdown("#### 🔎 Behavioural Investigation")

    signals = behavioural_signals(
        events,
        row=row,
    )

    if signals:
        for signal in signals:
            st.markdown(
                f'<div class="signal">{signal}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.success(
            "No major rule-based behavioural signals detected."
        )

    st.markdown("#### 🧬 Session Feature Vector")

    metadata_columns = {
        "session_id",
        "user_id",
        "label",
        "attack_type",
        "Isolation Forest",
        "Autoencoder",
        "LSTM Autoencoder",
    }

    feature_columns = [
        column
        for column in row.index
        if column not in metadata_columns
    ]

    feature_table = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Value": [row[column] for column in feature_columns],
        }
    )

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Historical model scores are the stored outputs from the "
        "held-out evaluation pipeline; simulator detections perform "
        "fresh inference."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

else:

    st.markdown('<div class="eyebrow">MODEL EVALUATION</div>', unsafe_allow_html=True)
    st.markdown("### 📊 Model Performance")
    st.caption(
        "Held-out test-set results from the trained anomaly-detection models."
    )

    performance = performance_table()

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### 🏆 Best PR-AUC")

    best = performance.loc[
        performance["PR-AUC"].idxmax()
    ]

    a, b, c = st.columns(3)

    with a:
        st.metric(
            "Best Model",
            best["Model"],
        )

    with b:
        st.metric(
            "PR-AUC",
            f"{best['PR-AUC']:.3f}",
        )

    with c:
        st.metric(
            "Recall @ 1% FPR",
            f"{best['Recall @ 1% FPR']:.3f}",
        )

    st.divider()

    st.markdown("#### 🧠 Why Three Models?")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Isolation Forest**")
        st.write(
            "Detects observations that are isolated from the learned "
            "normal behavioural population."
        )

    with c2:
        st.markdown("**Dense Autoencoder**")
        st.write(
            "Learns to reconstruct normal session feature vectors. "
            "Large reconstruction error indicates unusual behaviour."
        )

    with c3:
        st.markdown("**LSTM Autoencoder**")
        st.write(
            "Models the sequence of session events and can identify "
            "unusual behavioural orderings."
        )

    st.divider()

    st.markdown("#### 📦 Dataset")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.metric("Total Sessions", f"{len(df):,}")

    with d2:
        st.metric(
            "Known Anomalies",
            f"{int(df['label'].sum()):,}",
        )

    with d3:
        st.metric(
            "Anomaly Rate",
            f"{100 * df['label'].mean():.1f}%",
        )

    st.info(
        "The models are complementary. A session may be flagged by "
        "one detector and missed by another. The live simulator uses "
        "a transparent 2-of-3 consensus rule for HIGH RISK."
    )

    st.caption(
        "All data is synthetic. Metrics are from the held-out synthetic "
        "test set and should not be interpreted as production banking performance."
    )
