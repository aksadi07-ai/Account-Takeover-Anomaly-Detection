"""
Simple model-agnostic local explanation for a session.

This intentionally explains the model through the session's behavioural
features rather than pretending the anomaly score itself is causal.
"""

import pandas as pd
import numpy as np

FEATURE_LABELS = {
    "failed_logins": "Repeated failed logins",
    "new_device": "New/unseen device",
    "new_location": "New geographic location",
    "impossible_travel": "Impossible-travel pattern",
    "beneficiary_adds": "New beneficiary added",
    "password_changes": "Password changed",
    "security_changes": "Security settings changed",
    "device_registers": "New device registered",
    "transaction_velocity_per_min": "High transaction velocity",
    "max_transfer_amount": "Unusually large transfer",
    "amount_zscore": "Transaction amount deviates from user's baseline",
}


def explain_session(row: pd.Series, top_n=5):
    reasons = []
    for col, label in FEATURE_LABELS.items():
        value = row.get(col, 0)
        if pd.isna(value):
            continue
        if col in {"new_device", "new_location", "impossible_travel",
                   "password_changes", "security_changes", "device_registers"} and value > 0:
            reasons.append((label, float(value)))
        elif col == "failed_logins" and value >= 2:
            reasons.append((label, float(value)))
        elif col == "beneficiary_adds" and value > 0:
            reasons.append((label, float(value)))
        elif col == "transaction_velocity_per_min" and value > 0.12:
            reasons.append((label, float(value)))
        elif col == "max_transfer_amount" and value > 25000:
            reasons.append((label, float(value)))
        elif col == "amount_zscore" and abs(value) > 2:
            reasons.append((label, float(abs(value))))

    return sorted(reasons, key=lambda x: x[1], reverse=True)[:top_n]
