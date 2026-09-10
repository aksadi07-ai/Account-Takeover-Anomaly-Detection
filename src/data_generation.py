"""
Generate synthetic banking event data with realistic user-specific behaviour
and injected account-takeover scenarios.

Usage:
    python src/data_generation.py
"""

from pathlib import Path
import math
import random
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

OUT = Path("data/raw/events.csv")
N_USERS = 600
SESSIONS_PER_USER = 20

LOCATIONS = {
    "Chennai": (13.0827, 80.2707),
    "Bengaluru": (12.9716, 77.5946),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
}

EVENTS = [
    "LOGIN", "ACCOUNT_VIEW", "BALANCE_CHECK", "BENEFICIARY_VIEW",
    "BENEFICIARY_ADD", "PASSWORD_CHANGE", "SECURITY_CHANGE",
    "DEVICE_REGISTER", "TRANSFER", "LOGOUT", "FAILED_LOGIN"
]

DEVICES = ["Windows-Chrome", "Mac-Safari", "Android-Chrome", "iPhone-Safari"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sample_amount(user_mean):
    # Heavy-tailed but centred around each user's normal spending level.
    amount = rng.lognormal(mean=np.log(max(user_mean, 500)), sigma=0.55)
    return round(float(np.clip(amount, 100, 25000)), 2)


def add_event(rows, sid, uid, ts, event, device, location, amount=None,
              is_anomaly=False, attack_type="normal"):
    lat, lon = LOCATIONS[location]
    rows.append({
        "session_id": sid,
        "user_id": uid,
        "timestamp": ts,
        "event": event,
        "device": device,
        "location": location,
        "latitude": lat + rng.normal(0, 0.01),
        "longitude": lon + rng.normal(0, 0.01),
        "amount": amount,
        "is_anomaly": int(is_anomaly),
        "attack_type": attack_type,
    })


def generate():
    rows = []
    start = pd.Timestamp("2026-01-01 00:00:00")

    for u in range(N_USERS):
        uid = f"U{u:04d}"
        normal_device = random.choice(DEVICES)
        second_device = random.choice([d for d in DEVICES if d != normal_device])
        normal_location = random.choice(list(LOCATIONS))
        second_location = random.choice([x for x in LOCATIONS if x != normal_location])
        user_mean = float(rng.lognormal(np.log(5000), 0.65))

        for s in range(SESSIONS_PER_USER):
            sid = f"{uid}_S{s:03d}"
            ts = start + pd.Timedelta(days=int(s * 2 + rng.integers(0, 2)),
                                      hours=int(rng.integers(7, 23)),
                                      minutes=int(rng.integers(0, 60)))
            is_anomaly = (rng.random() < 0.10)
            attack_type = "normal"

            device = normal_device
            location = normal_location

            # Normal session.
            sequence = ["LOGIN"]
            if rng.random() < 0.65:
                sequence.append("ACCOUNT_VIEW")
            if rng.random() < 0.45:
                sequence.append("BALANCE_CHECK")
            if rng.random() < 0.10:
                sequence.append("BENEFICIARY_VIEW")
            if rng.random() < 0.35:
                sequence.append("TRANSFER")
            sequence.append("LOGOUT")

            # Inject one of several structured ATO patterns.
            if is_anomaly:
                attack_type = random.choice([
                    "new_device", "new_location", "credential_attack",
                    "beneficiary_attack", "velocity_attack",
                    "large_transfer", "composite_ato"
                ])
                device = second_device if attack_type != "new_location" else normal_device
                location = second_location if attack_type in {
                    "new_location", "credential_attack", "beneficiary_attack",
                    "composite_ato"
                } else normal_location

                if attack_type == "new_device":
                    sequence = ["LOGIN", "DEVICE_REGISTER", "ACCOUNT_VIEW", "TRANSFER", "LOGOUT"]
                elif attack_type == "new_location":
                    sequence = ["LOGIN", "ACCOUNT_VIEW", "BALANCE_CHECK", "TRANSFER", "LOGOUT"]
                elif attack_type == "credential_attack":
                    sequence = ["FAILED_LOGIN", "FAILED_LOGIN", "FAILED_LOGIN",
                                "LOGIN", "PASSWORD_CHANGE", "SECURITY_CHANGE", "LOGOUT"]
                elif attack_type == "beneficiary_attack":
                    sequence = ["LOGIN", "DEVICE_REGISTER", "BENEFICIARY_ADD",
                                "TRANSFER", "TRANSFER", "LOGOUT"]
                elif attack_type == "velocity_attack":
                    sequence = ["LOGIN"] + ["TRANSFER"] * int(rng.integers(5, 9)) + ["LOGOUT"]
                elif attack_type == "large_transfer":
                    sequence = ["LOGIN", "ACCOUNT_VIEW", "TRANSFER", "LOGOUT"]
                else:
                    sequence = [
                        "FAILED_LOGIN", "LOGIN", "DEVICE_REGISTER",
                        "BENEFICIARY_ADD", "PASSWORD_CHANGE",
                        "SECURITY_CHANGE", "TRANSFER", "TRANSFER", "LOGOUT"
                    ]

            current = ts
            for i, event in enumerate(sequence):
                # Attack sessions are deliberately compressed in time.
                delta = int(rng.integers(10, 180) if is_anomaly else rng.integers(30, 900))
                current += pd.Timedelta(seconds=delta)

                amount = None
                if event == "TRANSFER":
                    if is_anomaly and attack_type == "large_transfer":
                        amount = round(float(rng.uniform(30000, 90000)), 2)
                    elif is_anomaly and attack_type == "composite_ato":
                        amount = round(float(rng.uniform(20000, 70000)), 2)
                    else:
                        amount = sample_amount(user_mean)

                add_event(
                    rows, sid, uid, current, event, device, location, amount,
                    is_anomaly, attack_type
                )

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} events to {OUT}")
    print(df.groupby("attack_type").size().sort_values(ascending=False))


if __name__ == "__main__":
    generate()
