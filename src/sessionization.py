"""
Sessionise raw events.

The generator already provides session IDs, but this module deliberately
reconstructs sessions from user + inactivity gap so the pipeline resembles
a real event-stream workflow.
"""

from pathlib import Path
import pandas as pd

RAW = Path("data/raw/events.csv")
OUT = Path("data/processed/sessionized_events.csv")


def main():
    df = pd.read_csv(RAW, parse_dates=["timestamp"])
    df = df.sort_values(["user_id", "timestamp"]).copy()

    gap = df.groupby("user_id")["timestamp"].diff().dt.total_seconds().div(60)
    new_session = gap.isna() | (gap > 30)
    df["session_number"] = new_session.groupby(df["user_id"]).cumsum()
    df["derived_session_id"] = (
        df["user_id"] + "_D" + df["session_number"].astype(int).astype(str)
    )

    # Keep the original attack labels for controlled evaluation.
    df["session_anomaly"] = df.groupby("derived_session_id")["is_anomaly"].transform("max")
    df["session_attack_type"] = df.groupby("derived_session_id")["attack_type"].transform(
        lambda x: x[x != "normal"].iloc[0] if (x != "normal").any() else "normal"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} sessionized events to {OUT}")
    print(f"Sessions: {df['derived_session_id'].nunique():,}")


if __name__ == "__main__":
    main()
