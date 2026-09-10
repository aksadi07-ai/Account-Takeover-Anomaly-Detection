from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model_utils import ensure_dirs, threshold_from_validation, evaluate_scores

DATA = Path("data/processed/session_features.csv")

def main():
    ensure_dirs()
    df = pd.read_csv(DATA)
    feature_cols = [c for c in df.columns if c not in {"session_id","user_id","label","attack_type"}]

    train_val, test = train_test_split(df, test_size=0.20, stratify=df["label"], random_state=42)
    train, val = train_test_split(train_val, test_size=0.25, stratify=train_val["label"], random_state=42)

    normal_train = train[train.label == 0]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(normal_train[feature_cols])
    X_val = scaler.transform(val[feature_cols])
    X_test = scaler.transform(test[feature_cols])

    model = IsolationForest(n_estimators=300, max_samples="auto", contamination="auto",
                            random_state=42, n_jobs=-1)
    model.fit(X_train)

    val_scores = -model.decision_function(X_val)
    threshold = threshold_from_validation(val.label.values, val_scores)

    test_scores = -model.decision_function(X_test)
    metrics = evaluate_scores(test.label.values, test_scores, threshold)

    joblib.dump(model, "models/isolation_forest.pkl")
    joblib.dump(scaler, "models/isolation_scaler.pkl")
    Path("models/isolation_forest_meta.json").write_text(json.dumps({
        "feature_cols": feature_cols,
        "threshold_selected_on": "validation",
        "threshold": threshold,
        "metrics_on_held_out_test": metrics
    }, indent=2))

    out = test[["session_id","user_id","label","attack_type"]].copy()
    out["isolation_forest_score"] = test_scores
    out.to_csv("reports/isolation_forest_scores.csv", index=False)

    print("Isolation Forest — validation-selected threshold, final held-out test")
    print(pd.Series(metrics).to_string())

if __name__ == "__main__":
    main()
