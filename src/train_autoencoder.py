from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from model_utils import ensure_dirs, threshold_from_validation, evaluate_scores

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DATA = Path("data/processed/session_features.csv")

class Autoencoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features,64), nn.ReLU(),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8,32), nn.ReLU(),
            nn.Linear(32,64), nn.ReLU(),
            nn.Linear(64,n_features)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

def score(model, X, device):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32, device=device)
        return ((model(x)-x)**2).mean(dim=1).cpu().numpy()

def main():
    ensure_dirs()
    df = pd.read_csv(DATA)
    feature_cols = [c for c in df.columns if c not in {"session_id","user_id","label","attack_type"}]

    train_val, test = train_test_split(df, test_size=0.20, stratify=df.label, random_state=42)
    train, val = train_test_split(train_val, test_size=0.25, stratify=train_val.label, random_state=42)
    normal = train[train.label == 0]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(normal[feature_cols]).astype("float32")
    X_val = scaler.transform(val[feature_cols]).astype("float32")
    X_test = scaler.transform(test[feature_cols]).astype("float32")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Autoencoder(len(feature_cols)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    x = torch.tensor(X_train, device=device)

    model.train()
    for epoch in range(80):
        opt.zero_grad()
        loss = loss_fn(model(x), x)
        loss.backward()
        opt.step()

    val_scores = score(model, X_val, device)
    threshold = threshold_from_validation(val.label.values, val_scores)
    test_scores = score(model, X_test, device)
    metrics = evaluate_scores(test.label.values, test_scores, threshold)

    torch.save({"state_dict":model.state_dict(),"n_features":len(feature_cols)}, "models/autoencoder.pt")
    joblib.dump(scaler, "models/autoencoder_scaler.pkl")
    Path("models/autoencoder_meta.json").write_text(json.dumps({
        "feature_cols":feature_cols,
        "threshold_selected_on":"validation",
        "threshold":threshold,
        "metrics_on_held_out_test":metrics
    }, indent=2))

    out = test[["session_id","user_id","label","attack_type"]].copy()
    out["autoencoder_score"] = test_scores
    out.to_csv("reports/autoencoder_scores.csv", index=False)

    print("Autoencoder — validation-selected threshold, final held-out test")
    print(pd.Series(metrics).to_string())

if __name__ == "__main__":
    main()
