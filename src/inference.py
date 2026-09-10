from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


# ============================================================
# Feature definitions
# ============================================================

EVENTS = [
    "LOGIN",
    "ACCOUNT_VIEW",
    "BALANCE_CHECK",
    "BENEFICIARY_VIEW",
    "BENEFICIARY_ADD",
    "PASSWORD_CHANGE",
    "SECURITY_CHANGE",
    "DEVICE_REGISTER",
    "TRANSFER",
    "LOGOUT",
    "FAILED_LOGIN",
]


# ============================================================
# Dense Autoencoder
# Same architecture used during training
# ============================================================

class Autoencoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, n_features),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ============================================================
# LSTM Autoencoder
# Same architecture used during training
# ============================================================

class LSTMAutoencoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=32, latent_dim=16):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size + 1,
            embedding_dim,
            padding_idx=0,
        )

        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
        )

        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        self.from_latent = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
        )

        self.output = nn.Linear(
            hidden_dim,
            embedding_dim,
        )

    def forward(self, x):
        embedded = self.embedding(x)

        _, (hidden, cell) = self.encoder(embedded)

        latent = self.to_latent(hidden[-1])

        decoder_hidden = self.from_latent(latent).unsqueeze(0)

        decoder_cell = torch.zeros_like(decoder_hidden)

        decoder_output, _ = self.decoder(
            embedded,
            (decoder_hidden, decoder_cell),
        )

        return self.output(decoder_output)


# ============================================================
# Inference Engine
# ============================================================

class InferenceEngine:

    def __init__(self):
        self.device = torch.device("cpu")

        self._load_isolation_forest()
        self._load_autoencoder()
        self._load_lstm_autoencoder()

    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------

    def _load_isolation_forest(self):

        model_path = MODELS / "isolation_forest.pkl"
        scaler_path = MODELS / "isolation_scaler.pkl"
        meta_path = MODELS / "isolation_forest_meta.json"

        self.isolation_model = joblib.load(model_path)
        self.isolation_scaler = joblib.load(scaler_path)

        with open(meta_path, "r") as f:
            self.isolation_meta = json.load(f)

        self.isolation_features = self.isolation_meta["feature_cols"]
        self.isolation_threshold = self.isolation_meta["threshold"]

    # --------------------------------------------------------
    # Dense Autoencoder
    # --------------------------------------------------------

    def _load_autoencoder(self):

        model_path = MODELS / "autoencoder.pt"
        scaler_path = MODELS / "autoencoder_scaler.pkl"
        meta_path = MODELS / "autoencoder_meta.json"

        self.autoencoder_scaler = joblib.load(scaler_path)

        with open(meta_path, "r") as f:
            self.autoencoder_meta = json.load(f)

        self.autoencoder_features = self.autoencoder_meta["feature_cols"]
        self.autoencoder_threshold = self.autoencoder_meta["threshold"]

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
        )

        self.autoencoder = Autoencoder(
            len(self.autoencoder_features)
        )

        self.autoencoder.load_state_dict(
            checkpoint["state_dict"]
        )

        self.autoencoder.to(self.device)
        self.autoencoder.eval()

    # --------------------------------------------------------
    # LSTM Autoencoder
    # --------------------------------------------------------

    def _load_lstm_autoencoder(self):

        model_path = MODELS / "lstm_autoencoder.pt"
        meta_path = MODELS / "lstm_autoencoder_meta.json"

        with open(meta_path, "r") as f:
            self.lstm_meta = json.load(f)

        self.lstm_vocab_size = self.lstm_meta["vocab_size"]
        self.lstm_threshold = self.lstm_meta["threshold"]

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
        )

        self.lstm_autoencoder = LSTMAutoencoder(
            vocab_size=self.lstm_vocab_size
        )

        self.lstm_autoencoder.load_state_dict(
            checkpoint["state_dict"]
        )

        self.lstm_autoencoder.to(self.device)
        self.lstm_autoencoder.eval()

    # ========================================================
    # Utility
    # ========================================================

    @staticmethod
    def _ensure_dataframe(features):

        if isinstance(features, pd.DataFrame):
            return features.copy()

        if isinstance(features, dict):
            return pd.DataFrame([features])

        raise TypeError(
            "features must be a pandas DataFrame or dictionary"
        )

    # ========================================================
    # Isolation Forest scoring
    # ========================================================

    def score_isolation_forest(self, features):

        df = self._ensure_dataframe(features)

        X = df[self.isolation_features]

        X_scaled = self.isolation_scaler.transform(X)

        score = float(
            -self.isolation_model.decision_function(X_scaled)[0]
        )

        return {
            "score": score,
            "threshold": self.isolation_threshold,
            "is_anomaly": score >= self.isolation_threshold,
        }

    # ========================================================
    # Dense Autoencoder scoring
    # ========================================================

    def score_autoencoder(self, features):

        df = self._ensure_dataframe(features)

        X = df[self.autoencoder_features]

        X_scaled = self.autoencoder_scaler.transform(X)

        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():

            reconstructed = self.autoencoder(X_tensor)

            error = (
                (reconstructed - X_tensor) ** 2
            ).mean(dim=1)

        score = float(error.cpu().numpy()[0])

        return {
            "score": score,
            "threshold": self.autoencoder_threshold,
            "is_anomaly": score >= self.autoencoder_threshold,
        }

    # ========================================================
    # LSTM scoring
    # ========================================================

    def score_lstm(self, events):

        if not events:
            raise ValueError(
                "At least one event is required for LSTM scoring."
            )

        unknown_events = [
            event for event in events
            if event not in EVENTS
        ]

        if unknown_events:
            raise ValueError(
                f"Unknown events: {unknown_events}"
            )

        event_to_id = {
            event: index + 1
            for index, event in enumerate(EVENTS)
        }

        sequence = [
            event_to_id[event]
            for event in events
        ]

        X = torch.tensor(
            [sequence],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():

            target = self.lstm_autoencoder.embedding(X)

            predicted = self.lstm_autoencoder(X)

            error = (
                (predicted - target) ** 2
            ).mean(dim=2)

            mask = (X != 0).float()

            score = (
                (error * mask).sum(dim=1)
                /
                mask.sum(dim=1).clamp_min(1)
            )

        score = float(score.cpu().numpy()[0])

        return {
            "score": score,
            "threshold": self.lstm_threshold,
            "is_anomaly": score >= self.lstm_threshold,
        }

    # ========================================================
    # Run all three models
    # ========================================================

    def predict(self, features, events):

        isolation = self.score_isolation_forest(features)

        autoencoder = self.score_autoencoder(features)

        lstm = self.score_lstm(events)

        votes = sum([
            isolation["is_anomaly"],
            autoencoder["is_anomaly"],
            lstm["is_anomaly"],
        ])

        if votes >= 2:
            verdict = "HIGH RISK"
        elif votes == 1:
            verdict = "SUSPICIOUS"
        else:
            verdict = "NORMAL"

        return {
            "Isolation Forest": isolation,
            "Autoencoder": autoencoder,
            "LSTM Autoencoder": lstm,
            "votes": votes,
            "verdict": verdict,
        }
    