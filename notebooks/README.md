# Notebook plan

The project can be explored with notebooks in this order:

1. `01_data_generation.ipynb` — inspect raw synthetic events
2. `02_sessionisation.ipynb` — inspect session boundaries
3. `03_feature_engineering.ipynb` — compare normal and anomalous behaviour
4. `04_isolation_forest.ipynb` — baseline model
5. `05_autoencoder.ipynb` — reconstruction-based anomaly detection
6. `06_lstm_autoencoder.ipynb` — sequence-aware anomaly detection

The production/reproducible implementations live under `src/`; notebooks are intended for analysis and presentation.
