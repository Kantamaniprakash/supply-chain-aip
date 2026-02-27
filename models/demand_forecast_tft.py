"""
Temporal Fusion Transformer — Probabilistic Demand Forecaster
=============================================================
Multi-horizon probabilistic demand forecast (7 / 14 / 30 / 90-day)
per SKU × region. Outputs P10 / P50 / P90 quantiles for safety-stock
optimisation under demand uncertainty.

Architecture: Temporal Fusion Transformer (Lim et al., 2021)
  - Variable Selection Networks: learned feature importance per time step
  - Gated Residual Networks: non-linear feature processing with skip connections
  - Multi-head attention: captures long-range temporal dependencies
  - Quantile regression: simultaneous P10/P50/P90 output heads

Reference:
  Lim, B., Arık, S.Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion
  Transformers for interpretable multi-horizon time series forecasting.
  International Journal of Forecasting, 37(4), 1748–1764.
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import warnings

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUANTILES = [0.1, 0.5, 0.9]


# ── Dataset ────────────────────────────────────────────────────────────────
class DemandDataset(Dataset):
    def __init__(self, data: np.ndarray, lookback: int, horizon: int):
        """
        data: (T, F) array — T timesteps, F features (last column = target)
        """
        self.X, self.y = [], []
        T = len(data)
        for i in range(lookback, T - horizon + 1):
            self.X.append(data[i - lookback: i])
            self.y.append(data[i: i + horizon, -1])  # target = last column
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── GRN (Gated Residual Network) ───────────────────────────────────────────
class GatedResidualNetwork(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.fc1    = nn.Linear(d_model, d_hidden)
        self.fc2    = nn.Linear(d_hidden, d_model)
        self.gate   = nn.Linear(d_model, d_model)
        self.ln     = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.fc1(x))
        h = self.drop(self.fc2(h))
        g = torch.sigmoid(self.gate(x))
        return self.ln(x + g * h)


# ── Variable Selection Network ──────────────────────────────────────────────
class VariableSelectionNetwork(nn.Module):
    """Learns soft feature selection weights at each time step."""
    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj    = nn.Linear(n_features, d_model)
        self.grn     = GatedResidualNetwork(d_model, d_model * 2, dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.n_feat  = n_features
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F)
        weights = self.softmax(self.proj(x))           # (B, T, d_model)
        selected = self.grn(weights)                   # (B, T, d_model)
        return selected, weights


# ── Temporal Fusion Transformer ─────────────────────────────────────────────
class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        n_features:  int,
        d_model:     int   = 64,
        n_heads:     int   = 4,
        n_layers:    int   = 2,
        horizon:     int   = 30,
        n_quantiles: int   = 3,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.horizon     = horizon
        self.n_quantiles = n_quantiles

        # Variable selection
        self.vsn = VariableSelectionNetwork(n_features, d_model, dropout)

        # Positional encoding
        self.pos_enc = _PositionalEncoding(d_model, dropout)

        # Temporal self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # GRN post-attention
        self.grn_post = GatedResidualNetwork(d_model, d_model * 2, dropout)

        # Quantile output heads
        self.quantile_heads = nn.ModuleList([
            nn.Linear(d_model, horizon) for _ in range(n_quantiles)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        selected, _ = self.vsn(x)              # (B, T, d_model)
        encoded      = self.pos_enc(selected)  # (B, T, d_model)
        attended     = self.transformer(encoded)
        out          = self.grn_post(attended[:, -1, :])  # take last step

        # Stack quantile outputs: (B, H, Q)
        quantile_preds = torch.stack(
            [head(out) for head in self.quantile_heads], dim=-1
        )
        return quantile_preds  # (B, horizon, n_quantiles)


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, :x.size(1)])


# ── Quantile loss ───────────────────────────────────────────────────────────
def quantile_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                  quantiles: list[float]) -> torch.Tensor:
    """
    Pinball / quantile loss averaged across quantiles and horizon.
    y_pred: (B, H, Q) | y_true: (B, H)
    """
    losses = []
    for qi, q in enumerate(quantiles):
        err = y_true - y_pred[:, :, qi]
        losses.append(torch.max((q - 1) * err, q * err).mean())
    return torch.stack(losses).mean()


# ── Trainer ─────────────────────────────────────────────────────────────────
class DemandForecaster:
    def __init__(
        self,
        lookback:   int   = 90,
        horizon:    int   = 30,
        d_model:    int   = 64,
        n_heads:    int   = 4,
        n_layers:   int   = 2,
        lr:         float = 1e-3,
        epochs:     int   = 50,
        batch_size: int   = 64,
        dropout:    float = 0.1,
    ):
        self.lookback   = lookback
        self.horizon    = horizon
        self.epochs     = epochs
        self.batch_size = batch_size
        self.scaler     = StandardScaler()
        self.model_cfg  = dict(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            horizon=horizon, n_quantiles=len(QUANTILES), dropout=dropout,
        )
        self.lr         = lr
        self.model      = None
        self.n_features = None

    def _build_model(self) -> TemporalFusionTransformer:
        return TemporalFusionTransformer(
            n_features=self.n_features, **self.model_cfg
        ).to(DEVICE)

    def fit(self, df: pd.DataFrame, feature_cols: list[str], target_col: str):
        cols = feature_cols + [target_col]
        data = self.scaler.fit_transform(df[cols].values.astype(float))
        self.n_features  = len(cols)
        self.target_idx  = len(cols) - 1

        dataset = DemandDataset(data, self.lookback, self.horizon)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = self._build_model()
        optimiser  = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                       weight_decay=1e-4)
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.epochs
        )

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimiser.zero_grad()
                preds = self.model(X_batch)          # (B, H, Q)
                loss  = quantile_loss(preds, y_batch, QUANTILES)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimiser.step()
                epoch_loss += loss.item()
            scheduler.step()
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:03d}/{self.epochs} | Loss: {epoch_loss / len(loader):.5f}")

    def predict(self, df: pd.DataFrame, feature_cols: list[str],
                target_col: str) -> pd.DataFrame:
        cols = feature_cols + [target_col]
        data = self.scaler.transform(df[cols].values.astype(float))
        X    = torch.tensor(
            data[-self.lookback:].reshape(1, self.lookback, -1), dtype=torch.float32
        ).to(DEVICE)

        self.model.eval()
        with torch.no_grad():
            q_preds = self.model(X).cpu().numpy()[0]  # (H, Q)

        # Inverse transform (target column only)
        dummy = np.zeros((self.horizon, len(cols)))
        for qi, label in enumerate(["p10", "p50", "p90"]):
            dummy[:, self.target_idx] = q_preds[:, qi]
            inv = self.scaler.inverse_transform(dummy)
            q_preds[:, qi] = inv[:, self.target_idx]

        return pd.DataFrame(q_preds, columns=["demand_p10", "demand_p50", "demand_p90"])

    def evaluate_mape(self, y_true: np.ndarray, y_pred_p50: np.ndarray) -> float:
        return mean_absolute_percentage_error(y_true, y_pred_p50)


# ── Standalone demo ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")
    np.random.seed(42)
    T = 500

    t = np.arange(T)
    trend     = 0.05 * t
    seasonal  = 20 * np.sin(2 * np.pi * t / 52)   # weekly seasonality
    noise     = np.random.normal(0, 5, T)
    demand    = 100 + trend + seasonal + noise

    df_ts = pd.DataFrame({
        "demand":             demand,
        "price_index":        100 + 0.1 * t + np.random.normal(0, 2, T),
        "promo_flag":         (np.random.rand(T) > 0.85).astype(float),
        "weekday":            (t % 7).astype(float),
        "fourier_sin_52":     np.sin(2 * np.pi * t / 52),
        "fourier_cos_52":     np.cos(2 * np.pi * t / 52),
    })

    feature_cols = ["price_index", "promo_flag", "weekday",
                    "fourier_sin_52", "fourier_cos_52"]
    target_col   = "demand"

    print("Training TFT Demand Forecaster...")
    forecaster = DemandForecaster(lookback=60, horizon=14, epochs=30,
                                  d_model=32, n_heads=4, n_layers=2)
    forecaster.fit(df_ts.iloc[:400], feature_cols, target_col)

    forecast = forecaster.predict(df_ts.iloc[:400], feature_cols, target_col)
    print("\n14-Day Probabilistic Demand Forecast:")
    print(forecast.round(2).to_string(index=False))

    mape = forecaster.evaluate_mape(
        df_ts["demand"].iloc[400:414].values,
        forecast["demand_p50"].values
    )
    print(f"\nMAPE (P50 vs actuals): {mape:.2%}")
