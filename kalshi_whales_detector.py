#kalshi_whales_detector
# kalshi_whales_detector.py
# Day-1 drop-in: whale definition, rolling cluster count, Poisson tail + z-score.
# Dependencies: pandas (for time resampling), numpy (optional). No SciPy required.
#One liner on how model works: Flag 1 calculates probability of  seeing ≥ k whales in a window, if whale arrivals are random and independent at an average rate λ per window. Flag 2 calculates z-score,
# where we check if this burst of whales is unusually high compared to recent behaviour. 

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import math
import sys
import pandas as pd
import numpy as np

# -----------------------------
# Configuration (tune as needed)
# -----------------------------
@dataclass
class DetectorParams:
    window_sec: int = 120           # Δt for cluster window
    k_cluster: int = 3              # k whales per window to consider a cluster
    pval_cutoff: float = 0.01       # Poisson tail threshold
    z_cutoff: float = 3.0           # z-score threshold
    lookback_days: int = 7          # for percentile threshold (per-market)
    baseline_hours: int = 24        # for λ, μ, σ
    notional_floor_usd: float = 10_000.0  # safety floor (Kalshi)

# -----------------------------------
# Utility: stable Poisson tail P(X>=k)
# -----------------------------------
def poisson_tail_geq(k: int, lam: float) -> float:
    """
    Compute P(X >= k) for X ~ Poisson(lam).
    Uses the series 1 - sum_{i=0}^{k-1} e^{-lam} lam^i / i!
    Handles edge cases (lam ~ 0, k <= 0).
    """
    if k <= 0:
        return 1.0
    if lam <= 0:
        # If lambda ~ 0, only k=0 has probability ~1; otherwise tail is 0
        return 0.0
    term = math.exp(-lam)  # i=0 term = e^{-lam} lam^0 / 0!
    s = term
    for i in range(1, k):
        term *= lam / i
        s += term
    p_tail = 1.0 - s
    if p_tail < 0.0:
        p_tail = 0.0
    if p_tail > 1.0:
        p_tail = 1.0
    return p_tail

# -------------------------------------------------------
# Core detector: takes a trades DataFrame and returns dict
# -------------------------------------------------------
def detect_whale_alert(
    trades: pd.DataFrame,
    market_id: str,
    now: Optional[pd.Timestamp] = None,
    params: DetectorParams = DetectorParams(),
) -> Dict[str, Any]:
    """
    trades columns required:
      - 'timestamp' (datetime-like or int seconds/millis)
      - 'market' (string)
      - 'price' (float)
      - 'size'  (float or int, contracts)
    All rows should be individual trades (prints).

    Returns a dict with threshold, x_current, lambda, p_tail, z_score, and alert flag.
    """

    # --- Basic checks / copy ---
    if trades is None or len(trades) == 0:
        raise ValueError("Empty trades DataFrame.")
    df = trades.copy()

    # --- Coerce timestamp to pandas datetime (UTC-naive ok for now) ---
    if np.issubdtype(df["timestamp"].dtype, np.number):
        
        if df["timestamp"].max() > 1e12:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # --- Filter market and time sort ---
    df = df[df["market"] == market_id].sort_values("timestamp")
    if df.empty:
        raise ValueError(f"No trades found for market '{market_id}'.")

    # --- Choose 'now' (default = last trade time) ---
    if now is None:
        now = df["timestamp"].max()
    else:
        now = pd.to_datetime(now)

    # --- Compute notional for Kalshi mode ---
    if "price" not in df or "size" not in df:
        raise ValueError("Trades must include 'price' and 'size' columns.")
    df["notional"] = df["price"].astype(float) * df["size"].astype(float)

    # --- Lookback windows ---
    lookback_start = now - pd.Timedelta(days=params.lookback_days)
    baseline_start = now - pd.Timedelta(hours=params.baseline_hours)
    window_delta = pd.Timedelta(seconds=params.window_sec)

    # --- 1) Whale threshold (per market) ---
    hist = df[(df["timestamp"] >= lookback_start) & (df["timestamp"] < now)]
    if hist.empty:
        # Fallback: use all available history if lookback is empty
        hist = df[df["timestamp"] < now]
    if hist.empty:
        raise ValueError("Insufficient history to compute percentile threshold.")

    q99 = np.percentile(hist["notional"], 99) if len(hist) > 1 else hist["notional"].max()
    whale_threshold = max(q99, params.notional_floor_usd)

    # --- 2) Flag whales ---
    df["is_whale"] = df["notional"] >= whale_threshold

    # --- Restrict to last baseline_hours for rolling counts/z/λ ---
    df_base = df[(df["timestamp"] >= baseline_start) & (df["timestamp"] <= now)].copy()
    if df_base.empty:
        raise ValueError("No trades in the baseline window; widen baseline_hours or check data.")

    # --- 3) Build per-second whale counts and rolling counts over WINDOW_SEC ---
    # Create a per-second time index covering baseline to now
    idx = pd.date_range(start=baseline_start, end=now, freq="1s")
    # Series of whale hits per second
    whales_per_second = (
        df_base[df_base["is_whale"]]
        .groupby(pd.Grouper(key="timestamp", freq="1s"))
        .size()
        .reindex(idx, fill_value=0)
    )
    # Rolling count over the last WINDOW_SEC seconds (include current second)
    rolling_whales = whales_per_second.rolling(f"{params.window_sec}S").sum()

    # --- 4) Baseline stats (exclude current window to avoid leakage) ---
    baseline_end = now - pd.Timedelta(seconds=params.window_sec)
    baseline_vals = rolling_whales.loc[(rolling_whales.index >= baseline_start) &
                                       (rolling_whales.index <= baseline_end)]
    if len(baseline_vals) == 0:
        # Edge case: if the window is too large, allow a tiny baseline
        baseline_vals = rolling_whales.iloc[:-1]

    lambda_hat = float(baseline_vals.mean()) if len(baseline_vals) else 0.0
    mu = lambda_hat
    # Backstop: tiny Laplace-style smoothing so stats aren't degenerate
    lambda_hat = max(lambda_hat, 1e-3)     # ≥ 0.001 whales/window
    # If baseline is flat, use Poisson std as a fallback; floor at 0.5 to keep z reasonable
    sigma = float(baseline_vals.std(ddof=0)) if len(baseline_vals) else 0.0
    if sigma < 0.5:
        sigma = max(0.5, math.sqrt(lambda_hat))

    # --- 5) Current window metrics ---
    x_curr = float(rolling_whales.iloc[-1])  # whales in (now-window, now]
    p_tail = poisson_tail_geq(int(math.floor(x_curr)), lambda_hat)
    z_score = (x_curr - mu) / sigma

    # --- 6) Decision ---
    passes_flag1 = (x_curr >= params.k_cluster) and (p_tail < params.pval_cutoff)
    passes_flag2 = (x_curr >= params.k_cluster) and (z_score > params.z_cutoff)
    alert = passes_flag1 and passes_flag2

    # --- 7) Recent whales (for dashboard table) ---
    recent_whales = (
        df_base[(df_base["is_whale"]) & (df_base["timestamp"] <= now)]
        .sort_values("timestamp", ascending=False)
        .head(10)
        .loc[:, ["timestamp", "price", "size", "notional"]]
        .to_dict(orient="records")
    )

    return {
        "market": market_id,
        "now": now,
        "threshold_notional": whale_threshold,
        "window_sec": params.window_sec,
        "k_cluster": params.k_cluster,
        "x_current": x_curr,
        "lambda": lambda_hat,
        "p_tail": p_tail,
        "z_score": z_score,
        "passes_flag1": passes_flag1,
        "passes_flag2": passes_flag2,
        "alert": alert,
        "recent_whales": recent_whales,
    }


if __name__ == "__main__":
    import traceback

    if len(sys.argv) < 3:
        print("Usage: python3 kalshi_whales_detector.py <trades.csv> <MARKET_TICKER> [ISO_NOW] [BASELINE_HOURS]")
        sys.exit(1)

    csv_path = sys.argv[1]
    market = sys.argv[2]
    now_arg = pd.to_datetime(sys.argv[3]) if len(sys.argv) >= 4 else None

    # Optional: allow short baselines for small demo CSVs
    params = DetectorParams()
    if len(sys.argv) >= 5:
        try:
            params.baseline_hours = float(sys.argv[4])
        except Exception:
            pass

    try:
        df = pd.read_csv(csv_path)
        res = detect_whale_alert(df, market_id=market, now=now_arg, params=params)

        print("\n=== Whale Detector Result ===")
        for k, v in res.items():
            if k != "recent_whales":
                print(f"{k:>18}: {v}")

        print("\nRecent whales (up to 10):")
        for row in res["recent_whales"]:
            print(row)

    except Exception as e:
        print("[error] Detector failed:", repr(e))
        print(traceback.format_exc())
        sys.exit(2)