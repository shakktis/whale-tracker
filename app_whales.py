# app_whales.py — Whale Log (auto-pull Kalshi daily JSON with fallback to latest available)
# - Tries https://kalshi-public-docs.s3.amazonaws.com/reporting/trade_data_YYYY-MM-DD.json
# - If 404, looks back up to N days (configurable) to find the latest available file
# - Normalizes to columns: timestamp, market, price, size, notional
# - Threshold = max(percentile_notional, fixed floor)
# - Flat whale list for the effective day; CSV download

import io
import json
from datetime import datetime, date, timedelta, timezone
import requests
import pandas as pd
import numpy as np
import streamlit as st

# ---------- UI setup ----------
st.set_page_config(page_title="Kalshi Whale Tracker", layout="wide")
st.title("🐋 Kalshi Whale Tracker — Daily Whale Log")

# ---------- constants / helpers ----------
S3_TMPL = "https://kalshi-public-docs.s3.amazonaws.com/reporting/trade_data_{yyyy}-{mm}-{dd}.json"

COMMON_COLMAP = {
    # timestamps
    "create_ts": "timestamp", "created_at": "timestamp", "ts": "timestamp",
    # market
    "report_ticker": "market", "ticker_name": "market", "ticker": "market", "symbol": "market",
    # price
    "price": "price", "price_avg": "price", "avg_price": "price", "last_price": "price",
    # size
    "contracts_traded": "size", "qty": "size", "quantity": "size", "amount": "size", "contracts": "size",
}
REQUIRED = {"timestamp", "market", "price", "size"}

def et_today() -> date:
    now_utc = datetime.now(timezone.utc)
    # crude DST heuristic is fine for our purpose
    offset_hours = 4 if 3 <= now_utc.month <= 11 else 5
    return (now_utc - timedelta(hours=offset_hours)).date()

def url_for_day(d: date) -> str:
    return S3_TMPL.format(yyyy=d.year, mm=f"{d.month:02d}", dd=f"{d.day:02d}")

def head_ok(url: str) -> bool:
    try:
        r = requests.head(url, allow_redirects=True, timeout=15)
        return r.status_code == 200
    except requests.RequestException:
        return False

def fetch_json_df(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return pd.json_normalize(data)

def expand_dict_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    for c in cols:
        try:
            if df[c].apply(lambda x: isinstance(x, dict)).any():
                expanded = pd.json_normalize(df[c]).add_prefix(f"{c}.")
                df = pd.concat([df.drop(columns=[c]), expanded], axis=1)
        except Exception:
            pass
    return df

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    df = expand_dict_columns(df)
    for src, tgt in COMMON_COLMAP.items():
        if src in df.columns and tgt not in df.columns:
            df[tgt] = df[src]
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    df["market"] = df["market"].astype(str)
    df["price"]  = pd.to_numeric(df["price"], errors="coerce")
    df["size"]   = pd.to_numeric(df["size"], errors="coerce")
    df = df.dropna(subset=["timestamp", "market", "price", "size"])
    df = df[(df["size"] > 0) & (df["price"] >= 0)].copy()
    df["notional"] = df["price"] * df["size"]
    df["date"] = df["timestamp"].dt.date
    return df

def compute_threshold(notional: pd.Series, pct: float, floor_usd: float) -> float:
    if len(notional) == 0:
        return floor_usd
    q = float(np.quantile(notional, pct / 100.0))
    return max(q, floor_usd)

def smart_fetch_day(target_day: date, lookback_days: int):
    """
    Try target_day; if missing (404), step back day-by-day up to lookback_days.
    Returns (effective_day, url, df_raw).
    Raises if nothing found within window.
    """
    tried = []
    for k in range(0, lookback_days + 1):
        d = target_day - timedelta(days=k)
        url = url_for_day(d)
        if head_ok(url):
            df = fetch_json_df(url)
            return d, url, df
        tried.append(url)
    raise FileNotFoundError(f"No daily JSON found from {target_day} looking back {lookback_days} days.\nTried:\n" + "\n".join(tried))

# ---------- sidebar controls ----------
with st.sidebar:
    st.header("Data & Threshold")
    sel_day = st.date_input("Requested day (ET)", value=et_today())
    auto_latest = st.checkbox("Auto-find latest available (look back)", value=True)
    back_n = st.slider("Look back up to (days)", min_value=0, max_value=10, value=3, disabled=not auto_latest)
    pct = st.slider("Percentile for whale threshold", min_value=95.0, max_value=99.9, value=99.5, step=0.1)
    floor = st.number_input("Backstop floor (USD)", value=20000.0, step=1000.0, min_value=0.0)

# ---------- main flow ----------
try:
    if auto_latest:
        eff_day, used_url, raw = smart_fetch_day(sel_day, lookback_days=back_n)
    else:
        used_url = url_for_day(sel_day)
        raw = fetch_json_df(used_url)
        eff_day = sel_day

    df = normalize_df(raw)
    # Filter to the effective day explicitly
    df_day = df[df["date"] == eff_day].copy()
    total_rows = len(df_day)

    if total_rows == 0:
        st.warning(f"File exists but contains 0 rows for {eff_day}. Try a nearby day.")
        st.stop()

    thr = compute_threshold(df_day["notional"], pct=pct, floor_usd=floor)
    whales = df_day[df_day["notional"] >= thr].sort_values(["notional", "timestamp"], ascending=[False, True])

    st.success(f"Fetched {len(raw):,} rows from {used_url}")
    info1, info2, info3, info4 = st.columns(4)
    info1.metric("Effective day (ET)", str(eff_day))
    info2.metric("Whale threshold (USD)", f"{thr:,.0f}")
    info3.metric("# whale trades", f"{len(whales):,}")
    info4.metric("Total trades (day)", f"{total_rows:,}")

    st.subheader(f"Whale orders on {eff_day} (sorted by notional desc)")
    st.dataframe(whales[["timestamp", "market", "price", "size", "notional"]], use_container_width=True)

    st.download_button(
        "Download whale log (CSV)",
        whales[["timestamp", "market", "price", "size", "notional"]].to_csv(index=False),
        file_name=f"whales_{eff_day}.csv",
        mime="text/csv"
    )

    with st.expander("Preview first rows of the day (pre-threshold)"):
        st.dataframe(df_day.head(20))

except requests.HTTPError as e:
    st.error(f"HTTP error fetching Kalshi daily JSON for {sel_day}.\n{e}\nURL tried: {url_for_day(sel_day)}")
except FileNotFoundError as e:
    st.error(str(e))
except Exception as e:
    st.exception(e)