# app_whales.py — Whale Log with projected direction + timezone toggle
# - Auto fetches Kalshi daily JSON (with look-back to the latest available day)
# - Normalizes to: timestamp, market, price, size, notional
# - Threshold = max(percentile_notional, fixed floor)
# - Shows flat whale table, with "direction_projected", notional in "mio",
#   and timestamp displayed in SGT or NYT (toggle)

from datetime import datetime, date, timedelta, timezone
import io
import json
import numpy as np
import pandas as pd
import requests
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
TZ_LABELS = {
    "SGT (Asia/Singapore)": "Asia/Singapore",
    "NYT (America/New_York)": "America/New_York",
}

def et_today() -> date:
    """Return today's date in ET (rough DST heuristic is fine here)."""
    now_utc = datetime.now(timezone.utc)
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
    """Map columns to required names; keep a tz-aware UTC timestamp for conversion."""
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    df = expand_dict_columns(df)

    for src, tgt in COMMON_COLMAP.items():
        if src in df.columns and tgt not in df.columns:
            df[tgt] = df[src]

    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    # Parse as tz-aware UTC (keep as aware for later timezone conversion)
    ts_utc = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["timestamp_utc"] = ts_utc

    df["market"] = df["market"].astype(str)
    df["price"]  = pd.to_numeric(df["price"], errors="coerce")
    df["size"]   = pd.to_numeric(df["size"], errors="coerce")

    df = df.dropna(subset=["timestamp_utc", "market", "price", "size"]).copy()
    df = df[(df["size"] > 0) & (df["price"] >= 0)]
    df["notional"] = df["price"] * df["size"]
    # Date in ET (file semantics are ET); for filtering we can use UTC date of timestamp_utc if preferred.
    df["date_utc"] = df["timestamp_utc"].dt.tz_convert("UTC").dt.date
    return df

def compute_threshold(notional: pd.Series, pct: float, floor_usd: float) -> float:
    if len(notional) == 0:
        return floor_usd
    q = float(np.quantile(notional, pct / 100.0))
    return max(q, floor_usd)

def smart_fetch_day(target_day: date, lookback_days: int):
    """Try target_day; if 404, step back up to lookback_days. Return (effective_day, url, raw_df)."""
    tried = []
    for k in range(0, lookback_days + 1):
        d = target_day - timedelta(days=k)
        url = url_for_day(d)
        if head_ok(url):
            df = fetch_json_df(url)
            return d, url, df
        tried.append(url)
    raise FileNotFoundError(f"No daily JSON found from {target_day} looking back {lookback_days} days.\nTried:\n" + "\n".join(tried))

def add_projected_direction(df_day: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """
    Project direction by comparing price to the previous trade *within the same market*.
    - price > prev_price + eps ⇒ 'Buy'
    - price < prev_price - eps ⇒ 'Sell'
    - otherwise ⇒ 'Flat'
    """
    df_day = df_day.sort_values(["market", "timestamp_utc"]).copy()
    prev_price = df_day.groupby("market")["price"].shift(1)
    diff = df_day["price"] - prev_price
    direction = np.where(diff > eps, "Buy",
                  np.where(diff < -eps, "Sell", "Flat"))
    df_day["direction_projected"] = direction
    return df_day

def format_tz(series_utc: pd.Series, tz_name: str) -> pd.Series:
    """Return a string-formatted timestamp column converted to tz_name."""
    # Convert from aware UTC to target TZ, then as naive strings for display:
    local = series_utc.dt.tz_convert(tz_name)
    return local.dt.strftime("%Y-%m-%d %H:%M:%S %Z")

# ---------- sidebar controls ----------
with st.sidebar:
    st.header("Data & Threshold")
    sel_day = st.date_input("Requested day (ET)", value=et_today())
    auto_latest = st.checkbox("Auto-find latest available (look back)", value=True)
    back_n = st.slider("Look back up to (days)", min_value=0, max_value=10, value=3, disabled=not auto_latest)

    st.header("Display")
    tz_label = st.radio("Timestamp Timezone", list(TZ_LABELS.keys()), index=1)  # default NYT
    tz_name = TZ_LABELS[tz_label]

    st.header("Whale Threshold")
    pct = st.slider("Percentile", min_value=95.0, max_value=99.9, value=99.5, step=0.1)
    floor = st.number_input("Backstop floor (USD)", value=20000.0, step=1000.0, min_value=0.0)
    st.caption("Threshold = max(percentile_notional, floor)")

# ---------- main flow ----------
try:
    if auto_latest:
        eff_day, used_url, raw = smart_fetch_day(sel_day, lookback_days=back_n)
    else:
        used_url = url_for_day(sel_day)
        raw = fetch_json_df(used_url)
        eff_day = sel_day

    df = normalize_df(raw)

    # Filter to effective day by UTC date (matches how we parsed timestamps)
    df_day = df[df["date_utc"] == eff_day].copy()
    total_rows = len(df_day)
    if total_rows == 0:
        st.warning(f"File exists but contains 0 rows for {eff_day}. Try a nearby day.")
        st.stop()

    # Project direction (per-market price change)
    df_day = add_projected_direction(df_day)

    # Threshold + whales
    thr = compute_threshold(df_day["notional"], pct=pct, floor_usd=floor)
    whales = df_day[df_day["notional"] >= thr].copy()
    whales.sort_values(["notional", "timestamp_utc"], ascending=[False, True], inplace=True)

    # Display metrics
    st.success(f"Fetched {len(raw):,} rows from {used_url}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Effective day (ET)", str(eff_day))
    c2.metric("Whale threshold (USD)", f"{thr:,.0f}")
    c3.metric("# whale trades", f"{len(whales):,}")
    c4.metric("Total trades (day)", f"{total_rows:,}")

    # Build display table
    whales["notional_mio"] = whales["notional"] / 1e6
    whales_display = pd.DataFrame({
        "timestamp": format_tz(whales["timestamp_utc"], tz_name),
        "market": whales["market"],
        "price": whales["price"],
        "size": whales["size"],
        "notional_mio": whales["notional_mio"].round(3),
        "direction_projected": whales["direction_projected"],
    })

    st.subheader(f"Whale orders on {eff_day} — shown in {tz_label}")
    st.dataframe(whales_display, use_container_width=True)

    # Download (include raw notional and UTC timestamp for analysis)
    whales_out = whales.assign(
        timestamp_display=whales_display["timestamp"]
    )[["timestamp_display", "timestamp_utc", "market", "price", "size", "notional", "notional_mio", "direction_projected"]]

    st.download_button(
        "Download whale log (CSV)",
        whales_out.to_csv(index=False),
        file_name=f"whales_{eff_day}.csv",
        mime="text/csv"
    )

    with st.expander("Preview first rows (pre-threshold)"):
        preview = df_day.head(20).copy()
        preview["timestamp_display"] = format_tz(preview["timestamp_utc"], tz_name)
        st.dataframe(preview[["timestamp_display", "market", "price", "size", "notional"]])

except requests.HTTPError as e:
    st.error(f"HTTP error fetching Kalshi daily JSON for {sel_day}.\n{e}\nURL tried: {url_for_day(sel_day)}")
except FileNotFoundError as e:
    st.error(str(e))
except Exception as e:
    st.exception(e)