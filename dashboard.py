#!/usr/bin/env python3
"""
Natural Gas Commodity Analytics Dashboard
Real-time: NG Prices (Yahoo Finance), Weather (OpenMeteo), EIA API (storage/production/demand)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests
from datetime import datetime, timedelta

# Load .env file if present (picks up EIA_API_KEY without exposing it in code)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except Exception:
    _HAS_AUTOREFRESH = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PRICE_TTL   = 120     # seconds — align with 2-min auto-refresh
WEATHER_TTL = 900     # seconds — forecasts change slowly (15 min)
EIA_TTL     = 3600    # EIA storage data published weekly — cache 1 hr
DEFAULT_REFRESH_SEC = 120

EIA_BASE = "https://api.eia.gov/v2"

# EIA v2 API endpoints + process codes (validated against live API)
EIA_ENDPOINTS = {
    "storage":     "natural-gas/stor/wkly/data/",        # weekly working gas, Lower 48 (Bcf)
    "production":  "natural-gas/prod/sum/data/",          # monthly, process FPD = Dry Production
    "consumption": "natural-gas/cons/sum/data/",          # monthly, process VC0/VEU/VRS/VIN/VCS
    "elec_demand": "electricity/rto/daily-region-data/data/",  # daily US48 demand (MWh)
}
# Consumption process codes
EIA_CONS = {"total": "VC0", "power": "VEU", "res": "VRS", "ind": "VIN", "com": "VCS"}


def _mmcf_month_to_bcfd(value_mmcf: float, period) -> float:
    """Convert a monthly MMcf figure to an average Bcf/day rate."""
    import calendar
    dt = pd.to_datetime(period)
    days = calendar.monthrange(dt.year, dt.month)[1]
    return value_mmcf / 1000.0 / days


def _safe_pct(value: float, base: float) -> float:
    """Percentage change of `value` relative to `base`, robust to tiny/zero/negative base."""
    if base is None or abs(base) < 1e-6:
        return 0.0
    return (value - base) / abs(base) * 100

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⛽ Natural Gas Analytics",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #080c18; }
[data-testid="stSidebar"]           { background: #0e1421; }
[data-testid="stSidebar"] *         { color: #cbd5e1 !important; }
.section-hdr {
    background: linear-gradient(90deg, #1e3a5f 0%, #080c18 100%);
    border-left: 4px solid #3b82f6;
    padding: 10px 16px;
    border-radius: 4px;
    margin: 22px 0 10px 0;
    font-size: 17px;
    font-weight: 600;
    color: #e2e8f0;
}
.kpi-card {
    background: linear-gradient(135deg, #141c2e 0%, #1c2845 100%);
    border: 1px solid #2a3f5f;
    border-radius: 12px;
    padding: 16px 14px;
    text-align: center;
    height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.kpi-label  { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
.kpi-value  { font-size: 26px; font-weight: 700; color: #e2e8f0; line-height: 1.2; }
.kpi-delta  { font-size: 13px; margin-top: 2px; }
.pos        { color: #10b981; }
.neg        { color: #ef4444; }
.neu        { color: #f59e0b; }
.rec-bull   { background: linear-gradient(135deg,#064e3b,#065f46); border:1px solid #10b981; border-radius:12px; padding:20px; }
.rec-bear   { background: linear-gradient(135deg,#7f1d1d,#991b1b); border:1px solid #ef4444; border-radius:12px; padding:20px; }
.rec-neu    { background: linear-gradient(135deg,#1e3a5f,#1e40af); border:1px solid #3b82f6; border-radius:12px; padding:20px; }
.dev-card   { border-radius:8px; padding:14px; margin:8px 0; }
.tag        { display:inline-block; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=PRICE_TTL)
def fetch_ng_prices() -> tuple[pd.DataFrame, bool]:
    """Henry Hub NG Futures from Yahoo Finance (NG=F). Returns (df, is_live)."""
    try:
        ng = yf.Ticker("NG=F")
        hist = ng.history(period="35d", interval="1d")
        hist = hist[hist["Volume"] > 0].dropna(subset=["Close"])
        if len(hist) >= 7:
            return hist, True
    except Exception:
        pass
    # Realistic fallback centred around ~$2.65/MMBtu
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=datetime.today(), periods=30)
    px = 2.65 + np.cumsum(rng.normal(0, 0.04, 30))
    df = pd.DataFrame(
        {"Open": px - 0.02, "High": px + 0.07, "Low": px - 0.07,
         "Close": px, "Volume": rng.integers(60_000, 160_000, 30)},
        index=dates,
    )
    return df, False


@st.cache_data(ttl=WEATHER_TTL)
def fetch_weather(lat: float = 40.71, lon: float = -74.01) -> dict | None:
    """14-day weather from OpenMeteo (free, no key needed)."""
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "daily": ["temperature_2m_max", "temperature_2m_min",
                          "precipitation_sum", "windspeed_10m_max", "weathercode"],
                "past_days": 7, "forecast_days": 7,
                "timezone": "America/New_York",
            },
            timeout=12,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=EIA_TTL)
def fetch_eia_storage(api_key: str) -> dict | None:
    """
    Pull live weekly natural gas storage from EIA API v2 and derive injections.

    The EIA series returns the working-gas STOCK LEVEL (Bcf); the weekly
    injection/withdrawal is the week-over-week change in that level.

    Returns a dict with current/last_wk/last_yr/avg5 injections, deviations,
    a daily breakdown, an 8-week history, and the latest stock level — or
    None on failure (caller falls back to the seasonal model).
    """
    if not api_key:
        return None
    try:
        # ── Fetch ~320 weekly working-gas stock points (Bcf, Lower 48) ──────
        resp = requests.get(
            f"{EIA_BASE}/natural-gas/stor/wkly/data/",
            params={
                "api_key": api_key,
                "frequency": "weekly",
                "data[0]": "value",
                "facets[process][]": "SWO",        # Underground Storage - Working Gas
                "facets[duoarea][]": "R48",        # Lower 48
                "facets[series][]": "NW2_EPG0_SWO_R48_BCF",
                "sort[0][column]": "period",
                "sort[0][direction]": "desc",
                "length": 320,
                "offset": 0,
            },
            timeout=15,
        )
        resp.raise_for_status()
        rows = resp.json().get("response", {}).get("data", [])
        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["period"] = pd.to_datetime(df["period"])
        df["value"]  = pd.to_numeric(df["value"], errors="coerce")   # stock level
        df = df.dropna(subset=["value"]).sort_values("period").reset_index(drop=True)
        if len(df) < 3:
            return None

        # Weekly injection (+) / withdrawal (−) = change in stock level
        df["inj"] = df["value"].diff()
        df = df.dropna(subset=["inj"]).reset_index(drop=True)

        cur_row   = df.iloc[-1]
        prev_row  = df.iloc[-2] if len(df) >= 2 else cur_row
        cur_wk    = float(cur_row["inj"])
        last_wk   = float(prev_row["inj"])
        cur_date  = cur_row["period"]
        cur_stock = float(cur_row["value"])

        def _inj_near(ref_date, weeks_back: int, window: int = 2) -> float | None:
            """Average injection across all EIA weekly rows within ±window weeks of ref_date."""
            mask = (df["period"] >= ref_date - timedelta(weeks=weeks_back + window)) & \
                   (df["period"] <= ref_date - timedelta(weeks=weeks_back - window))
            sub = df[mask]
            return float(sub["inj"].mean()) if len(sub) else None

        last_yr = _inj_near(cur_date, 52) or last_wk

        # True 5-year average: mean of the same-week injection for each of the past 5 years
        avgs = [v for yr in range(1, 6) if (v := _inj_near(cur_date, yr * 52)) is not None]
        avg5 = float(np.mean(avgs)) if avgs else last_yr

        # 8-week injection history — compute per-week 5yr average and last-year value
        # so the reference lines in the chart reflect the correct seasonal pattern
        hist_rows = []
        for _, r in df.iloc[-9:-1].iterrows():
            wk_date = r["period"]
            wk_avgs = [v for yr in range(1, 6)
                       if (v := _inj_near(wk_date, yr * 52)) is not None]
            wk_avg5    = float(np.mean(wk_avgs)) if wk_avgs else avg5
            wk_last_yr = _inj_near(wk_date, 52) or last_yr
            hist_rows.append({
                "Week": wk_date.strftime("%b %d"),
                "Injection_Bcf": round(float(r["inj"]), 1),
                "Avg5yr_Bcf": round(wk_avg5, 1),
                "LastYr_Bcf": round(wk_last_yr, 1),
            })
        hist_rows.append({
            "Week": "Current",
            "Injection_Bcf": round(cur_wk, 1),
            "Avg5yr_Bcf": round(avg5, 1),
            "LastYr_Bcf": round(last_yr, 1),
        })

        # Daily estimate (split current week's injection across 7 days)
        daily_pct  = np.array([0.12, 0.15, 0.16, 0.16, 0.15, 0.14, 0.12])
        dates_7    = pd.date_range(end=cur_date, periods=7, freq="D")
        daily_vals = daily_pct * cur_wk

        return {
            "current":      round(cur_wk, 1),
            "last_wk":      round(last_wk, 1),
            "last_yr":      round(last_yr, 1),
            "avg5":         round(avg5, 1),
            "stock":        round(cur_stock, 0),
            "dev_wow":      round(cur_wk - last_wk, 1),
            "dev_wow_pct":  round(_safe_pct(cur_wk, last_wk), 1),
            "dev_yoy":      round(cur_wk - last_yr, 1),
            "dev_yoy_pct":  round(_safe_pct(cur_wk, last_yr), 1),
            "dev_5yr":      round(cur_wk - avg5, 1),
            "dev_5yr_pct":  round(_safe_pct(cur_wk, avg5), 1),
            "daily":        pd.DataFrame({"Date": dates_7,
                                          "Injection_Bcf": np.round(daily_vals, 2)}),
            "history":      pd.DataFrame(hist_rows),
            "as_of":        cur_date.strftime("%b %d, %Y"),
        }
    except Exception:
        return None


@st.cache_data(ttl=EIA_TTL)
def fetch_eia_fundamentals(api_key: str) -> dict | None:
    """
    Pull NG supply/demand fundamentals from EIA API v2 (monthly, converted to Bcf/day):
      • Dry production (FPD)        — primary supply driver
      • Total consumption (VC0)     — overall demand
      • Electric power burn (VEU)   — power-sector demand (largest swing factor)
      • Residential / Industrial / Commercial breakdown

    Returns a dict with full DataFrames, latest values, and YoY % changes — or
    None on failure (caller falls back to static estimates).
    """
    if not api_key:
        return None
    try:
        def _series(path: str, process: str, n: int = 18) -> pd.DataFrame | None:
            r = requests.get(
                f"{EIA_BASE}/{path}",
                params={
                    "api_key": api_key, "frequency": "monthly", "data[0]": "value",
                    "facets[duoarea][]": "NUS", "facets[process][]": process,
                    "sort[0][column]": "period", "sort[0][direction]": "desc",
                    "length": n,
                },
                timeout=15,
            )
            r.raise_for_status()
            rows = r.json().get("response", {}).get("data", [])
            if not rows:
                return None
            df = pd.DataFrame(rows)
            df["period"] = pd.to_datetime(df["period"])
            df["value"]  = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"]).sort_values("period").reset_index(drop=True)
            df["bcfd"] = [_mmcf_month_to_bcfd(v, p) for v, p in zip(df["value"], df["period"])]
            return df

        prod  = _series(EIA_ENDPOINTS["production"],  "FPD")
        cons  = _series(EIA_ENDPOINTS["consumption"], EIA_CONS["total"])
        power = _series(EIA_ENDPOINTS["consumption"], EIA_CONS["power"])
        res   = _series(EIA_ENDPOINTS["consumption"], EIA_CONS["res"])
        ind   = _series(EIA_ENDPOINTS["consumption"], EIA_CONS["ind"])
        com   = _series(EIA_ENDPOINTS["consumption"], EIA_CONS["com"])

        if prod is None or cons is None:
            return None

        def _latest(df):
            return float(df["bcfd"].iloc[-1]) if df is not None and len(df) else 0.0

        def _yoy(df):
            if df is None or len(df) < 13:
                return 0.0
            return _safe_pct(df["bcfd"].iloc[-1], df["bcfd"].iloc[-13])

        def _mom(df):
            if df is None or len(df) < 2:
                return 0.0
            return _safe_pct(df["bcfd"].iloc[-1], df["bcfd"].iloc[-2])

        return {
            "prod": prod, "cons": cons, "power": power,
            "res": res, "ind": ind, "com": com,
            "prod_latest":  _latest(prod),  "prod_yoy":  _yoy(prod),  "prod_mom":  _mom(prod),
            "cons_latest":  _latest(cons),  "cons_yoy":  _yoy(cons),  "cons_mom":  _mom(cons),
            "power_latest": _latest(power), "power_yoy": _yoy(power), "power_mom": _mom(power),
            "res_latest":   _latest(res),   "ind_latest": _latest(ind), "com_latest": _latest(com),
            "balance": round(_latest(prod) - _latest(cons), 1),
            "as_of": prod["period"].iloc[-1].strftime("%b %Y"),
        }
    except Exception:
        return None


@st.cache_data(ttl=PRICE_TTL)
def fetch_eia_electricity(api_key: str) -> pd.DataFrame | None:
    """
    Real US Lower-48 electricity demand (EIA-930, daily) for the last 7 days.
    Returns DataFrame[Date, GWh] or None on failure.
    """
    if not api_key:
        return None
    try:
        r = requests.get(
            f"{EIA_BASE}/{EIA_ENDPOINTS['elec_demand']}",
            params={
                "api_key": api_key, "frequency": "daily", "data[0]": "value",
                "facets[respondent][]": "US48", "facets[type][]": "D",
                "sort[0][column]": "period", "sort[0][direction]": "desc",
                "length": 60,
            },
            timeout=15,
        )
        r.raise_for_status()
        rows = r.json().get("response", {}).get("data", [])
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["period"] = pd.to_datetime(df["period"])
        df["value"]  = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        # The endpoint can return multiple rows per day — collapse to one daily value
        daily = df.groupby("period", as_index=False)["value"].mean().sort_values("period")
        daily = daily.tail(7)
        return pd.DataFrame({
            "Date": daily["period"].values,
            "GWh":  np.round(daily["value"].values / 1000.0, 0),  # MWh → GWh
        })
    except Exception:
        return None


def build_electricity_df(weather: dict | None) -> pd.DataFrame:
    """
    7-day electricity consumption (GWh/day) correlated with temperature.
    Mon–Sun base profile adjusted by HDD/CDD.
    """
    rng = np.random.default_rng(int(datetime.today().strftime("%Y%j")))
    dates = pd.date_range(end=datetime.today(), periods=7, freq="D")
    # Mon=410, Tue-Fri≈430, Sat=395, Sun=375
    dow_base = {0: 410, 1: 430, 2: 432, 3: 430, 4: 425, 5: 395, 6: 375}
    base = np.array([dow_base[d.weekday()] for d in dates], dtype=float)

    if weather and "daily" in weather:
        try:
            tmax = np.array(weather["daily"]["temperature_2m_max"][:7], dtype=float)
            tmin = np.array(weather["daily"]["temperature_2m_min"][:7], dtype=float)
            tmid = (tmax + tmin) / 2
            hdd = np.maximum(0, 18.3 - tmid) * 3.2
            cdd = np.maximum(0, tmid - 18.3) * 4.0
            base = base + hdd + cdd
        except Exception:
            pass

    consumption = base + rng.normal(0, 6, 7)
    return pd.DataFrame({"Date": dates, "GWh": np.round(consumption, 1)})


def build_storage_data() -> dict:
    """
    EIA-realistic weekly natural gas storage injection data.
    Returns current / last-week / last-year / 5yr-avg and daily breakdown.
    """
    rng = np.random.default_rng(int(datetime.today().strftime("%Y%j")))
    woy = datetime.today().isocalendar()[1]

    # 5-year average injection by week-of-year (Bcf) — injection season Apr-Oct
    five_yr = {
        14:24, 15:33, 16:46, 17:59, 18:68, 19:76, 20:83,
        21:89, 22:91, 23:88, 24:85, 25:80, 26:75, 27:68,
        28:62, 29:54, 30:48, 31:44, 32:38, 33:30, 34:22,
        35:15, 36:8,  37:2,  38:-6, 39:-16, 40:-29, 41:-44,
    }
    avg5 = float(five_yr.get(woy, 65))

    last_yr = avg5 * float(rng.uniform(0.85, 1.15))
    last_wk = avg5 * float(rng.uniform(0.88, 1.12))
    cur_wk  = last_wk * float(rng.uniform(0.90, 1.10))

    # Daily profile within the week (sums to ~1)
    daily_pct = np.array([0.12, 0.15, 0.16, 0.16, 0.15, 0.14, 0.12])
    daily_vals = daily_pct * cur_wk * rng.uniform(0.96, 1.04, 7)
    dates_7 = pd.date_range(end=datetime.today(), periods=7, freq="D")

    # 8-week history
    rows = []
    for i in range(8, 0, -1):
        w = woy - i
        a = float(five_yr.get(w % 52, 65))
        rows.append({
            "Week": (datetime.today() - timedelta(weeks=i)).strftime("%b %d"),
            "Injection_Bcf": round(a * float(rng.uniform(0.90, 1.10)), 1),
            "Avg5yr_Bcf": round(a, 1),
            "LastYr_Bcf": round(a * float(rng.uniform(0.85, 1.15)), 1),
        })
    rows.append({
        "Week": "Current",
        "Injection_Bcf": round(cur_wk, 1),
        "Avg5yr_Bcf": round(avg5, 1),
        "LastYr_Bcf": round(last_yr, 1),
    })

    return {
        "current": round(cur_wk, 1),
        "last_wk": round(last_wk, 1),
        "last_yr": round(last_yr, 1),
        "avg5": round(avg5, 1),
        "dev_wow": round(cur_wk - last_wk, 1),
        "dev_wow_pct": round(_safe_pct(cur_wk, last_wk), 1),
        "dev_yoy": round(cur_wk - last_yr, 1),
        "dev_yoy_pct": round(_safe_pct(cur_wk, last_yr), 1),
        "dev_5yr": round(cur_wk - avg5, 1),
        "dev_5yr_pct": round(_safe_pct(cur_wk, avg5), 1),
        "daily": pd.DataFrame({"Date": dates_7, "Injection_Bcf": np.round(daily_vals, 2)}),
        "history": pd.DataFrame(rows),
    }


def build_recommendation(prices: pd.DataFrame, storage: dict, weather: dict | None,
                         fundamentals: dict | None = None) -> dict:
    """Multi-factor signal: price momentum + storage + weather + production + demand."""
    closes = prices["Close"].values
    ma5  = np.mean(closes[-5:])
    ma10 = np.mean(closes[-10:]) if len(closes) >= 10 else ma5
    mom  = 1 if ma5 > ma10 else -1

    stor_sig = -1 if storage["dev_5yr"] > 5 else (1 if storage["dev_5yr"] < -5 else 0)

    wx_sig = 0
    if weather and "daily" in weather:
        try:
            fut = [t for t in weather["daily"]["temperature_2m_max"][7:] if t is not None]
            avg_t = np.mean(fut) if fut else 18.0
            wx_sig = 1 if avg_t < 8 or avg_t > 28 else -1
        except Exception:
            pass

    # ── EIA fundamentals (deterministic supply/demand drivers) ───────────────
    prod_sig = 0   # supply: rising YoY production → oversupply → bearish
    dem_sig  = 0   # demand: rising consumption / power burn → bullish
    if fundamentals:
        if fundamentals["prod_yoy"] > 2:
            prod_sig = -1
        elif fundamentals["prod_yoy"] < -2:
            prod_sig = 1
        demand_growth = (fundamentals["cons_yoy"] + fundamentals["power_yoy"]) / 2.0
        if demand_growth > 2:
            dem_sig = 1
        elif demand_growth < -2:
            dem_sig = -1

    total = mom + stor_sig + wx_sig + prod_sig + dem_sig
    cur   = float(closes[-1])
    chg7d = _safe_pct(cur, float(closes[-7])) if len(closes) >= 7 else 0.0

    if total >= 2:
        direction, action = "BULLISH", "BUY / LONG"
        css, icon = "rec-bull", "🟢"
        hi, lo = cur * 1.055, cur * 1.015
    elif total <= -2:
        direction, action = "BEARISH", "SELL / SHORT"
        css, icon = "rec-bear", "🔴"
        hi, lo = cur * 0.985, cur * 0.945
    elif total == 1:
        direction, action = "MILDLY BULLISH", "ACCUMULATE"
        css, icon = "rec-bull", "🟢"
        hi, lo = cur * 1.040, cur * 1.005
    elif total == -1:
        direction, action = "MILDLY BEARISH", "REDUCE"
        css, icon = "rec-bear", "🔴"
        hi, lo = cur * 0.995, cur * 0.960
    else:
        direction, action = "NEUTRAL", "HOLD / WATCH"
        css, icon = "rec-neu", "🟡"
        hi, lo = cur * 1.030, cur * 0.970

    n_factors = 5 if fundamentals else 3
    conf = min(94, 55 + abs(total) * (8 if fundamentals else 12))
    return {
        "direction": direction, "action": action, "css": css, "icon": icon,
        "confidence": conf, "cur": round(cur, 3), "hi": round(hi, 3), "lo": round(lo, 3),
        "chg7d": round(chg7d, 2), "mom": mom, "stor": stor_sig, "wx": wx_sig,
        "prod": prod_sig, "dem": dem_sig, "total": total, "n_factors": n_factors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
DARK = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(14,20,40,0.85)")


def _fig(**kw):
    fig = go.Figure()
    fig.update_layout(**DARK, **kw)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():

    # ── SIDEBAR ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⛽ NG Analytics")
        st.markdown("---")
        location = st.selectbox(
            "📍 Market Hub",
            ["New York (Henry Hub)", "Chicago", "Houston", "Los Angeles", "Denver"],
        )
        coords = {
            "New York (Henry Hub)": (40.71, -74.01),
            "Chicago":              (41.88, -87.63),
            "Houston":              (29.76, -95.37),
            "Los Angeles":          (34.05, -118.24),
            "Denver":               (39.74, -104.98),
        }
        lat, lon = coords[location]

        st.markdown("---")
        st.markdown("**🔑 EIA API Key**")
        # Pre-fill from .env / environment; user can override in sidebar at runtime
        eia_key_env = os.environ.get("EIA_API_KEY", "")
        eia_api_key = st.text_input(
            "EIA API Key",
            value=eia_key_env,
            type="password",
            placeholder="Paste your key here or set EIA_API_KEY in .env",
            help=(
                "Free key from https://www.eia.gov/opendata/  \n"
                "Unlocks real weekly storage, production, and LNG data. "
                "Leave blank to use the built-in seasonal model."
            ),
        )
        eia_active = bool(eia_api_key.strip())

        st.markdown("---")
        st.markdown("**🔁 Auto-Refresh**")
        auto_on = st.toggle("Enable auto-refresh", value=True,
                            help="Automatically re-fetch live data on an interval.")
        interval_label = st.selectbox(
            "Refresh interval",
            ["1 min", "2 min", "5 min", "10 min"],
            index=1,
            disabled=not auto_on,
        )
        interval_sec = {"1 min": 60, "2 min": 120, "5 min": 300, "10 min": 600}[interval_label]

        st.markdown("---")
        st.markdown("**Data Sources**")
        st.caption("📈 NG Prices · Yahoo Finance (NG=F)")
        st.caption("🌤️ Weather · OpenMeteo API")
        st.caption("🏭 Storage · " + ("🟢 EIA Live" if eia_active else "🟠 Seasonal Model"))
        st.caption("🛢️ Production/Demand · " + ("🟢 EIA Live" if eia_active else "🟠 Static"))
        st.caption("⚡ Electricity · " + ("🟢 EIA-930 Live" if eia_active else "🟠 Temp Model"))
        st.markdown("---")
        if st.button("🔄 Refresh Now", width='stretch'):
            st.cache_data.clear()
            st.rerun()

        status_ph = st.empty()  # filled after data fetch (live/fallback + timestamps)

    # ── AUTO-REFRESH TRIGGER ─────────────────────────────────────────────────
    if auto_on:
        if _HAS_AUTOREFRESH:
            # Soft rerun that preserves widget state. Cache TTLs ensure fresh data.
            st_autorefresh(interval=interval_sec * 1000, key="ng_autorefresh")
        else:
            # Dependency-free fallback: full-page reload via JS.
            components.html(
                f"<script>setTimeout(function(){{window.parent.location.reload();}},"
                f"{interval_sec * 1000});</script>",
                height=0,
            )

    # ── HEADER ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center;padding:18px 0 8px'>
      <h1 style='color:#e2e8f0;font-size:30px;margin:0'>
        ⛽ Natural Gas Commodity Analytics Dashboard
      </h1>
      <p style='color:#64748b;margin:4px 0 0'>
        Real-time market intelligence · Storage &amp; Injection · Price Recommendation · Supply/Demand
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── FETCH DATA ───────────────────────────────────────────────────────────
    with st.spinner("Fetching live market data …"):
        key_clean  = eia_api_key.strip()
        prices, price_live = fetch_ng_prices()
        weather    = fetch_weather(lat, lon)
        eia_stor   = fetch_eia_storage(key_clean)        if eia_active else None
        fundamentals = fetch_eia_fundamentals(key_clean) if eia_active else None
        eia_elec   = fetch_eia_electricity(key_clean)    if eia_active else None
        # Use live EIA storage if available, else fall back to seasonal model
        storage    = eia_stor if eia_stor else build_storage_data()
        stor_live  = eia_stor is not None
        # Real EIA-930 electricity demand if available, else temp-correlated model
        if eia_elec is not None and len(eia_elec) >= 5:
            elec_df, elec_live = eia_elec, True
        else:
            elec_df, elec_live = build_electricity_df(weather), False
        rec        = build_recommendation(prices, storage, weather, fundamentals)

    weather_live = weather is not None
    fund_live    = fundamentals is not None
    now = datetime.now()

    # Populate sidebar status placeholder
    px_badge   = "🟢 Live" if price_live else "🟠 Fallback"
    wx_badge   = "🟢 Live" if weather_live else "🟠 Offline"
    stor_badge = f"🟢 EIA · as of {storage.get('as_of', 'N/A')}" if stor_live else "🟠 Model"
    fund_badge = f"🟢 EIA · {fundamentals['as_of']}" if fund_live else "🟠 Static"
    elec_badge = "🟢 EIA-930" if elec_live else "🟠 Model"
    status_ph.markdown(
        f"""**Status**
- Prices: {px_badge}
- Weather: {wx_badge}
- Storage: {stor_badge}
- Prod/Demand: {fund_badge}
- Electricity: {elec_badge}
- Updated: `{now.strftime('%H:%M:%S')}`
- Auto-refresh: {'ON · ' + interval_label if auto_on else 'OFF'}"""
    )
    status_ph.markdown(
        f"""**Status**
- Prices: {px_badge}
- Weather: {wx_badge}
- Storage: {stor_badge}
- Updated: `{now.strftime('%H:%M:%S')}`
- Auto-refresh: {'ON · ' + interval_label if auto_on else 'OFF'}"""
    )

    cur_px  = prices["Close"].iloc[-1]
    prev_px = prices["Close"].iloc[-2]
    d_px    = cur_px - prev_px
    d_pct   = _safe_pct(cur_px, prev_px)

    # ── KPI STRIP ────────────────────────────────────────────────────────────
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    kcols = st.columns(6)

    def kpi(col, label, value, delta, pos):
        dc = "pos" if pos else "neg"
        col.markdown(
            f"""<div class="kpi-card">
                  <div class="kpi-label">{label}</div>
                  <div class="kpi-value">{value}</div>
                  <div class="kpi-delta {dc}">{delta}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    kpi(kcols[0], "NG Spot Price",    f"${cur_px:.3f}",
        f"{'▲' if d_px>=0 else '▼'} {d_pct:+.2f}% 1-day", d_px >= 0)
    kpi(kcols[1], "7-Day Price Chg",  f"{rec['chg7d']:+.2f}%",
        "vs 7 sessions ago", rec["chg7d"] >= 0)
    kpi(kcols[2], "Weekly Injection", f"{storage['current']} Bcf",
        f"WoW: {storage['dev_wow']:+.1f} Bcf", storage["dev_wow"] >= 0)
    kpi(kcols[3], "WoW Deviation",    f"{storage['dev_wow_pct']:+.1f}%",
        f"Abs: {storage['dev_wow']:+.1f} Bcf", storage["dev_wow_pct"] >= 0)
    kpi(kcols[4], "Avg Elec Demand",  f"{elec_df['GWh'].mean():.0f} GWh/day",
        "7-day average", True)
    kpi(kcols[5], "Recommendation",   f"{rec['icon']} {rec['direction']}",
        f"{rec['confidence']:.0f}% confidence",
        rec["direction"] == "BULLISH")

    st.markdown("---")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 1 — PRICE CHART
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-hdr">📈 Natural Gas Price Analysis — 30 Days</div>',
                unsafe_allow_html=True)

    p1, p2 = st.columns([3, 1])
    with p1:
        fig = _fig(height=380, margin=dict(l=10, r=10, t=30, b=10),
                   xaxis_rangeslider_visible=False,
                   legend=dict(orientation="h", y=1.02),
                   yaxis_title="$/MMBtu")
        fig.add_trace(go.Candlestick(
            x=prices.index,
            open=prices["Open"], high=prices["High"],
            low=prices["Low"],   close=prices["Close"],
            name="NG Futures",
            increasing_line_color="#10b981", decreasing_line_color="#ef4444",
            increasing_fillcolor="rgba(16,185,129,0.25)",
            decreasing_fillcolor="rgba(239,68,68,0.25)",
        ))
        for n, w, c, d in [("MA5",5,"#f59e0b","dot"),
                            ("MA10",10,"#8b5cf6","dash"),
                            ("MA20",20,"#3b82f6","solid")]:
            ma = prices["Close"].rolling(w).mean()
            fig.add_trace(go.Scatter(x=prices.index, y=ma, name=n,
                                     line=dict(color=c, width=1.5, dash=d)))
        st.plotly_chart(fig, width='stretch')

    with p2:
        st.markdown("**Price Statistics**")
        c7 = prices.iloc[-7:]
        stats = {
            "Current":      f"${cur_px:.3f}",
            "7-Day High":   f"${c7['High'].max():.3f}",
            "7-Day Low":    f"${c7['Low'].min():.3f}",
            "30-Day High":  f"${prices['High'].max():.3f}",
            "30-Day Low":   f"${prices['Low'].min():.3f}",
            "30-Day Avg":   f"${prices['Close'].mean():.3f}",
            "Volatility":   f"{prices['Close'].pct_change().std()*100:.2f}%",
        }
        for k, v in stats.items():
            st.markdown(f"**{k}:** {v}")

        # Volume micro-chart
        vfig = _fig(height=170, margin=dict(l=0, r=0, t=8, b=0),
                    showlegend=False, yaxis_title="Vol")
        vcolors = ["#10b981" if prices["Close"].iloc[i] >= prices["Open"].iloc[i]
                   else "#ef4444" for i in range(-7, 0)]
        vfig.add_trace(go.Bar(x=prices.index[-7:],
                              y=prices["Volume"].iloc[-7:],
                              marker_color=vcolors))
        st.plotly_chart(vfig, width='stretch')

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 2 — ELECTRICITY CONSUMPTION
    # ═════════════════════════════════════════════════════════════════════════
    elec_src = "EIA-930 · US Lower-48 (Live)" if elec_live else "Temp-Correlated Model"
    st.markdown(f'<div class="section-hdr">⚡ Last 7 Days Electricity Demand — '
                f'<span style="font-size:13px;color:#94a3b8">{elec_src}</span></div>',
                unsafe_allow_html=True)

    e1, e2 = st.columns([2, 1])
    avg_elec = elec_df["GWh"].mean()
    with e1:
        ecolors = ["#3b82f6" if v >= avg_elec else "#475569" for v in elec_df["GWh"]]
        efig = _fig(height=300, margin=dict(l=10, r=10, t=20, b=10),
                    yaxis_title="GWh", showlegend=False)
        efig.add_trace(go.Bar(
            x=elec_df["Date"].dt.strftime("%a %m/%d"),
            y=elec_df["GWh"],
            marker_color=ecolors,
            text=elec_df["GWh"].apply(lambda x: f"{x:.0f}"),
            textposition="outside",
        ))
        efig.add_hline(y=avg_elec, line_dash="dash", line_color="#f59e0b",
                       annotation_text=f"7-Day Avg: {avg_elec:.0f} GWh",
                       annotation_position="top right")
        st.plotly_chart(efig, width='stretch')

    with e2:
        peak_idx = elec_df["GWh"].idxmax()
        low_idx  = elec_df["GWh"].idxmin()
        st.metric("7-Day Total",   f"{elec_df['GWh'].sum():.0f} GWh")
        st.metric("Daily Average", f"{avg_elec:.0f} GWh")
        st.metric("Peak Day",      f"{elec_df.loc[peak_idx,'Date'].strftime('%A')} "
                                   f"({elec_df.loc[peak_idx,'GWh']:.0f})")
        st.metric("Low Day",       f"{elec_df.loc[low_idx,'Date'].strftime('%A')} "
                                   f"({elec_df.loc[low_idx,'GWh']:.0f})")
        trend = "📈 Rising" if elec_df["GWh"].iloc[-1] > elec_df["GWh"].iloc[0] else "📉 Falling"
        st.metric("7-Day Trend",   trend)

        # Real Pearson correlation between daily demand and temperature (when available)
        corr_txt = "N/A"
        if weather and "daily" in weather:
            try:
                tmax = np.array(weather["daily"]["temperature_2m_max"][:7], dtype=float)
                tmin = np.array(weather["daily"]["temperature_2m_min"][:7], dtype=float)
                tmid = (tmax + tmin) / 2
                gwh  = elec_df["GWh"].to_numpy(dtype=float)
                n = min(len(tmid), len(gwh))
                if n >= 3 and np.std(tmid[:n]) > 0 and np.std(gwh[:n]) > 0:
                    r = float(np.corrcoef(tmid[:n], gwh[:n])[0, 1])
                    strength = "High" if abs(r) >= 0.6 else "Moderate" if abs(r) >= 0.3 else "Low"
                    corr_txt = f"{r:+.2f} ({strength})"
            except Exception:
                pass
        st.metric("Demand–Temp Corr", corr_txt)

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 3 — WEATHER FORECAST
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-hdr">🌤️ 14-Day Weather Forecast & NG Demand Impact</div>',
                unsafe_allow_html=True)

    if weather and "daily" in weather:
        wdf = pd.DataFrame({
            "Date":   pd.to_datetime(weather["daily"]["time"]),
            "TMax":   weather["daily"]["temperature_2m_max"],
            "TMin":   weather["daily"]["temperature_2m_min"],
            "Precip": weather["daily"]["precipitation_sum"],
            "Wind":   weather["daily"]["windspeed_10m_max"],
            "Code":   weather["daily"].get("weathercode", [0]*14),
        }).dropna(subset=["TMax", "TMin"])
        wdf["TMid"] = (wdf["TMax"] + wdf["TMin"]) / 2
        wdf["HDD"]  = np.maximum(0, 18.3 - wdf["TMid"])
        wdf["CDD"]  = np.maximum(0, wdf["TMid"] - 18.3)
        wdf["IsPast"] = wdf["Date"] <= pd.Timestamp(datetime.today().date())

        wfig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                "Temperature (°C) — Past 7 Days + 7-Day Forecast",
                "Heating (HDD) & Cooling (CDD) Degree Days — NG Demand Proxy",
            ),
            vertical_spacing=0.13, row_heights=[0.55, 0.45],
        )
        wfig.update_layout(**DARK, height=520,
                           margin=dict(l=10, r=10, t=45, b=10),
                           legend=dict(orientation="h", y=1.02))

        # Temp band
        wfig.add_trace(go.Scatter(
            x=list(wdf["Date"]) + list(wdf["Date"][::-1]),
            y=list(wdf["TMax"]) + list(wdf["TMin"][::-1]),
            fill="toself", fillcolor="rgba(59,130,246,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="Temp Range",
        ), row=1, col=1)

        past = wdf[wdf["IsPast"]]
        futr = wdf[~wdf["IsPast"]]
        wfig.add_trace(go.Scatter(x=past["Date"], y=past["TMid"],
                                  name="Actual Temp", mode="lines+markers",
                                  line=dict(color="#60a5fa", width=2.5)), row=1, col=1)
        wfig.add_trace(go.Scatter(x=futr["Date"], y=futr["TMid"],
                                  name="Forecast Temp", mode="lines+markers",
                                  line=dict(color="#f59e0b", width=2.5, dash="dash")), row=1, col=1)
        wfig.add_vline(x=datetime.today(), line_dash="dot", line_color="#94a3b8",
                       annotation_text="Today", row=1, col=1)

        wfig.add_trace(go.Bar(x=wdf["Date"], y=wdf["HDD"],
                              name="HDD (Heating)", marker_color="#3b82f6", opacity=0.8),
                       row=2, col=1)
        wfig.add_trace(go.Bar(x=wdf["Date"], y=-wdf["CDD"],
                              name="CDD (Cooling)", marker_color="#ef4444", opacity=0.8),
                       row=2, col=1)
        wfig.update_yaxes(title_text="°C", row=1, col=1)
        wfig.update_yaxes(title_text="Degree Days", row=2, col=1)
        wfig.update_layout(barmode="relative")
        st.plotly_chart(wfig, width='stretch')

        # Day cards — next 7 days
        wc_map = {0:"☀️",1:"🌤️",2:"⛅",3:"🌥️",45:"🌫️",
                  61:"🌧️",71:"🌨️",80:"🌦️",95:"⛈️"}
        days7 = wdf.tail(7)
        dcols = st.columns(7)
        for dc, (_, row) in zip(dcols, days7.iterrows()):
            code = int(row["Code"]) if pd.notna(row["Code"]) else 0
            icon = wc_map.get(code, "🌤️")
            is_f = not row["IsPast"]
            border = "#f59e0b" if is_f else "#3b82f6"
            dc.markdown(
                f"""<div style='background:#141c2e;border:1px solid {border};
                     border-radius:8px;padding:10px;text-align:center;font-size:12px'>
                  <div style='font-size:22px'>{icon}</div>
                  <div style='color:#94a3b8'>{row['Date'].strftime('%a %m/%d')}</div>
                  <div style='color:#e2e8f0;font-weight:700'>
                    {row['TMax']:.0f}° / {row['TMin']:.0f}°</div>
                  <div style='color:#64748b'>{row['Precip']:.1f} mm</div>
                  <div style='color:{"#f59e0b" if is_f else "#60a5fa"};font-size:10px;
                       margin-top:4px'>{"FORECAST" if is_f else "ACTUAL"}</div>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.warning("Weather data unavailable — check network connectivity.")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 4 — STORAGE & INJECTION
    # ═════════════════════════════════════════════════════════════════════════
    stock_txt = (f' — <span style="font-size:13px;color:#94a3b8">Total working gas: '
                 f'{storage["stock"]:,.0f} Bcf · as of {storage["as_of"]}</span>'
                 if stor_live and "stock" in storage else "")
    st.markdown(f'<div class="section-hdr">🏭 Natural Gas Storage & Injection Analysis'
                f'{stock_txt}</div>', unsafe_allow_html=True)

    s1, s2 = st.columns([1, 2])

    with s1:
        st.markdown("**📊 Injection Summary (Bcf)**")
        rows_kv = [
            ("Current Week",         storage["current"],   None),
            ("Last Week",            storage["last_wk"],   None),
            ("Last Year (same wk)",  storage["last_yr"],   None),
            ("5-Year Average",       storage["avg5"],      None),
            ("─────────────", "", None),
            ("WoW Deviation",        f"{storage['dev_wow']:+.1f}",  storage["dev_wow"] >= 0),
            ("WoW % Change",         f"{storage['dev_wow_pct']:+.1f}%", storage["dev_wow_pct"] >= 0),
            ("YoY Deviation",        f"{storage['dev_yoy']:+.1f}",  storage["dev_yoy"] >= 0),
            ("YoY % Change",         f"{storage['dev_yoy_pct']:+.1f}%", storage["dev_yoy_pct"] >= 0),
            ("vs 5yr Avg",           f"{storage['dev_5yr']:+.1f}", storage["dev_5yr"] >= 0),
            ("vs 5yr %",             f"{storage['dev_5yr_pct']:+.1f}%", storage["dev_5yr_pct"] >= 0),
        ]
        for label, val, pos in rows_kv:
            if label.startswith("─"):
                st.markdown("---")
                continue
            if pos is None:
                st.markdown(f"**{label}:** {val} Bcf")
            else:
                c = "#10b981" if pos else "#ef4444"
                st.markdown(f"**{label}:** <span style='color:{c}'>{val}</span>",
                            unsafe_allow_html=True)

        # Gauge
        gfig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=storage["current"],
            delta={"reference": storage["avg5"],
                   "increasing": {"color": "#10b981"},
                   "decreasing": {"color": "#ef4444"}},
            gauge={
                "axis": {"range": [0, max(storage["avg5"] * 1.6, storage["current"] * 1.3)],
                         "tickcolor": "#94a3b8"},
                "bar": {"color": "#3b82f6"},
                "steps": [
                    {"range": [0, storage["avg5"] * 0.9], "color": "#1a2035"},
                    {"range": [storage["avg5"] * 0.9, storage["avg5"] * 1.1], "color": "#1e3a5f"},
                ],
                "threshold": {"line": {"color": "#f59e0b", "width": 3},
                              "thickness": 0.75, "value": storage["avg5"]},
            },
            title={"text": "Current vs 5yr Avg (Bcf)", "font": {"color": "#94a3b8", "size": 13}},
            number={"font": {"color": "#e2e8f0"}},
        ))
        gfig.update_layout(**DARK, height=240, margin=dict(l=20, r=20, t=40, b=10),
                           font={"color": "#e2e8f0"})
        st.plotly_chart(gfig, width='stretch')

    with s2:
        hist_df = storage["history"]
        wfig2 = _fig(height=400, margin=dict(l=10, r=10, t=25, b=10),
                     yaxis_title="Injection (Bcf)",
                     legend=dict(orientation="h", y=1.02), barmode="group")
        bar_colors = ["#10b981" if w == "Current" else "#3b82f6"
                      for w in hist_df["Week"]]
        wfig2.add_trace(go.Bar(
            x=hist_df["Week"], y=hist_df["Injection_Bcf"],
            name="Weekly Injection", marker_color=bar_colors,
            text=hist_df["Injection_Bcf"].apply(lambda x: f"{x:.1f}"),
            textposition="outside",
        ))
        wfig2.add_trace(go.Scatter(
            x=hist_df["Week"], y=hist_df["Avg5yr_Bcf"],
            name="5-Year Average", mode="lines+markers",
            line=dict(color="#f59e0b", width=2, dash="dash"),
        ))
        wfig2.add_trace(go.Scatter(
            x=hist_df["Week"], y=hist_df["LastYr_Bcf"],
            name="Last Year", mode="lines+markers",
            line=dict(color="#8b5cf6", width=1.8, dash="dot"),
        ))
        st.plotly_chart(wfig2, width='stretch')

    # ── DAILY INJECTION BREAKDOWN ─────────────────────────────────────────
    daily_note = ("estimated daily split of the EIA weekly figure"
                  if stor_live else "modelled")
    st.markdown(f"**📅 Daily Injection Breakdown — Current Week** "
                f"<span style='color:#64748b;font-size:12px'>({daily_note}; "
                f"EIA reports storage weekly)</span>", unsafe_allow_html=True)
    d1, d2 = st.columns([2, 1])
    daily = storage["daily"]
    with d1:
        dfig = _fig(height=280, margin=dict(l=10, r=10, t=20, b=10),
                    yaxis_title="Bcf", legend=dict(orientation="h", y=1.02))
        dfig.add_trace(go.Bar(
            x=daily["Date"].dt.strftime("%a %m/%d"),
            y=daily["Injection_Bcf"],
            name="Daily Injection", marker_color="#3b82f6",
            text=daily["Injection_Bcf"].apply(lambda x: f"{x:.2f}"),
            textposition="outside",
        ))
        dfig.add_trace(go.Scatter(
            x=daily["Date"].dt.strftime("%a %m/%d"),
            y=daily["Injection_Bcf"].cumsum(),
            name="Cumulative", mode="lines+markers",
            line=dict(color="#10b981", width=2),
            yaxis="y2",
        ))
        dfig.update_layout(
            yaxis2=dict(title="Cumulative (Bcf)", overlaying="y",
                        side="right", showgrid=False),
        )
        st.plotly_chart(dfig, width='stretch')

    with d2:
        tbl = daily.copy()
        tbl["Date"] = tbl["Date"].dt.strftime("%a, %b %d")
        tbl["% of Week"] = (daily["Injection_Bcf"] / daily["Injection_Bcf"].sum() * 100).round(1)
        tbl.columns = ["Date", "Injection (Bcf)", "% of Week"]
        st.dataframe(tbl, width='stretch', hide_index=True)
        st.metric("Weekly Total", f"{daily['Injection_Bcf'].sum():.2f} Bcf")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 5 — DEVIATION ANALYSIS
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-hdr">📉 Deviation Analysis: Current Week vs Historical Benchmarks</div>',
                unsafe_allow_html=True)

    dev1, dev2 = st.columns([2, 1])
    categories = ["Current Week", "Last Week", "Last Year", "5-Year Average"]
    values      = [storage["current"], storage["last_wk"], storage["last_yr"], storage["avg5"]]
    cols_bar    = ["#10b981", "#3b82f6", "#8b5cf6", "#f59e0b"]

    with dev1:
        devfig = _fig(height=320, margin=dict(l=10, r=10, t=40, b=10),
                      yaxis_title="Injection (Bcf)", showlegend=False)
        devfig.add_trace(go.Bar(
            x=categories, y=values,
            marker_color=cols_bar,
            text=[f"{v:.1f} Bcf" for v in values],
            textposition="outside", width=0.5,
        ))
        ymax = max(values) * 1.25
        for i in range(1, 4):
            diff = values[0] - values[i]
            c = "#10b981" if diff > 0 else "#ef4444"
            devfig.add_annotation(
                x=categories[i], y=ymax * 0.95,
                text=f"Δ {diff:+.1f} Bcf", showarrow=False,
                font=dict(color=c, size=12),
            )
        devfig.update_yaxes(range=[0, ymax])
        st.plotly_chart(devfig, width='stretch')

    with dev2:
        st.markdown("**Deviation Summary**")
        dev_rows = [
            ("vs Last Week",    storage["dev_wow"],   storage["dev_wow_pct"]),
            ("vs Last Year",    storage["dev_yoy"],   storage["dev_yoy_pct"]),
            ("vs 5yr Average",  storage["dev_5yr"],   storage["dev_5yr_pct"]),
        ]
        for label, abs_d, pct_d in dev_rows:
            c = "#10b981" if abs_d >= 0 else "#ef4444"
            arrow = "▲" if abs_d >= 0 else "▼"
            st.markdown(
                f"""<div style='background:#141c2e;border-left:4px solid {c};
                     border-radius:6px;padding:12px;margin:8px 0'>
                  <div style='color:#94a3b8;font-size:11px'>{label}</div>
                  <div style='color:{c};font-size:22px;font-weight:700'>
                    {arrow} {abs_d:+.1f} Bcf</div>
                  <div style='color:{c};font-size:14px'>{pct_d:+.1f}%</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 5b — EIA FUNDAMENTALS / PRICE DRIVERS  (real production & demand)
    # ═════════════════════════════════════════════════════════════════════════
    fund_label = f"EIA Live · {fundamentals['as_of']}" if fund_live else "Connect EIA key for live data"
    st.markdown(f'<div class="section-hdr">🛢️ NG Fundamentals & Price Drivers — '
                f'<span style="font-size:13px;color:#94a3b8">{fund_label}</span></div>',
                unsafe_allow_html=True)

    if fund_live:
        # KPI row — deterministic supply/demand drivers
        fk = st.columns(4)
        def drv(col, label, val, yoy, bullish_when_up):
            up = yoy >= 0
            # Production up = bearish (red); Demand up = bullish (green)
            good = (up == bullish_when_up)
            c = "pos" if good else "neg"
            arrow = "▲" if up else "▼"
            col.markdown(
                f"""<div class="kpi-card">
                      <div class="kpi-label">{label}</div>
                      <div class="kpi-value">{val:.1f}<span style='font-size:13px'> Bcf/d</span></div>
                      <div class="kpi-delta {c}">{arrow} {yoy:+.1f}% YoY</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        drv(fk[0], "🛢️ Dry Production",  fundamentals["prod_latest"],  fundamentals["prod_yoy"],  False)
        drv(fk[1], "🔥 Total Demand",    fundamentals["cons_latest"],  fundamentals["cons_yoy"],  True)
        drv(fk[2], "⚡ Power Burn",      fundamentals["power_latest"], fundamentals["power_yoy"], True)
        bal = fundamentals["balance"]
        fk[3].markdown(
            f"""<div class="kpi-card">
                  <div class="kpi-label">⚖️ Prod − Demand</div>
                  <div class="kpi-value">{bal:+.1f}<span style='font-size:13px'> Bcf/d</span></div>
                  <div class="kpi-delta {'neg' if bal>0 else 'pos'}">
                    {'Surplus (bearish)' if bal>0 else 'Deficit (bullish)'}</div>
                </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        fc1, fc2 = st.columns([2, 1])
        with fc1:
            # Production vs Total Consumption trend (18 months)
            pr, cs = fundamentals["prod"], fundamentals["cons"]
            pw = fundamentals["power"]
            ffig = _fig(height=340, margin=dict(l=10, r=10, t=25, b=10),
                        yaxis_title="Bcf/day", legend=dict(orientation="h", y=1.02))
            ffig.add_trace(go.Scatter(x=pr["period"], y=pr["bcfd"], name="Dry Production",
                                      line=dict(color="#10b981", width=2.5),
                                      fill="tozeroy", fillcolor="rgba(16,185,129,0.08)"))
            ffig.add_trace(go.Scatter(x=cs["period"], y=cs["bcfd"], name="Total Consumption",
                                      line=dict(color="#ef4444", width=2.5)))
            if pw is not None:
                ffig.add_trace(go.Scatter(x=pw["period"], y=pw["bcfd"], name="Power Burn",
                                          line=dict(color="#f59e0b", width=1.8, dash="dot")))
            st.plotly_chart(ffig, width='stretch')
        with fc2:
            st.markdown("**Sector Demand Split (latest)**")
            sectors = {
                "🏠 Residential": fundamentals["res_latest"],
                "🏢 Commercial":  fundamentals["com_latest"],
                "🏭 Industrial":  fundamentals["ind_latest"],
                "⚡ Power Gen":   fundamentals["power_latest"],
            }
            sec_df = pd.DataFrame({"Sector": list(sectors.keys()),
                                   "Bcf/d": [round(v, 1) for v in sectors.values()]})
            pie = go.Figure(go.Pie(
                labels=sec_df["Sector"], values=sec_df["Bcf/d"], hole=0.55,
                marker=dict(colors=["#3b82f6", "#8b5cf6", "#f59e0b", "#ef4444"]),
                textinfo="percent",
            ))
            pie.update_layout(**DARK, height=240, margin=dict(l=0, r=0, t=10, b=0),
                              showlegend=True, legend=dict(font=dict(size=10)))
            st.plotly_chart(pie, width='stretch')
    else:
        st.info("🔑 Add your EIA API key in the sidebar to unlock **real dry production, "
                "total consumption, and power-burn** data that deterministically drive NG prices.")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 6 — SUPPLY & DEMAND
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-hdr">⚖️ Supply & Demand Impact Analysis</div>',
                unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3)

    def factor_card(icon_c, title, items, bg, border):
        cards = "".join(
            f"""<div style='background:{bg};border:1px solid {border};
                 border-radius:8px;padding:10px;margin:6px 0'>
              <div style='color:{border};font-weight:700'>{icon_c} {t}</div>
              <div style='color:#94a3b8;font-size:12px'>{d}</div>
            </div>"""
            for t, d in items
        )
        return f"<div>{cards}</div>"

    bull_items = []
    if rec["wx"] > 0:
        bull_items.append(("Extreme Weather", "High heating/cooling demand next 7 days"))
    if storage["dev_5yr"] < -5:
        bull_items.append(("Below-Avg Storage", f"{abs(storage['dev_5yr']):.1f} Bcf below 5yr avg"))
    if rec["chg7d"] > 2:
        bull_items.append(("Price Momentum", f"7-day trend +{rec['chg7d']:.1f}%"))
    if fund_live and rec["dem"] > 0:
        bull_items.append(("Rising Demand (EIA)",
                           f"Consumption +{fundamentals['cons_yoy']:.1f}% / Power burn "
                           f"+{fundamentals['power_yoy']:.1f}% YoY"))
    if fund_live and rec["prod"] > 0:
        bull_items.append(("Falling Production (EIA)",
                           f"Dry output {fundamentals['prod_yoy']:+.1f}% YoY"))
    if not bull_items:
        bull_items.append(("Seasonal Demand", "Injection season nearing end"))

    bear_items = []
    if rec["wx"] < 0:
        bear_items.append(("Mild Weather", "Moderate temps reduce gas demand"))
    if storage["dev_5yr"] > 5:
        bear_items.append(("Above-Avg Storage", f"+{storage['dev_5yr']:.1f} Bcf vs 5yr avg"))
    if rec["chg7d"] < -2:
        bear_items.append(("Downward Momentum", f"7-day trend {rec['chg7d']:.1f}%"))
    if fund_live and rec["prod"] < 0:
        bear_items.append(("Rising Production (EIA)",
                           f"Dry output +{fundamentals['prod_yoy']:.1f}% YoY → oversupply"))
    if fund_live and rec["dem"] < 0:
        bear_items.append(("Weak Demand (EIA)",
                           f"Consumption {fundamentals['cons_yoy']:+.1f}% YoY"))
    if not bear_items:
        bear_items.append(("Record Production", "US dry gas output near historic highs"))

    watch_items = [
        ("Prod−Demand Balance", f"{fundamentals['balance']:+.1f} Bcf/d (EIA)" if fund_live
                                 else "Connect EIA key"),
        ("LNG Export Volumes", "Watch weekly terminal flows"),
        ("Hurricane Activity", "Gulf Coast supply disruption risk"),
        ("EIA Weekly Report", "Thursday 10:30 AM ET release"),
    ]

    sc1.markdown("**🟢 Bullish Factors (↑ Price)**")
    sc1.markdown(factor_card("↑", "", bull_items, "rgba(16,185,129,0.08)", "#10b981"),
                 unsafe_allow_html=True)
    sc2.markdown("**🔴 Bearish Factors (↓ Price)**")
    sc2.markdown(factor_card("↓", "", bear_items, "rgba(239,68,68,0.08)", "#ef4444"),
                 unsafe_allow_html=True)
    sc3.markdown("**🟡 Key Watch Items**")
    sc3.markdown(factor_card("◆", "", watch_items, "rgba(245,158,11,0.08)", "#f59e0b"),
                 unsafe_allow_html=True)

    # Supply/Demand balance — real EIA figures when available
    if fund_live:
        sup_cats = ["Dry Production"]
        sup_vals = [round(fundamentals["prod_latest"], 1)]
        dem_cats = ["Residential", "Commercial", "Industrial", "Power Burn"]
        dem_vals = [round(fundamentals["res_latest"], 1), round(fundamentals["com_latest"], 1),
                    round(fundamentals["ind_latest"], 1), round(fundamentals["power_latest"], 1)]
        sd_caption = f"Source: EIA · {fundamentals['as_of']} (Bcf/day)"
    else:
        sup_cats = ["Dry Gas Prod.", "LNG Imports", "Canadian Imports"]
        sup_vals = [106.2, 2.1, 8.4]
        dem_cats = ["Residential", "Commercial", "Industrial", "Power Gen", "LNG Exports", "Mexico Exports"]
        dem_vals = [22.1, 12.6, 28.4, 35.8, 14.2, 6.8]
        sd_caption = "Static estimates — connect EIA key for live values"

    sdfig = _fig(height=280, margin=dict(l=10, r=10, t=25, b=10),
                 yaxis_title="Bcf/day", barmode="group",
                 legend=dict(orientation="h", y=1.02))
    sdfig.add_trace(go.Bar(x=sup_cats, y=sup_vals, name="Supply",
                           marker_color="#10b981", opacity=0.85,
                           text=[f"{v:.1f}" for v in sup_vals], textposition="outside"))
    sdfig.add_trace(go.Bar(x=dem_cats, y=dem_vals, name="Demand",
                           marker_color="#ef4444", opacity=0.85,
                           text=[f"{v:.1f}" for v in dem_vals], textposition="outside"))
    sdfig.add_hline(y=np.mean(sup_vals + dem_vals), line_dash="dot",
                    line_color="#f59e0b", annotation_text="Avg",
                    annotation_position="top right")
    st.plotly_chart(sdfig, width='stretch')
    st.caption(sd_caption)

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 7 — PRICE RECOMMENDATION
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-hdr">🎯 Natural Gas Price Recommendation — Upcoming Week</div>',
                unsafe_allow_html=True)

    r1, r2 = st.columns([2, 1])
    with r1:
        st.markdown(
            f"""<div class="{rec['css']}">
              <h2 style='margin:0;font-size:28px'>
                {rec['icon']} {rec['direction']} — {rec['action']}
              </h2>
              <p style='font-size:16px;margin:8px 0'>
                Confidence: <strong>{rec['confidence']:.0f}%</strong> &nbsp;|&nbsp;
                Current: <strong>${rec['cur']:.3f}/MMBtu</strong>
              </p>
              <hr style='opacity:0.3;margin:10px 0'>
              <div style='display:flex;gap:32px;flex-wrap:wrap'>
                <div>
                  <div style='font-size:11px;opacity:0.7'>PRICE TARGET HIGH</div>
                  <div style='font-size:26px;font-weight:700'>${rec['hi']:.3f}</div>
                </div>
                <div>
                  <div style='font-size:11px;opacity:0.7'>PRICE TARGET LOW</div>
                  <div style='font-size:26px;font-weight:700'>${rec['lo']:.3f}</div>
                </div>
                <div>
                  <div style='font-size:11px;opacity:0.7'>EXPECTED RANGE</div>
                  <div style='font-size:26px;font-weight:700'>
                    ${min(rec['hi'],rec['lo']):.3f} – ${max(rec['hi'],rec['lo']):.3f}
                  </div>
                </div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.markdown(f"**Signal Breakdown** "
                    f"<span style='color:#64748b;font-size:12px'>"
                    f"(net score {rec['total']:+d} from {rec['n_factors']} factors)</span>",
                    unsafe_allow_html=True)
        signals = [
            ("Price Momentum",  rec["mom"],
             "MA5 > MA10" if rec["mom"] > 0 else "MA5 < MA10"),
            ("Storage Level",   rec["stor"],
             f"Below 5yr avg ({storage['dev_5yr']:+.1f})" if rec["stor"] > 0
             else f"Above 5yr avg ({storage['dev_5yr']:+.1f})" if rec["stor"] < 0
             else "Near 5yr avg"),
            ("Weather Demand",  rec["wx"],
             "Extreme temps" if rec["wx"] > 0 else "Mild temps"),
        ]
        if fund_live:
            signals.append((
                "Production (EIA)", rec["prod"],
                f"Dry output {fundamentals['prod_yoy']:+.1f}% YoY"))
            signals.append((
                "Demand (EIA)", rec["dem"],
                f"Cons {fundamentals['cons_yoy']:+.1f}% / Burn {fundamentals['power_yoy']:+.1f}% YoY"))

        sig_cols = st.columns(len(signals))
        for sc, (lbl, sig, desc) in zip(sig_cols, signals):
            c     = "#10b981" if sig > 0 else "#ef4444" if sig < 0 else "#f59e0b"
            ico   = "🟢" if sig > 0 else "🔴" if sig < 0 else "🟡"
            label = "BULLISH" if sig > 0 else "BEARISH" if sig < 0 else "NEUTRAL"
            sc.markdown(
                f"""<div style='background:#141c2e;border:1px solid {c};
                     border-radius:8px;padding:12px 8px;text-align:center'>
                  <div style='font-size:22px'>{ico}</div>
                  <div style='color:#e2e8f0;font-weight:700;font-size:13px;margin-top:4px'>{lbl}</div>
                  <div style='color:{c};font-size:11px;margin-top:2px'>{label}</div>
                  <div style='color:#64748b;font-size:10px;margin-top:6px'>{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with r2:
        # 7-day deterministic price path derived from the signal score (rec["total"])
        fdates = [datetime.today() + timedelta(days=i) for i in range(8)]
        # Slope scales with net signal strength: ±0.012/day per signal point, capped
        per_day = max(-0.018, min(0.018, rec["total"] * 0.006))
        trend   = np.array([(1 + per_day) ** i for i in range(8)])
        fmid    = rec["cur"] * trend
        # Uncertainty band widens with the forecast horizon
        fband   = np.linspace(0.004, 0.020, 8)
        fhi     = fmid * (1 + fband)
        flo     = fmid * (1 - fband)

        fcfig = _fig(height=310, margin=dict(l=10, r=10, t=30, b=10),
                     yaxis_title="$/MMBtu", showlegend=False,
                     title=dict(text="7-Day Price Forecast", font=dict(color="#94a3b8", size=13)))
        fcfig.add_trace(go.Scatter(
            x=fdates + fdates[::-1], y=list(fhi) + list(flo[::-1]),
            fill="toself", fillcolor="rgba(59,130,246,0.14)",
            line=dict(color="rgba(0,0,0,0)"), name="Uncertainty Band",
        ))
        fcfig.add_trace(go.Scatter(
            x=fdates, y=fmid, name="Forecast",
            line=dict(color="#3b82f6", width=2.5), mode="lines+markers",
        ))
        fcfig.add_trace(go.Scatter(
            x=[fdates[0]], y=[rec["cur"]], name="Current",
            marker=dict(color="#10b981", size=10, symbol="circle"),
            mode="markers",
        ))
        st.plotly_chart(fcfig, width='stretch')

    # ── FOOTER ──────────────────────────────────────────────────────────────
    st.markdown("---")
    refresh_note = (f"🔁 Auto-refresh every {interval_label}" if auto_on
                    else "⏸️ Auto-refresh OFF")
    st.markdown(
        f"""<div style='text-align:center;color:#475569;font-size:12px;padding:8px 0'>
          ⛽ Natural Gas Analytics Dashboard &nbsp;|&nbsp;
          Data: Yahoo Finance · OpenMeteo · EIA Seasonal Pattern &nbsp;|&nbsp;
          {refresh_note} · Last update {now.strftime('%H:%M:%S')} &nbsp;|&nbsp;
          <span style='color:#ef4444'>⚠️ For informational purposes only — not financial advice.</span>
        </div>""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
