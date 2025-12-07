import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone

# ---- Core modules (your new files) ----
from core.indicators import trend_and_signal
from core.grid_engine import (
    check_bot_budget,
    estimate_grid_cycle_profit,
    describe_grid_width,
)
from core.alerts import (
    send_telegram_message,
    prepare_dip_alerts,
)

# ------------------ CONFIG ------------------

COINGECKO_API = "https://api.coingecko.com/api/v3"
WATCH_COINS = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
}
ID_BY_SYMBOL = {sym: cid for cid, sym in WATCH_COINS.items()}

MAX_BOTS = 2
MAX_PER_BOT = 50.0   # USDT
MAX_TOTAL = 100.0    # USDT

# ------------------ ST PAGE SETUP & CSS ------------------

st.set_page_config(
    page_title="🤖 AI Crypto Grid Helper",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp { background-color: #020617; color: #e5e7eb; }
    body { background-color: #020617; color: #e5e7eb; }

    .main-header {
        background: radial-gradient(circle at top left, #22c55e 0%, #0ea5e9 45%, #1f2937 100%);
        padding: 18px; border-radius: 18px; color: white; margin-bottom: 12px;
        box-shadow: 0 12px 28px rgba(0,0,0,0.45); border: 1px solid rgba(226,232,240,0.18);
    }
    .main-header h1 { margin-bottom: 4px; font-size: clamp(1.6rem, 3vw, 2.3rem); }
    .main-header p { margin: 0; font-size: 0.9rem; opacity: 0.96; }

    .status-badge {
        display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.7rem;
        text-transform: uppercase; letter-spacing: 0.07em; background: rgba(15,23,42,0.7);
        border: 1px solid rgba(226,232,240,0.7); margin-top: 6px;
    }

    .metric-card {
        padding: 12px; border-radius: 14px; background: #020617;
        border: 1px solid #1f2937; margin-bottom: 10px; color: #e5e7eb;
    }
    .metric-card h3 { font-size: 0.95rem; color: #e5e7eb; margin-bottom: 4px; }
    .metric-card .value { font-size: 1.05rem; font-weight: 600; color: #e5e7eb; }
    .metric-card .sub { font-size: 0.8rem; color: #9ca3af; }

    .chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
    .chip {
        padding: 2px 8px; border-radius: 999px; font-size: 0.7rem;
        background: #0b1120; border: 1px solid #1f2937; color: #e5e7eb;
    }

    .dark-table {
        width: 100%;
        border-collapse: collapse;
        background-color: #020617;
        color: #f9fafb;
        border-radius: 12px;
        overflow: hidden;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    .dark-table th, .dark-table td {
        padding: 8px 10px;
        border: 1px solid #1f2937;
        font-size: 0.85rem;
    }
    .dark-table th {
        background-color: #020617;
        font-weight: 600;
        text-align: left;
    }

    div[data-testid="stDataFrame"] table {
        background-color: #020617 !important;
        color: #f9fafb !important;
    }
    div[data-testid="stDataFrame"] th,
    div[data-testid="stDataFrame"] td {
        background-color: #020617 !important;
        color: #f9fafb !important;
        border-color: #1f2937 !important;
        font-size: 0.85rem !important;
    }

    .ok-badge {
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        background: #166534;
        color: #bbf7d0;
        border: 1px solid #22c55e;
    }
    .warn-badge {
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        background: #7f1d1d;
        color: #fecaca;
        border: 1px solid #ef4444;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------ SESSION STATE DEFAULTS ------------------

if "alerts_enabled" not in st.session_state:
    st.session_state["alerts_enabled"] = False
if "tg_token" not in st.session_state:
    st.session_state["tg_token"] = ""
if "tg_chat_id" not in st.session_state:
    st.session_state["tg_chat_id"] = ""
if "last_alerts" not in st.session_state:
    # { "BTC": "YYYY-MM-DD", ... }
    st.session_state["last_alerts"] = {}
if "last_daily_summary_date" not in st.session_state:
    st.session_state["last_daily_summary_date"] = ""
if "bot_total_cap" not in st.session_state:
    st.session_state["bot_total_cap"] = None
if "bot_total_pnl_usdt" not in st.session_state:
    st.session_state["bot_total_pnl_usdt"] = None
if "bot_total_pnl_inr" not in st.session_state:
    st.session_state["bot_total_pnl_inr"] = None

# ------------------ HELPERS ------------------

def fetch_simple_prices():
    """Fetch current prices and 24h change for BTC/ETH/SOL from CoinGecko."""
    ids = ",".join(WATCH_COINS.keys())
    params = {
        "ids": ids,
        "vs_currencies": "usd",
        "include_24hr_change": "true",
    }
    try:
        r = requests.get(f"{COINGECKO_API}/simple/price", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        out = {}
        for cid, symbol in WATCH_COINS.items():
            d = data.get(cid, {})
            out[symbol] = {
                "price": float(d.get("usd", np.nan)),
                "change_24h": float(d.get("usd_24h_change", np.nan)),
            }
        return out
    except Exception:
        return {}


def fetch_usdt_inr():
    """Fetch USDT→INR rate from CoinGecko (via Tether)."""
    try:
        r = requests.get(
            f"{COINGECKO_API}/simple/price",
            params={"ids": "tether", "vs_currencies": "inr"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        rate = data.get("tether", {}).get("inr", None)
        if rate is None:
            raise ValueError("No tether->inr in response")
        return float(rate)
    except Exception:
        # Safe fallback
        return 85.0


def fetch_price_history(coin_id: str, days: int = 30) -> pd.Series:
    """
    Fetch historical prices (close) from CoinGecko for last N days.
    Returns pandas Series of price indexed by datetime.
    """
    try:
        r = requests.get(
            f"{COINGECKO_API}/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": days},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        prices = data.get("prices", [])
        if not prices:
            return pd.Series(dtype=float)
        ts = [datetime.fromtimestamp(p[0] / 1000, tz=timezone.utc) for p in prices]
        vals = [p[1] for p in prices]
        s = pd.Series(vals, index=pd.to_datetime(ts))
        return s
    except Exception:
        return pd.Series(dtype=float)


def fmt_usdt(x: float) -> str:
    if np.isnan(x):
        return "-"
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.2f}M USDT"
    if abs(x) >= 1_000:
        return f"{x/1_000:.2f}K USDT"
    return f"{x:.2f} USDT"


def fmt_inr_compact(x: float) -> str:
    if np.isnan(x):
        return "-"
    if abs(x) >= 1e7:
        return f"₹{x/1e7:.2f} Cr"
    if abs(x) >= 1e5:
        return f"₹{x/1e5:.2f} L"
    if abs(x) >= 1e3:
        return f"₹{x/1e3:.2f} K"
    return f"₹{x:.2f}"


def fmt_pct(x: float) -> str:
    if np.isnan(x):
        return "-"
    return f"{x:.2f}%"


def maybe_send_dip_alerts(df_signals: pd.DataFrame, usdt_inr: float):
    """
    Use core.alerts.prepare_dip_alerts + send_telegram_message
    to fire dip alerts (RSI<=35 / Buy on Dips) – once per coin per day.
    """
    if not st.session_state.get("alerts_enabled", False):
        return

    token = st.session_state.get("tg_token", "")
    chat_id = st.session_state.get("tg_chat_id", "")
    if not token or not chat_id:
        return

    last_alerts = st.session_state.get("last_alerts", {}) or {}
    alerts, new_state = prepare_dip_alerts(df_signals, usdt_inr, last_alerts)

    if alerts:
        for a in alerts:
            send_telegram_message(token, chat_id, a.text)
        st.session_state["last_alerts"] = new_state
        st.caption("📤 Telegram dip alerts sent for: " + ", ".join(a.symbol for a in alerts))


def maybe_send_daily_summary(df_signals: pd.DataFrame, usdt_inr: float):
    """
    Send ONE daily summary message (all three coins + bot allocation)
    when user opens Market Overview and Telegram alerts are enabled.
    """
    if not st.session_state.get("alerts_enabled", False):
        return

    token = st.session_state.get("tg_token", "")
    chat_id = st.session_state.get("tg_chat_id", "")
    if not token or not chat_id:
        return

    today_str = datetime.now().strftime("%Y-%m-%d")
    last_date = st.session_state.get("last_daily_summary_date", "")
    if last_date == today_str:
        return

    if df_signals is None or df_signals.empty:
        return

    lines = []
    lines.append("📊 Daily Crypto Grid Summary")
    lines.append(today_str)
    lines.append("")

    for _, r in df_signals.iterrows():
        symbol = r.get("symbol", "")
        price = r.get("price", np.nan)
        price_inr = r.get("price_inr", np.nan)
        rsi = r.get("rsi", np.nan)
        signal = r.get("signal", "")
        change = r.get("change_24h", np.nan)

        if not symbol or np.isnan(price):
            continue

        lines.append(
            f"{symbol}: {price:.2f} USDT "
            f"(≈ {fmt_inr_compact(price_inr)}) | "
            f"24h: {fmt_pct(change)} | RSI: {rsi:.1f} | {signal}"
        )
    lines.append("")

    # Bot capital info from My Bot Notes
    total_cap = st.session_state.get("bot_total_cap", None)
    total_cap_inr = st.session_state.get("bot_total_cap", None)
    total_pnl_usdt = st.session_state.get("bot_total_pnl_usdt", None)
    total_pnl_inr = st.session_state.get("bot_total_pnl_inr", None)

    lines.append("🤖 Bot Allocation (Manual Notes)")
    if total_cap is None:
        lines.append("No bot notes logged yet in app.")
    else:
        lines.append(
            f"Total capital logged: {total_cap:.2f} USDT "
            f"(≈ {fmt_inr_compact(total_cap * usdt_inr)})"
        )
        lines.append(
            f"Soft limit: {MAX_TOTAL:.2f} USDT "
            f"({total_cap / MAX_TOTAL * 100:.1f}% used)"
        )
        if total_pnl_usdt is not None:
            lines.append(
                f"Approx Unrealized P&L: {total_pnl_usdt:.2f} USDT "
                f"(≈ {fmt_inr_compact(total_pnl_inr)})"
            )

    lines.append("")
    lines.append("#DailySummary #GridBot #BTC #ETH #SOL")

    text = "\n".join(lines)
    send_telegram_message(token, chat_id, text)
    st.session_state["last_daily_summary_date"] = today_str
    st.caption("📤 Daily Telegram summary sent.")


def compute_recommended_ranges(symbol: str):
    """
    Compute three suggested ranges (Wide/Swing, Normal, Tight Scalping)
    based on last 30 days volatility for the selected coin.
    """
    coin_id = ID_BY_SYMBOL.get(symbol)
    if not coin_id:
        return None

    series = fetch_price_history(coin_id, days=30)
    if series is None or series.empty:
        return None

    close = float(series.iloc[-1])
    hi = float(series.max())
    lo = float(series.min())
    mean_price = float(series.mean()) if series.mean() != 0 else close

    vol_pct = (hi - lo) / mean_price if mean_price > 0 else 0.0

    # Base band depending on volatility
    if vol_pct < 0.25:        # low vol
        base_down, base_up = 0.12, 0.08
    elif vol_pct < 0.5:       # medium vol
        base_down, base_up = 0.18, 0.12
    else:                     # very volatile
        base_down, base_up = 0.25, 0.18

    def band(mult_down: float, mult_up: float):
        lower = close * (1 - base_down * mult_down)
        upper = close * (1 + base_up * mult_up)
        return round(lower, 2), round(upper, 2)

    wide_lower, wide_upper = band(1.0, 1.0)    # safer / swing
    norm_lower, norm_upper = band(0.7, 0.6)    # balanced
    tight_lower, tight_upper = band(0.4, 0.4)  # scalping

    return {
        "price": close,
        "vol_pct": vol_pct * 100,
        "ranges": {
            "Wide / Swing (safer)": (wide_lower, wide_upper),
            "Normal Grid": (norm_lower, norm_upper),
            "Tight Scalping": (tight_lower, tight_upper),
        },
    }


# ------------------ SIDEBAR NAV ------------------

PAGES = [
    "📊 Market Overview",
    "🤖 Grid Bot Planner",
    "📓 My Bot Notes",
    "⚙️ Configuration",
]

def sidebar():
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        page = st.radio("Go to", PAGES, label_visibility="collapsed")
        st.markdown("---")
        st.markdown("**Bot Capital Rules**")
        st.markdown(
            "- Max **2** bots\n"
            "- Max **50 USDT** per bot\n"
            "- Max **100 USDT** total"
        )
        return page


# ------------------ MAIN UI SECTIONS ------------------

def page_market_overview(usdt_inr: float):
    st.subheader("📊 Market Overview – BTC, ETH, SOL")

    prices = fetch_simple_prices()
    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    if not prices:
        st.error("Could not fetch live prices from CoinGecko. Try again later.")
        return

    st.caption(f"Live snapshots as of **{now}** (from CoinGecko, in USDT & INR)")

    # Fetch history & signals
    rows = []
    for cid, symbol in WATCH_COINS.items():
        price_series = fetch_price_history(cid, days=30)
        sig = trend_and_signal(price_series, symbol)  # from core.indicators

        p = prices.get(symbol, {})
        price_usdt = float(p.get("price", np.nan))

        row = {
            "symbol": sig.symbol,
            "trend": sig.trend,
            "rsi": sig.rsi,
            "ema50": sig.ema50,
            "ema200": sig.ema200,
            "signal": sig.signal,
            "comment": sig.comment,
            "score": sig.score,
            "last_price": sig.last_price,
            "price": price_usdt,
            "change_24h": float(p.get("change_24h", np.nan)),
        }
        row["price_inr"] = row["price"] * usdt_inr if not np.isnan(row["price"]) else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)

    # Top safe 2 coins in trend = highest score
    df_sorted = df.sort_values("score", ascending=False)
    top2 = df_sorted.head(2)

    st.markdown("#### 🛡 Top 2 \"Safer\" Trend Coins (for Grid Based DCA)")

    safe_rows = []
    for _, r in top2.iterrows():
        safe_rows.append(
            {
                "Coin": r["symbol"],
                "Signal": r["signal"],
                "Trend": r["trend"],
                "RSI": f"{r['rsi']:.1f}" if not np.isnan(r["rsi"]) else "-",
                "Last Price (USDT)": f"{r['price']:.2f}" if not np.isnan(r["price"]) else "-",
                "Last Price (INR)": fmt_inr_compact(r["price_inr"]) if not np.isnan(r["price_inr"]) else "-",
                "Score": int(r["score"]),
            }
        )
    safe_df = pd.DataFrame(safe_rows)
    st.markdown(safe_df.to_html(classes="dark-table", index=False, escape=False), unsafe_allow_html=True)

    st.markdown("#### 🔍 Detailed View – All Watched Coins")

    detail_rows = []
    for _, r in df_sorted.iterrows():
        detail_rows.append(
            {
                "Coin": r["symbol"],
                "Price (USDT)": f"{r['price']:.2f}" if not np.isnan(r["price"]) else "-",
                "Price (INR)": fmt_inr_compact(r["price_inr"]) if not np.isnan(r["price_inr"]) else "-",
                "24h Change": fmt_pct(r["change_24h"]),
                "Trend": r["trend"],
                "RSI (14)": f"{r['rsi']:.1f}" if not np.isnan(r["rsi"]) else "-",
                "EMA50": f"{r['ema50']:.2f}" if not np.isnan(r["ema50"]) else "-",
                "EMA200": f"{r['ema200']:.2f}" if not np.isnan(r["ema200"]) else "-",
                "AI Signal": r["signal"],
                "Score": int(r["score"]),
                "Comment": r["comment"],
            }
        )
    detail_df = pd.DataFrame(detail_rows)
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

    st.markdown(
        "> ⚙️ Idea: Use **Top 2 safe coins** for your next grids, "
        "preferably when RSI is in 30–55 zone and price is near lower support."
    )

    # 🔔 Telegram dip alerts (RSI <= 35 / Buy on Dips)
    maybe_send_dip_alerts(df_sorted, usdt_inr)

    # 📬 Daily summary (once per day)
    maybe_send_daily_summary(df_sorted, usdt_inr)


def page_grid_planner(usdt_inr: float):
    st.subheader("🤖 Grid Bot Planner – Buy on Dips, Book Profits on Spikes")

    st.markdown(
        "Fill your planned Binance grid parameters here. "
        "This tool will **check your capital limits** and estimate **potential profit for a full swing**."
    )

    col1, col2 = st.columns(2)
    with col1:
        coin = st.selectbox("Select Coin", ["BTC", "ETH", "SOL"])
        num_bots = st.radio("How many bots will you run?", [1, 2], horizontal=True)
        per_bot_capital = st.number_input(
            "Planned Capital per Bot (USDT)",
            min_value=0.0,
            value=25.0,
            step=1.0,
        )

    with col2:
        lower_price = st.number_input("Grid Lower Price (USDT)", min_value=0.0, value=2000.0, step=10.0)
        upper_price = st.number_input("Grid Upper Price (USDT)", min_value=0.0, value=2400.0, step=10.0)
        grid_count = st.number_input("Number of Grids", min_value=1, max_value=120, value=10, step=1)
        tp = st.number_input("Take Profit Price (optional, USDT)", min_value=0.0, value=0.0, step=10.0)
        sl = st.number_input("Stop Loss Price (optional, USDT)", min_value=0.0, value=0.0, step=10.0)

    # 🔮 Recommended ranges based on volatility
    st.markdown("### 🎯 AI Suggested Grid Ranges (based on last 30 days)")
    rec = compute_recommended_ranges(coin)
    if rec is None:
        st.info("Could not fetch enough data for recommended ranges. Try again later.")
    else:
        rows = []
        for label, (lo, hi) in rec["ranges"].items():
            rows.append(
                {
                    "Preset": label,
                    "Lower (USDT)": f"{lo:.2f}",
                    "Upper (USDT)": f"{hi:.2f}",
                }
            )
        rec_df = pd.DataFrame(rows)
        st.markdown(
            f"Reference price: **{rec['price']:.2f} USDT** • "
            f"30-day swing ≈ **{rec['vol_pct']:.1f}%**",
        )
        st.markdown(
            rec_df.to_html(classes="dark-table", index=False, escape=False),
            unsafe_allow_html=True,
        )
        st.caption(
            "You can manually copy any of these ranges into the inputs above. "
            "For safer bots, prefer **Wide / Swing**; for fast scalping, use **Tight**."
        )

    st.markdown("---")

    # Budget check via core.grid_engine
    budget = check_bot_budget(
        num_bots=num_bots,
        per_bot=per_bot_capital,
        max_bots=MAX_BOTS,
        max_per_bot=MAX_PER_BOT,
        max_total=MAX_TOTAL,
    )
    total_capital = budget.total

    if budget.ok:
        st.markdown(f"<span class='ok-badge'>{budget.messages[-1]}</span>", unsafe_allow_html=True)
        if len(budget.messages) > 1:
            for m in budget.messages[:-1]:
                st.caption(m)
    else:
        for m in budget.messages:
            st.markdown(f"<span class='warn-badge'>{m}</span>", unsafe_allow_html=True)
        st.stop()

    # Capital per bot & per grid
    per_grid_capital = per_bot_capital / grid_count if grid_count > 0 else 0.0
    grid_width = upper_price - lower_price
    per_grid_price_gap = grid_width / grid_count if grid_count > 0 else 0.0

    est = estimate_grid_cycle_profit(
        lower=lower_price,
        upper=upper_price,
        grids=grid_count,
        capital=per_bot_capital,
    )

    st.markdown("### 📌 Bot Summary (per Bot)")

    summary_rows = [
        {"Metric": "Coin", "Value": coin},
        {"Metric": "Capital per Bot", "Value": fmt_usdt(per_bot_capital)},
        {"Metric": "Capital per Bot (INR approx)", "Value": fmt_inr_compact(per_bot_capital * usdt_inr)},
        {"Metric": "Number of Grids", "Value": f"{grid_count}"},
        {"Metric": "Grid Price Range", "Value": f"{lower_price:.2f} – {upper_price:.2f} USDT"},
        {"Metric": "Average Price in Range", "Value": f"{est.avg_price:.2f} USDT" if not np.isnan(est.avg_price) else "-"},
        {"Metric": "Approx. Price Gap per Grid", "Value": f"{per_grid_price_gap:.2f} USDT"},
        {"Metric": "Capital per Grid (approx)", "Value": fmt_usdt(per_grid_capital)},
        {"Metric": "Capital per Grid (INR approx)", "Value": fmt_inr_compact(per_grid_capital * usdt_inr)},
    ]
    if tp > 0:
        summary_rows.append({"Metric": "Take Profit (TP)", "Value": f"{tp:.2f} USDT"})
    if sl > 0:
        summary_rows.append({"Metric": "Stop Loss (SL)", "Value": f"{sl:.2f} USDT"})

    summary_df = pd.DataFrame(summary_rows)
    st.markdown(summary_df.to_html(classes="dark-table", index=False, escape=False), unsafe_allow_html=True)

    st.markdown("### 📈 Estimated Profit Potential (Per Bot, Full Swing)")

    if np.isnan(est.gross_profit):
        st.warning("Check that lower < upper, grids > 0, and capital > 0.")
    else:
        profit_table = pd.DataFrame(
            [
                {
                    "Metric": "Approx. Gross Profit for 1 Full Range Cycle",
                    "Value": fmt_usdt(est.gross_profit),
                },
                {
                    "Metric": "Approx. Gross Profit (INR)",
                    "Value": fmt_inr_compact(est.gross_profit * usdt_inr),
                },
                {
                    "Metric": "Approx. ROI per Full Range Cycle",
                    "Value": fmt_pct(est.roi_pct),
                },
            ]
        )
        st.markdown(profit_table.to_html(classes="dark-table", index=False, escape=False), unsafe_allow_html=True)

        st.markdown(
            "> ⚠️ This is a rough theoretical estimate assuming **multiple scalps across the whole range**. "
            "Real results depend on volatility, fill quality, and how long price stays inside the grid."
        )

    st.markdown("### 🧠 AI-style Comments on Your Plan")

    comments = []

    # Range interpretation from core.grid_engine
    comments.append(describe_grid_width(lower_price, upper_price))

    # TP / SL comments
    if tp > 0 and tp <= upper_price:
        comments.append("TP is **inside or near the top** of your grid; okay for quick booking.")
    elif tp > upper_price:
        comments.append("TP is **above your upper grid**; your grid may close early only if price breaks out strongly.")
    if sl > 0 and sl >= lower_price:
        comments.append("SL is **inside or above** your grid; high chance of stop-out. Consider placing SL a bit lower.")
    elif sl > 0 and sl < lower_price:
        comments.append("SL is **below your grid**; more room for the bot to work before hard exit.")

    # Bot count & diversification
    if num_bots == 2:
        comments.append(
            "You are planning **2 bots**. Consider splitting between different coins "
            "(e.g., BTC + ETH or BTC + SOL) to diversify trend risk."
        )
    else:
        comments.append("Single bot plan – you can still diversify later with a second bot (within 100 USDT limit).")

    for c in comments:
        st.write("• " + c)

    st.markdown(
        "> ✅ Once this looks good, copy these parameters into your **Binance Grid Bot** and let it run. "
        "This app is only for planning and risk sanity-checks."
    )


def page_my_bot_notes(usdt_inr: float):
    st.subheader("📓 My Bot Notes & Tracking (Manual)")

    st.markdown(
        "Use this section to **log your running bots** from Binance / CoinSwitch etc. "
        "This is manual tracking – nothing connects to your exchange."
    )

    default_data = [
        {"Exchange": "Binance", "Coin": "BTC", "Lower": 85000, "Upper": 95000, "Grids": 8,
         "Capital (USDT)": 50, "Avg Entry (USDT)": 88000, "Current Price (USDT)": 90000,
         "Notes": "Wide safety range, trend bullish."},
        {"Exchange": "Binance", "Coin": "ETH", "Lower": 3000, "Upper": 3400, "Grids": 6,
         "Capital (USDT)": 50, "Avg Entry (USDT)": 3100, "Current Price (USDT)": 3200,
         "Notes": "Mid-range scalp bot."},
    ]

    df_default = pd.DataFrame(default_data)
    df = st.data_editor(
        df_default,
        use_container_width=True,
        num_rows="dynamic",
        hide_index=True,
        key="bot_notes_editor",
    )

    # Simple aggregate – capital & unrealized P&L (approx)
    try:
        caps = pd.to_numeric(df["Capital (USDT)"], errors="coerce").fillna(0.0)
    except Exception:
        caps = pd.Series(dtype=float)
    total_cap = float(caps.sum()) if not caps.empty else 0.0

    try:
        avg_entry = pd.to_numeric(df.get("Avg Entry (USDT)", 0.0), errors="coerce").fillna(0.0)
        cur_price = pd.to_numeric(df.get("Current Price (USDT)", 0.0), errors="coerce").fillna(0.0)
        qty = np.where(avg_entry > 0, caps / avg_entry, 0.0)
        pnl_usdt = qty * (cur_price - avg_entry)
        total_pnl_usdt = float(pd.Series(pnl_usdt).sum())
        total_pnl_inr = total_pnl_usdt * usdt_inr
    except Exception:
        total_pnl_usdt = 0.0
        total_pnl_inr = 0.0

    # Store in session for daily summary
    st.session_state["bot_total_cap"] = total_cap
    st.session_state["bot_total_pnl_usdt"] = total_pnl_usdt
    st.session_state["bot_total_pnl_inr"] = total_pnl_inr

    st.markdown("### 💰 Total Capital & Approx Unrealized P&L (All Logged Bots)")
    cap_table = pd.DataFrame(
        [
            {"Metric": "Total Logged Capital", "Value": fmt_usdt(total_cap)},
            {"Metric": "Total Logged Capital (INR approx)", "Value": fmt_inr_compact(total_cap * usdt_inr)},
            {"Metric": "Model Soft Limit (Your Rule)", "Value": fmt_usdt(MAX_TOTAL)},
            {"Metric": "Approx Unrealized P&L (USDT)", "Value": fmt_usdt(total_pnl_usdt)},
            {"Metric": "Approx Unrealized P&L (INR)", "Value": fmt_inr_compact(total_pnl_inr)},
        ]
    )
    st.markdown(cap_table.to_html(classes="dark-table", index=False, escape=False), unsafe_allow_html=True)

    if total_cap > MAX_TOTAL:
        st.markdown(
            "<span class='warn-badge'>You have more than 100 USDT deployed across bots. "
            "Consider reducing bots or capital per bot.</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<span class='ok-badge'>Your logged total capital is within the 100 USDT rule.</span>",
            unsafe_allow_html=True,
        )


def page_configuration():
    st.subheader("⚙️ Configuration – Telegram Alerts")

    st.markdown(
        "Configure Telegram to receive **Dip Alerts** when BTC / ETH / SOL enter "
        "**Buy on Dips / Oversold** zones (RSI ≤ 35) and a **Daily Summary**."
    )

    alerts_enabled = st.checkbox(
        "Enable Telegram Dip Alerts + Daily Summary (BTC / ETH / SOL)",
        value=st.session_state.get("alerts_enabled", False),
    )
    st.session_state["alerts_enabled"] = alerts_enabled

    token = st.text_input(
        "Telegram Bot Token",
        value=st.session_state.get("tg_token", ""),
        type="password",
    )
    chat_id = st.text_input(
        "Telegram Chat ID",
        value=st.session_state.get("tg_chat_id", ""),
    )

    st.session_state["tg_token"] = token
    st.session_state["tg_chat_id"] = chat_id

    cfg_rows = [
        {"Field": "Alerts", "Value": "Enabled" if alerts_enabled else "Disabled"},
        {"Field": "Bot Token", "Value": "●●●●●●●●●●" if token else "Not set"},
        {"Field": "Chat ID", "Value": chat_id or "Not set"},
        {"Field": "Last Daily Summary Date", "Value": st.session_state.get("last_daily_summary_date") or "Not sent"},
    ]
    cfg_df = pd.DataFrame(cfg_rows)
    st.markdown("### 🔎 Config Snapshot")
    st.markdown(cfg_df.to_html(classes="dark-table", index=False, escape=False), unsafe_allow_html=True)

    if st.button("🧪 Send Test Message"):
        if not token or not chat_id:
            st.error("Set both Token and Chat ID first.")
        else:
            resp = send_telegram_message(
                token,
                chat_id,
                "✅ Test message from AI Crypto Grid Helper – dip alerts & daily summary configured.",
            )
            st.success("Test message triggered. Check your Telegram.")
            if resp is not None:
                st.json(resp)


# ------------------ MAIN ------------------

def main():
    st.markdown(
        "<div class='main-header'>"
        "<h1>🤖 AI Crypto Grid Helper</h1>"
        "<p>BTC • ETH • SOL • Buy the Dips • Book the Spikes • View Everything in INR</p>"
        "<div class='status-badge'>Live (Planning Only – Binance Safe Mode)</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Fetch USDT→INR once per run
    usdt_inr = fetch_usdt_inr()
    st.caption(f"FX: 1 USDT ≈ {fmt_inr_compact(usdt_inr)}")

    page = sidebar()

    if page == "📊 Market Overview":
        page_market_overview(usdt_inr)
    elif page == "🤖 Grid Bot Planner":
        page_grid_planner(usdt_inr)
    elif page == "📓 My Bot Notes":
        page_my_bot_notes(usdt_inr)
    elif page == "⚙️ Configuration":
        page_configuration()


if __name__ == "__main__":
    main()
