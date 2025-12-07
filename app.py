import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone

# ------------------ CONFIG ------------------

COINGECKO_API = "https://api.coingecko.com/api/v3"
WATCH_COINS = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
}

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


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Classic RSI implementation."""
    if series is None or len(series) < period + 1:
        return np.nan
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def compute_ema(series: pd.Series, span: int) -> float:
    if series is None or len(series) < span:
        return np.nan
    ema = series.ewm(span=span, adjust=False).mean()
    return float(ema.iloc[-1])


def trend_and_signal(price_series: pd.Series, symbol: str) -> dict:
    """Return trend, RSI, EMAs and AI-style signal."""
    info = {
        "symbol": symbol,
        "trend": "Unknown",
        "rsi": np.nan,
        "ema50": np.nan,
        "ema200": np.nan,
        "signal": "No data",
        "comment": "",
        "score": 0,
    }
    if price_series is None or price_series.empty:
        info["comment"] = "No price history from API."
        return info

    close = price_series
    last_price = float(close.iloc[-1])
    rsi_val = compute_rsi(close, period=14)
    ema50 = compute_ema(close, span=50)
    ema200 = compute_ema(close, span=200)

    info["rsi"] = rsi_val
    info["ema50"] = ema50
    info["ema200"] = ema200

    # Trend classification
    if np.isnan(ema200):
        trend = "Sideways / Short History"
    elif last_price > ema200 * 1.02 and ema50 > ema200:
        trend = "Uptrend"
    elif abs(last_price - ema200) / ema200 <= 0.03:
        trend = "Sideways"
    else:
        trend = "Downtrend"
    info["trend"] = trend

    # Simple dip / FOMO logic
    comment_parts = []

    # RSI logic
    if rsi_val <= 35:
        signal = "Buy on Dips (Oversold)"
        score = 90
        comment_parts.append("Price in oversold zone; good area to accumulate in grids.")
    elif 35 < rsi_val <= 55:
        signal = "Accumulation Zone"
        score = 80
        comment_parts.append("Healthy RSI; staggered buying with grids is sensible.")
    elif 55 < rsi_val <= 70:
        signal = "Hold / Wait for Dip"
        score = 60
        comment_parts.append("Momentum is up; wait for pullback before fresh grids.")
    else:
        signal = "Avoid New Buys (Overbought)"
        score = 40
        comment_parts.append("RSI is overheated; avoid chasing, set wider lower grids.")

    # Trend bonus / penalty
    if trend == "Uptrend":
        score += 10
        comment_parts.append("Higher time-frame trend is positive.")
    elif trend == "Downtrend":
        score -= 15
        comment_parts.append("Macro trend weak; use very conservative grids.")

    score = max(0, min(100, score))
    info["signal"] = signal
    info["comment"] = " ".join(comment_parts)
    info["score"] = int(score)
    info["last_price"] = last_price
    return info


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


def check_bot_budget(num_bots: int, per_bot: float) -> dict:
    total = num_bots * per_bot
    ok = True
    msgs = []
    if num_bots > MAX_BOTS:
        ok = False
        msgs.append(f"❌ Max {MAX_BOTS} bots allowed.")
    if per_bot > MAX_PER_BOT:
        ok = False
        msgs.append(f"❌ Per bot limit is {MAX_PER_BOT} USDT. You entered {per_bot:.2f}.")
    if total > MAX_TOTAL:
        ok = False
        msgs.append(f"❌ Total allocation cannot exceed {MAX_TOTAL} USDT. You entered {total:.2f}.")
    if ok:
        msgs.append(f"✅ Budget OK: {num_bots} bot(s) × {per_bot:.2f} USDT = {total:.2f} USDT.")
    return {"ok": ok, "total": total, "messages": msgs}


def estimate_grid_cycle_profit(lower: float, upper: float, grids: int, capital: float) -> dict:
    """
    Very rough approximation of full up-move profit in a classic buy-low/sell-high grid.
    Assumes equal capital per grid and one round-trip per level.
    """
    if lower <= 0 or upper <= lower or grids <= 0 or capital <= 0:
        return {
            "gross_profit": np.nan,
            "roi_pct": np.nan,
            "avg_price": np.nan,
        }
    price_range = upper - lower
    avg_price = (upper + lower) / 2
    # Theoretical ROI approx for 1 full swing through range:
    roi_pct = (price_range / avg_price) * 2 * 100  # 2x for multiple scalps
    gross_profit = capital * roi_pct / 100.0
    return {
        "gross_profit": gross_profit,
        "roi_pct": roi_pct,
        "avg_price": avg_price,
    }


# ------------------ TELEGRAM HELPERS ------------------

def send_telegram_message(token: str, chat_id: str, text: str):
    if not token or not chat_id:
        return None
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.get(url, params={"chat_id": chat_id, "text": text}, timeout=10)
        return r.json()
    except Exception:
        return None


def maybe_send_dip_alerts(df_signals: pd.DataFrame, usdt_inr: float):
    """Send Telegram alerts when coins enter Buy on Dips / RSI<=35. One alert per coin per day."""
    if not st.session_state.get("alerts_enabled", False):
        return

    token = st.session_state.get("tg_token", "")
    chat_id = st.session_state.get("tg_chat_id", "")
    if not token or not chat_id:
        return

    last_alerts = st.session_state.get("last_alerts", {}) or {}
    today_str = datetime.now().strftime("%Y-%m-%d")
    alerted_coins = []

    for _, r in df_signals.iterrows():
        symbol = r.get("symbol", "")
        rsi = r.get("rsi", np.nan)
        signal = r.get("signal", "")
        price = r.get("price", np.nan)

        if symbol == "" or np.isnan(rsi) or np.isnan(price):
            continue

        # Dip condition
        if rsi <= 35 or "Buy on Dips" in str(signal):
            last_for_coin = last_alerts.get(symbol)
            if last_for_coin == today_str:
                continue  # already alerted today

            price_inr = price * usdt_inr
            msg = (
                f"🪙 Dip Alert for {symbol}\n"
                f"Price: {price:.2f} USDT ({fmt_inr_compact(price_inr)})\n"
                f"RSI(14): {rsi:.1f}\n"
                f"Trend: {r.get('trend','Unknown')}\n"
                f"Signal: {signal}\n\n"
                f"Bias: Good zone for **grid accumulation**, stick to max 50 USDT per bot & total 100 USDT."
            )
            send_telegram_message(token, chat_id, msg)
            last_alerts[symbol] = today_str
            alerted_coins.append(symbol)

    if alerted_coins:
        st.caption("📤 Telegram dip alerts sent for: " + ", ".join(alerted_coins))

    st.session_state["last_alerts"] = last_alerts


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
        info = trend_and_signal(price_series, symbol)
        p = prices.get(symbol, {})
        info["price"] = p.get("price", np.nan)
        info["change_24h"] = p.get("change_24h", np.nan)
        info["price_inr"] = info["price"] * usdt_inr if not np.isnan(info["price"]) else np.nan
        rows.append(info)

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

    st.markdown("---")

    budget_info = check_bot_budget(num_bots, per_bot_capital)
    total_capital = budget_info["total"]

    if budget_info["ok"]:
        st.markdown(f"<span class='ok-badge'>{budget_info['messages'][-1]}</span>", unsafe_allow_html=True)
        if len(budget_info["messages"]) > 1:
            for m in budget_info["messages"][:-1]:
                st.caption(m)
    else:
        for m in budget_info["messages"]:
            st.markdown(f"<span class='warn-badge'>{m}</span>", unsafe_allow_html=True)
        st.stop()

    # Capital per bot & per grid
    per_grid_capital = per_bot_capital / grid_count if grid_count > 0 else 0.0
    grid_width = upper_price - lower_price
    per_grid_price_gap = grid_width / grid_count if grid_count > 0 else 0.0

    est = estimate_grid_cycle_profit(lower_price, upper_price, grid_count, per_bot_capital)

    st.markdown("### 📌 Bot Summary (per Bot)")

    summary_rows = [
        {"Metric": "Coin", "Value": coin},
        {"Metric": "Capital per Bot", "Value": fmt_usdt(per_bot_capital)},
        {"Metric": "Capital per Bot (INR approx)", "Value": fmt_inr_compact(per_bot_capital * usdt_inr)},
        {"Metric": "Number of Grids", "Value": f"{grid_count}"},
        {"Metric": "Grid Price Range", "Value": f"{lower_price:.2f} – {upper_price:.2f} USDT"},
        {"Metric": "Average Price in Range", "Value": f"{est['avg_price']:.2f} USDT" if not np.isnan(est["avg_price"]) else "-"},
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

    if np.isnan(est["gross_profit"]):
        st.warning("Check that lower < upper, grids > 0, and capital > 0.")
    else:
        profit_table = pd.DataFrame(
            [
                {
                    "Metric": "Approx. Gross Profit for 1 Full Range Cycle",
                    "Value": fmt_usdt(est["gross_profit"]),
                },
                {
                    "Metric": "Approx. Gross Profit (INR)",
                    "Value": fmt_inr_compact(est["gross_profit"] * usdt_inr),
                },
                {
                    "Metric": "Approx. ROI per Full Range Cycle",
                    "Value": fmt_pct(est["roi_pct"]),
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

    # Range sanity
    if upper_price <= lower_price:
        comments.append("❌ Upper price must be **greater** than lower price. Adjust the range.")
    else:
        width_pct = (upper_price - lower_price) / lower_price * 100 if lower_price > 0 else 0
        if width_pct < 5:
            comments.append("Range is very **tight** (<5%). Good for scalp trading but more sensitive to whipsaws.")
        elif width_pct < 25:
            comments.append("Range is **medium width**; decent balance of safety and opportunity.")
        else:
            comments.append("Range is **very wide**; safer but profits per move may be slower to realize.")

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
        "**Buy on Dips / Oversold** zones (RSI ≤ 35)."
    )

    alerts_enabled = st.checkbox(
        "Enable Telegram Dip Alerts (BTC / ETH / SOL)",
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
                "✅ Test message from AI Crypto Grid Helper – dip alerts configured.",
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
