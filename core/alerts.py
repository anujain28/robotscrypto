"""
Telegram alert helpers for the AI Crypto Grid Helper.

This module is UI-agnostic. It does **not** use Streamlit directly.
You pass in your signals DataFrame / records & it returns messages + updated state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Iterable, Any, Optional

import numpy as np
import pandas as pd
import requests


@dataclass
class AlertMessage:
    symbol: str
    text: str


def send_telegram_message(
    token: str,
    chat_id: str,
    text: str,
    timeout: int = 10,
) -> Optional[Dict[str, Any]]:
    """
    Low-level Telegram send function.
    """
    if not token or not chat_id:
        return None
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.get(url, params={"chat_id": chat_id, "text": text}, timeout=timeout)
        return r.json()
    except Exception:
        return None


def build_dip_alert_text(
    symbol: str,
    price_usdt: float,
    price_inr: float,
    rsi: float,
    trend: str,
    signal: str,
) -> str:
    """
    Compose a human-friendly dip alert message.
    """
    return (
        f"🪙 Dip Alert for {symbol}\n"
        f"Price: {price_usdt:.2f} USDT (≈ ₹{price_inr:,.2f})\n"
        f"RSI(14): {rsi:.1f}\n"
        f"Trend: {trend}\n"
        f"Signal: {signal}\n\n"
        f"Bias: Good zone for **grid accumulation**. "
        f"Keep per-bot capital small and respect overall risk limits."
    )


def prepare_dip_alerts(
    df_signals: pd.DataFrame,
    usdt_inr: float,
    last_alerts: Dict[str, str],
    today_str: Optional[str] = None,
    rsi_threshold: float = 35.0,
) -> Tuple[List[AlertMessage], Dict[str, str]]:
    """
    Determine which coins need dip alerts for today.

    Parameters
    ----------
    df_signals : pd.DataFrame
        Must have columns: symbol, rsi, signal, price (USDT).
    usdt_inr : float
        Conversion rate from USDT to INR.
    last_alerts : dict
        Mapping symbol -> YYYY-MM-DD of last alert sent.
    today_str : str, optional
        If None, uses today's date (local).
    rsi_threshold : float
        RSI level below or equal to which a dip alert can be triggered.

    Returns
    -------
    (alerts, new_state)
        alerts : list[AlertMessage]
        new_state : updated mapping symbol -> YYYY-MM-DD
    """
    if today_str is None:
        today_str = datetime.now().strftime("%Y-%m-%d")

    last_alerts = dict(last_alerts or {})
    alerts: List[AlertMessage] = []

    if df_signals is None or df_signals.empty:
        return alerts, last_alerts

    required_cols = {"symbol", "rsi", "signal", "price"}
    if not required_cols.issubset(df_signals.columns):
        return alerts, last_alerts

    for _, row in df_signals.iterrows():
        symbol = str(row.get("symbol", "")).strip()
        if not symbol:
            continue

        rsi = float(row.get("rsi", np.nan))
        signal = str(row.get("signal", ""))
        price = float(row.get("price", np.nan))
        trend = str(row.get("trend", "Unknown"))

        if np.isnan(rsi) or np.isnan(price):
            continue

        # Dip condition: RSI <= threshold or explicit "Buy on Dips" signal
        if not (rsi <= rsi_threshold or "Buy on Dips" in signal):
            continue

        # Only once per coin per day
        if last_alerts.get(symbol) == today_str:
            continue

        price_inr = price * usdt_inr
        text = build_dip_alert_text(symbol, price, price_inr, rsi, trend, signal)
        alerts.append(AlertMessage(symbol=symbol, text=text))
        last_alerts[symbol] = today_str

    return alerts, last_alerts
