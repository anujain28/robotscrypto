"""
Core indicator utilities for the AI Crypto Grid Helper.

This module is intentionally independent of Streamlit so it can be reused
in backtests, CLI tools, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any

import numpy as np
import pandas as pd


@dataclass
class TrendSignal:
    symbol: str
    trend: str
    rsi: float
    ema50: float
    ema200: float
    signal: str
    comment: str
    score: int
    last_price: float


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    """
    Classic RSI implementation using simple moving averages of gains/losses.

    Parameters
    ----------
    series : pd.Series
        Price series (close prices).
    period : int
        RSI lookback period.

    Returns
    -------
    float
        Last RSI value, or NaN if insufficient data.
    """
    if series is None or len(series) < period + 1:
        return float("nan")

    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def compute_ema(series: pd.Series, span: int) -> float:
    """
    Compute exponential moving average.

    Parameters
    ----------
    series : pd.Series
        Price series.
    span : int
        EMA span.

    Returns
    -------
    float
        Last EMA value, or NaN if insufficient data.
    """
    if series is None or len(series) < span:
        return float("nan")
    ema = series.ewm(span=span, adjust=False).mean()
    return float(ema.iloc[-1])


def classify_trend(last_price: float, ema50: float, ema200: float) -> str:
    """
    Rough trend classifier using price vs EMA200 and EMA50 vs EMA200.
    """
    if np.isnan(ema200):
        return "Sideways / Short History"

    if last_price > ema200 * 1.02 and ema50 > ema200:
        return "Uptrend"
    if abs(last_price - ema200) / ema200 <= 0.03:
        return "Sideways"
    return "Downtrend"


def trend_and_signal(price_series: pd.Series, symbol: str) -> TrendSignal:
    """
    Generate high-level signal for a coin based on RSI and EMAs.

    Logic:
      - RSI <= 35  -> Buy on Dips (Oversold)
      - 35–55      -> Accumulation Zone
      - 55–70      -> Hold / Wait for Dip
      - > 70       -> Avoid New Buys (Overbought)
    Trend adds bonus/penalty to score.
    """
    if price_series is None or price_series.empty:
        return TrendSignal(
            symbol=symbol,
            trend="Unknown",
            rsi=float("nan"),
            ema50=float("nan"),
            ema200=float("nan"),
            signal="No data",
            comment="No price history supplied.",
            score=0,
            last_price=float("nan"),
        )

    close = price_series
    last_price = float(close.iloc[-1])
    rsi_val = compute_rsi(close, period=14)
    ema50_val = compute_ema(close, span=50)
    ema200_val = compute_ema(close, span=200)

    trend = classify_trend(last_price, ema50_val, ema200_val)

    # Base classification by RSI
    comment_parts = []
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
        comment_parts.append("Macro trend weak; use conservative grids.")

    score = int(max(0, min(100, score)))
    comment = " ".join(comment_parts)

    return TrendSignal(
        symbol=symbol,
        trend=trend,
        rsi=float(rsi_val),
        ema50=float(ema50_val),
        ema200=float(ema200_val),
        signal=signal,
        comment=comment,
        score=score,
        last_price=float(last_price),
    )


def series_from_coingecko_market_chart(data: Dict[str, Any]) -> pd.Series:
    """
    Utility to convert CoinGecko /market_chart 'prices' into a Series.

    Expects data like:
        { "prices": [[timestamp_ms, price], ...] }
    """
    prices = data.get("prices", [])
    if not prices:
        return pd.Series(dtype=float)

    ts = [datetime.fromtimestamp(p[0] / 1000, tz=timezone.utc) for p in prices]
    vals = [p[1] for p in prices]
    return pd.Series(vals, index=pd.to_datetime(ts))
