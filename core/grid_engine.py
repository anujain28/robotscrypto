"""
Grid planning & risk utilities for the AI Crypto Grid Helper.

Independent of Streamlit so it can be reused in other contexts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class BudgetCheckResult:
    ok: bool
    total: float
    messages: List[str]


@dataclass
class GridProfitEstimate:
    gross_profit: float
    roi_pct: float
    avg_price: float


def check_bot_budget(
    num_bots: int,
    per_bot: float,
    max_bots: int,
    max_per_bot: float,
    max_total: float,
) -> BudgetCheckResult:
    """
    Validate allocation across bots.

    Parameters
    ----------
    num_bots : int
        Number of bots user wants to run.
    per_bot : float
        Capital per bot (USDT).
    max_bots : int
        Hard limit on number of bots.
    max_per_bot : float
        Max allowed capital per bot (USDT).
    max_total : float
        Max total capital for all bots (USDT).

    Returns
    -------
    BudgetCheckResult
        ok flag, total capital, and messages list.
    """
    total = num_bots * per_bot
    ok = True
    msgs: List[str] = []

    if num_bots > max_bots:
        ok = False
        msgs.append(f"❌ Max {max_bots} bots allowed.")
    if per_bot > max_per_bot:
        ok = False
        msgs.append(f"❌ Per bot limit is {max_per_bot} USDT. You entered {per_bot:.2f}.")
    if total > max_total:
        ok = False
        msgs.append(f"❌ Total allocation cannot exceed {max_total} USDT. You entered {total:.2f}.")

    if ok:
        msgs.append(f"✅ Budget OK: {num_bots} bot(s) × {per_bot:.2f} USDT = {total:.2f} USDT.")

    return BudgetCheckResult(ok=ok, total=total, messages=msgs)


def estimate_grid_cycle_profit(
    lower: float,
    upper: float,
    grids: int,
    capital: float,
) -> GridProfitEstimate:
    """
    Very rough approximation of full up-move profit in a classic buy-low/sell-high grid.

    Assumes:
      - Equal capital per grid level.
      - Price traverses the full range once, generating multiple scalps.

    Parameters
    ----------
    lower : float
        Lower bound of grid (USDT).
    upper : float
        Upper bound of grid (USDT).
    grids : int
        Number of grid levels.
    capital : float
        Capital allocated to this grid bot (USDT).

    Returns
    -------
    GridProfitEstimate
        gross_profit (USDT), roi_pct, avg_price.
    """
    if lower <= 0 or upper <= lower or grids <= 0 or capital <= 0:
        return GridProfitEstimate(
            gross_profit=float("nan"),
            roi_pct=float("nan"),
            avg_price=float("nan"),
        )

    price_range = upper - lower
    avg_price = (upper + lower) / 2.0

    # Approximate ROI for a full swing through the grid:
    # range/avg_price gives single-move return; multiply by 2 to account for
    # repeated buy-low/sell-high actions across multiple bands.
    roi_pct = (price_range / avg_price) * 2.0 * 100.0
    gross_profit = capital * roi_pct / 100.0

    return GridProfitEstimate(
        gross_profit=float(gross_profit),
        roi_pct=float(roi_pct),
        avg_price=float(avg_price),
    )


def describe_grid_width(lower: float, upper: float) -> str:
    """
    Human-friendly explanation for how wide the chosen grid range is.
    """
    if lower <= 0 or upper <= lower:
        return "Invalid range; lower must be > 0 and upper > lower."

    width_pct = (upper - lower) / lower * 100.0
    if width_pct < 5:
        return "Range is very tight (<5%). High-frequency scalping, but more sensitive to whipsaws."
    if width_pct < 25:
        return "Range is medium width; balanced risk–reward for a typical grid."
    return "Range is very wide; safer but profits per move may take longer to realize."
