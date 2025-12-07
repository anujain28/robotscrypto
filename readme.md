# 🤖 AI Crypto Grid Helper

Planning-only assistant for Binance-style grid bots focusing on BTC, ETH, and SOL.

## Features

- Live **market overview** using CoinGecko (BTC, ETH, SOL)
- AI-style trend score (RSI, EMA-based)
- Shows **Top 2 safer coins** for grid bots
- **Grid Bot Planner**:
  - Enforces rules: max 2 bots, max 50 USDT per bot, max 100 USDT total
  - Estimates profit for a full grid range cycle
  - Comments on TP/SL and grid width
- **My Bot Notes**: manual tracking for your live bots

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
