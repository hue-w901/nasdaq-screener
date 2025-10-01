
"""Stock screener that pairs with the quant strategy engine."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import pandas as pd
import talib
import yfinance as yf

from quant_strategies import rank_strategies_across_tickers

DEFAULT_PERIOD = "1y"


def get_fundamentally_strong_stocks() -> List[str]:
    return [
        "AAPL",  # Apple - strong free cash flow, ecosystem moat
        "MSFT",  # Microsoft - diversified enterprise/cloud
        "NVDA",  # Nvidia - AI and data center leadership
        "GOOGL",  # Alphabet - advertising + cloud optionality
        "AMZN",  # Amazon - e-commerce and AWS scale
        "META",  # Meta Platforms - high-margin social platforms
        "AVGO",  # Broadcom - semiconductor and software exposure
        "ADBE",  # Adobe - subscription software dominance
        "ASML",  # ASML - lithography monopoly
        "CRM",  # Salesforce - enterprise SaaS leader
        "COST",  # Costco - resilient retail membership model
        "LIN",  # Linde - industrial gases with pricing power
    ]


def apply_technical_filters(
    tickers: Iterable[str],
    period: str = DEFAULT_PERIOD,
) -> pd.DataFrame:
    screened = []
    for ticker in tickers:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=False)
        if df.empty or len(df) < 200:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level="Ticker")
            df.columns = df.columns.get_level_values(0)

        df['SMA50'] = talib.SMA(df['Close'], timeperiod=50)
        df['SMA200'] = talib.SMA(df['Close'], timeperiod=200)
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['Volume_SMA20'] = talib.SMA(df['Volume'], timeperiod=20)

        latest = df.iloc[-1]

        conditions = (
            latest['SMA50'] >= 0.98 * latest['SMA200'] and
            latest['Close'] > latest['SMA50'] and
            40 <= latest['RSI'] <= 75 and
            latest['Volume'] >= 0.6 * latest['Volume_SMA20']
        )

        if conditions:
            screened.append({
                'Ticker': ticker,
                'Close': latest['Close'],
                'SMA50': latest['SMA50'],
                'SMA200': latest['SMA200'],
                'RSI': latest['RSI'],
                'Volume': latest['Volume'],
                'Volume Avg (20d)': latest['Volume_SMA20']
            })
    return pd.DataFrame(screened)


def run_screener(
    tickers: Sequence[str] | None = None,
    period: str = DEFAULT_PERIOD,
) -> pd.DataFrame:
    """Execute the screener for the provided tickers."""

    tickers = list(tickers) if tickers is not None else get_fundamentally_strong_stocks()
    return apply_technical_filters(tickers, period=period)


def main() -> None:
    tickers = get_fundamentally_strong_stocks()
    results_df = apply_technical_filters(tickers)
    if results_df.empty:
        print("⚠️ No tickers passed the technical filters. Skipping strategy ranking.")
        return

    results_df.to_csv("screener_results.csv", index=False)
    print("✅ Screener results saved to screener_results.csv")

    strategy_df = rank_strategies_across_tickers(results_df["Ticker"].tolist(), period="2y")
    if strategy_df.empty:
        print("⚠️ Unable to compute strategy rankings.")
        return

    strategy_df.to_csv("strategy_results.csv", index=False)
    print("✅ Strategy performance saved to strategy_results.csv")

if __name__ == "__main__":
    main()
