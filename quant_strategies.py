"""Quantitative trading strategies for NASDAQ/NYSE tickers.

The design takes inspiration from Microsoft's `RD-Agent` project which pairs
technical factor libraries with regime detection and portfolio construction.
While this module remains lightweight, it mirrors that architecture by:

* Building a rich feature set for every ticker (trend, momentum, volatility,
  and volume signals).
* Detecting market regimes from the S&P 500 proxy (SPY) to adapt strategy
  exposure.
* Evaluating a diverse library of base strategies plus a risk-parity ensemble.
* Reporting advanced performance analytics (Sortino, Calmar, beta, turnover,
  exposure) to help quants judge robustness.

Each strategy emits a daily position vector (long/flat). Positions get
translated into return streams which feed the performance attribution layer.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import talib
import yfinance as yf

TRADING_DAYS_PER_YEAR = 252


def _kmeans_fit_predict(
    data: np.ndarray,
    n_clusters: int,
    n_init: int = 10,
    max_iter: int = 100,
    random_state: int | None = 42,
) -> np.ndarray:
    """Lightweight KMeans implementation to avoid the sklearn dependency."""

    if data.size == 0:
        return np.zeros(0, dtype=int)

    n_samples = data.shape[0]
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    n_clusters = min(n_clusters, n_samples)

    rng = np.random.default_rng(random_state)
    best_inertia = np.inf
    best_labels = np.zeros(n_samples, dtype=int)

    for init_idx in range(max(1, n_init)):
        if n_samples == n_clusters:
            centers = data.copy()
        else:
            indices = rng.choice(n_samples, size=n_clusters, replace=False)
            centers = data[indices].copy()

        for _ in range(max_iter):
            distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
            labels = distances.argmin(axis=1)

            new_centers = centers.copy()
            for cluster_id in range(n_clusters):
                members = data[labels == cluster_id]
                if members.size == 0:
                    # Re-seed empty clusters with a random sample to maintain diversity.
                    new_centers[cluster_id] = data[rng.integers(0, n_samples)]
                else:
                    new_centers[cluster_id] = members.mean(axis=0)

            if np.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers

        distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        inertia = float((distances[np.arange(n_samples), labels] ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return best_labels


@dataclass
class StrategyPerformance:
    """Container for the performance statistics of a trading strategy."""

    ticker: str
    strategy: str
    description: str
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    win_rate: float
    alpha_vs_spy: float
    beta: float
    information_ratio: float
    turnover: float
    exposure: float
    cumulative_return: float

    def to_dict(self) -> Dict[str, float]:
        """Return a serialisable representation."""

        return asdict(self)


@dataclass
class StrategyDefinition:
    """Metadata describing a trading strategy."""

    name: str
    description: str
    generator: Callable[[pd.DataFrame, "MarketContext"], pd.Series]


@dataclass
class MarketContext:
    """Shared market information inspired by RD-Agent's regime module."""

    benchmark_prices: pd.DataFrame
    benchmark_returns: pd.Series
    regimes: pd.Series
    realized_vol: pd.Series


def fetch_price_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Download OHLCV data for a ticker using yfinance."""

    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False)
    if df.empty:
        raise ValueError(f"No price data returned for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level="Ticker")
        except (KeyError, ValueError):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(0)
    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Augment the price data with all indicators used across strategies."""

    out = df.copy()
    close = out["Close"].values
    high = out["High"].values
    low = out["Low"].values
    volume = out["Volume"].values.astype(float)

    out["EMA9"] = talib.EMA(close, timeperiod=9)
    out["EMA20"] = talib.EMA(close, timeperiod=20)
    out["EMA50"] = talib.EMA(close, timeperiod=50)
    out["EMA150"] = talib.EMA(close, timeperiod=150)
    out["RSI14"] = talib.RSI(close, timeperiod=14)
    out["ADX14"] = talib.ADX(high, low, close, timeperiod=14)
    out["PLUS_DI"] = talib.PLUS_DI(high, low, close, timeperiod=14)
    out["MINUS_DI"] = talib.MINUS_DI(high, low, close, timeperiod=14)

    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    out["BB_UPPER"] = upper
    out["BB_MIDDLE"] = middle
    out["BB_LOWER"] = lower

    macd, macd_signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    out["MACD"] = macd
    out["MACD_SIGNAL"] = macd_signal
    out["ATR14"] = talib.ATR(high, low, close, timeperiod=14)

    out["RETURNS"] = out["Close"].pct_change().fillna(0.0)
    out["RET_5"] = out["Close"].pct_change(5)
    out["RET_21"] = out["Close"].pct_change(21)
    out["RET_63"] = out["Close"].pct_change(63)
    out["HIGH_55"] = pd.Series(high, index=out.index).rolling(window=55).max()
    out["LOW_55"] = pd.Series(low, index=out.index).rolling(window=55).min()
    out["VOLUME_SMA20"] = talib.SMA(volume, timeperiod=20)
    out["VOLUME_SMA50"] = talib.SMA(volume, timeperiod=50)
    out["VOLATILITY_21"] = out["RETURNS"].rolling(21).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    out["VOLATILITY_63"] = out["RETURNS"].rolling(63).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    out.dropna(inplace=True)
    return out


def _hold_until_exit(long_condition: pd.Series, exit_condition: pd.Series) -> pd.Series:
    """Utility to transform entry/exit booleans into a position series."""

    position = []
    current = 0
    for enter, exit_ in zip(long_condition, exit_condition):
        if enter:
            current = 1
        elif exit_:
            current = 0
        position.append(current)
    return pd.Series(position, index=long_condition.index, dtype=float)


def _align_series(series: pd.Series, index: pd.Index, fill_value: float = 0.0) -> pd.Series:
    """Align a series to a target index with forward filling."""

    return series.reindex(index).ffill().fillna(fill_value)


def calculate_realized_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Annualised realised volatility."""

    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def detect_market_regimes(benchmark_returns: pd.Series) -> pd.Series:
    """Cluster SPY return/volatility features to infer market regimes."""

    features = pd.DataFrame(
        {
            "ret_5": benchmark_returns.rolling(5).sum(),
            "ret_21": benchmark_returns.rolling(21).sum(),
            "vol_21": calculate_realized_volatility(benchmark_returns, window=21),
        }
    ).dropna()

    if features.empty:
        return pd.Series(0, index=benchmark_returns.index, dtype=int)

    labels = _kmeans_fit_predict(features.values, n_clusters=2, n_init=10, random_state=42)
    feature_frame = features.copy()
    feature_frame["cluster"] = labels
    cluster_returns = feature_frame.groupby("cluster")["ret_21"].mean()
    risk_on_label = cluster_returns.idxmax()

    regimes = pd.Series(0, index=benchmark_returns.index, dtype=int)
    regimes.loc[features.index] = (labels == risk_on_label).astype(int)
    regimes = regimes.ffill().fillna(0).astype(int)
    return regimes


def build_market_context(period: str = "2y") -> MarketContext:
    """Prepare benchmark-derived context shared across strategies."""

    benchmark_prices = prepare_indicators(fetch_price_data("SPY", period=period))
    benchmark_returns = benchmark_prices["RETURNS"].copy()
    realized_vol = calculate_realized_volatility(benchmark_returns)
    regimes = detect_market_regimes(benchmark_returns)
    return MarketContext(
        benchmark_prices=benchmark_prices,
        benchmark_returns=benchmark_returns,
        regimes=regimes,
        realized_vol=realized_vol,
    )


def strategy_regime_trend(price_df: pd.DataFrame, context: MarketContext) -> pd.Series:
    """Trend following that only risks capital in risk-on regimes."""

    regimes = _align_series(context.regimes.astype(float), price_df.index)
    long_condition = (
        (regimes > 0.5)
        & (price_df["EMA50"] > price_df["EMA150"])
        & (price_df["Close"] > price_df["EMA20"])
        & (price_df["ADX14"] > 20)
        & (price_df["PLUS_DI"] > price_df["MINUS_DI"])
        & (price_df["RSI14"].between(45, 70))
    )
    exit_condition = (price_df["Close"] < price_df["EMA50"]) | (regimes < 0.5)
    return _hold_until_exit(long_condition, exit_condition)


def strategy_mean_reversion(price_df: pd.DataFrame, context: MarketContext) -> pd.Series:
    """Bollinger pullbacks prioritising volatility compression and liquidity."""

    base_vol = context.realized_vol.dropna()
    fill_value = float(base_vol.iloc[0]) if not base_vol.empty else 0.0
    vol_filter = _align_series(context.realized_vol, price_df.index, fill_value=fill_value)
    long_condition = (
        (price_df["Close"] < price_df["BB_LOWER"])
        & (price_df["RSI14"] < 35)
        & (price_df["Volume"] > 0.8 * price_df["VOLUME_SMA20"])
        & (vol_filter < vol_filter.rolling(10).median())
    )
    exit_condition = (price_df["Close"] >= price_df["BB_MIDDLE"]) | (price_df["RSI14"] > 55)
    return _hold_until_exit(long_condition, exit_condition)


def strategy_momentum_breakout(price_df: pd.DataFrame, context: MarketContext) -> pd.Series:
    """55-day breakout with MACD confirmation and volume thrust."""

    regimes = _align_series(context.regimes.astype(float), price_df.index)
    volume_ratio = price_df["Volume"] / price_df["VOLUME_SMA20"]
    breakout = (
        (regimes > 0.5)
        & (price_df["Close"] > price_df["HIGH_55"].shift(1))
        & (price_df["MACD"] > price_df["MACD_SIGNAL"])
        & (volume_ratio > 1.1)
    )
    exit_condition = (
        (price_df["Close"] < price_df["EMA20"])
        | (price_df["MACD"] < price_df["MACD_SIGNAL"])
        | ((price_df["Close"] - price_df["Close"].rolling(window=5).max()) < -1.5 * price_df["ATR14"])
    )
    return _hold_until_exit(breakout, exit_condition)


def strategy_pullback_momentum(price_df: pd.DataFrame, context: MarketContext) -> pd.Series:
    """Ride medium-term momentum and add on tactical pullbacks."""

    regimes = _align_series(context.regimes.astype(float), price_df.index)
    momentum_strength = price_df["RET_63"] > 0.12
    shallow_pullback = (price_df["Close"] > price_df["EMA50"]) & (price_df["Close"] < price_df["EMA20"])
    long_condition = (
        (regimes > 0.5)
        & momentum_strength
        & shallow_pullback
        & (price_df["RSI14"] < 60)
        & (price_df["Close"] > price_df["EMA9"])
    )
    exit_condition = (price_df["Close"] < price_df["EMA50"]) | (price_df["RSI14"] > 70)
    return _hold_until_exit(long_condition, exit_condition)


def strategy_volatility_carry(price_df: pd.DataFrame, context: MarketContext) -> pd.Series:
    """Harvest equity carry when realised volatility compresses."""

    base_vol = context.realized_vol.dropna()
    fill_value = float(base_vol.iloc[0]) if not base_vol.empty else 0.0
    vol_filter = _align_series(context.realized_vol, price_df.index, fill_value=fill_value)
    falling_vol = vol_filter < vol_filter.rolling(5).mean()
    momentum_ok = price_df["RET_21"] > 0
    long_condition = (
        falling_vol
        & momentum_ok
        & (price_df["Close"] > price_df["EMA20"])
        & (price_df["EMA20"] > price_df["EMA50"])
    )
    exit_condition = (price_df["Close"] < price_df["EMA20"]) | (~falling_vol)
    return _hold_until_exit(long_condition, exit_condition)


STRATEGY_DEFINITIONS: List[StrategyDefinition] = [
    StrategyDefinition(
        name="Regime-Aware Trend",
        description="EMA trend following gated by RD-Agent style regimes.",
        generator=strategy_regime_trend,
    ),
    StrategyDefinition(
        name="Adaptive Mean Reversion",
        description="Bollinger pullbacks with volatility compression and volume support.",
        generator=strategy_mean_reversion,
    ),
    StrategyDefinition(
        name="Momentum Breakout MACD",
        description="55-day breakout with MACD confirmation and volume thrust.",
        generator=strategy_momentum_breakout,
    ),
    StrategyDefinition(
        name="Momentum Pullback",
        description="Buy strong momentum names on EMA pullbacks while risk-on.",
        generator=strategy_pullback_momentum,
    ),
    StrategyDefinition(
        name="Volatility Carry",
        description="Long exposure when realised volatility falls and momentum is positive.",
        generator=strategy_volatility_carry,
    ),
]


def _summarise_returns(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    positions: pd.Series | None,
    ticker: str,
    strategy_name: str,
    description: str,
) -> StrategyPerformance:
    strategy_returns = strategy_returns.dropna()
    if strategy_returns.empty:
        raise ValueError("No returns to evaluate.")

    n_periods = len(strategy_returns)
    cumulative_return = float((1 + strategy_returns).prod() - 1)
    cagr = float((1 + cumulative_return) ** (TRADING_DAYS_PER_YEAR / n_periods) - 1)
    volatility = float(strategy_returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = 0.0 if volatility == 0 else float(
        strategy_returns.mean() * np.sqrt(TRADING_DAYS_PER_YEAR) / volatility
    )

    downside = strategy_returns.copy()
    downside[downside > 0] = 0
    downside_std = float((-downside).std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sortino = 0.0 if downside_std == 0 else float(
        strategy_returns.mean() * np.sqrt(TRADING_DAYS_PER_YEAR) / downside_std
    )

    equity_curve = (1 + strategy_returns).cumprod()
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = float(drawdown.min())
    calmar = float("nan")
    if max_drawdown < 0:
        calmar = cagr / abs(max_drawdown)

    if positions is not None:
        aligned_positions = positions.reindex(strategy_returns.index).fillna(0.0)
        active_days = aligned_positions > 0
        wins = (strategy_returns > 0) & active_days
        win_rate = float(wins.sum() / max(1, active_days.sum()))
        turnover = float(0.5 * np.abs(aligned_positions.diff().fillna(0.0)).sum())
        exposure = float(aligned_positions.mean())
    else:
        win_rate = float((strategy_returns > 0).mean())
        turnover = float("nan")
        exposure = float("nan")

    aligned_benchmark = benchmark_returns.reindex(strategy_returns.index).fillna(0.0)
    benchmark_cumulative = float((1 + aligned_benchmark).prod() - 1)
    benchmark_cagr = float(
        (1 + benchmark_cumulative) ** (TRADING_DAYS_PER_YEAR / n_periods) - 1
    )
    alpha_vs_spy = cagr - benchmark_cagr

    variance = float(np.var(aligned_benchmark))
    if variance == 0:
        beta = 0.0
    else:
        covariance = float(np.cov(strategy_returns, aligned_benchmark)[0, 1])
        beta = covariance / variance

    tracking_error = float(
        (strategy_returns - aligned_benchmark).std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    information_ratio = 0.0 if tracking_error == 0 else float(
        (strategy_returns.mean() - aligned_benchmark.mean())
        * np.sqrt(TRADING_DAYS_PER_YEAR)
        / tracking_error
    )

    return StrategyPerformance(
        ticker=ticker,
        strategy=strategy_name,
        description=description,
        cagr=cagr,
        volatility=volatility,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        alpha_vs_spy=alpha_vs_spy,
        beta=beta,
        information_ratio=information_ratio,
        turnover=turnover,
        exposure=exposure,
        cumulative_return=cumulative_return,
    )


def compute_performance(
    df: pd.DataFrame,
    positions: pd.Series,
    benchmark_returns: pd.Series,
    ticker: str,
    strategy_name: str,
    description: str,
) -> Tuple[StrategyPerformance, pd.Series, pd.Series]:
    """Calculate risk metrics for a strategy and return the return stream."""

    aligned_positions = positions.reindex(df.index).ffill().fillna(0.0)
    strategy_returns = df["RETURNS"] * aligned_positions
    performance = _summarise_returns(
        strategy_returns,
        benchmark_returns,
        aligned_positions,
        ticker,
        strategy_name,
        description,
    )
    return performance, strategy_returns, aligned_positions


def build_strategy_ensemble(
    price_df: pd.DataFrame,
    strategy_returns: Dict[str, pd.Series],
    strategy_positions: Dict[str, pd.Series],
    context: MarketContext,
    ticker: str,
) -> StrategyPerformance | None:
    """Combine base strategies using inverse-volatility weights."""

    if not strategy_returns:
        return None

    returns_df = pd.DataFrame(strategy_returns).reindex(price_df.index).fillna(0.0)
    valid_cols = [col for col in returns_df.columns if returns_df[col].std(ddof=0) > 0]
    if not valid_cols:
        return None

    returns_df = returns_df[valid_cols]
    cov_matrix = returns_df.cov()
    if cov_matrix.isnull().values.any():
        return None

    vol = np.sqrt(np.diag(cov_matrix.values))
    positive = vol > 0
    if not positive.any():
        return None

    vol = vol[positive]
    weights = 1 / vol
    weights = weights / weights.sum()
    selected_cols = [col for col, keep in zip(returns_df.columns, positive) if keep]

    weighted_returns = returns_df[selected_cols].mul(weights, axis=1).sum(axis=1)
    positions_df = (
        pd.DataFrame({name: strategy_positions[name] for name in selected_cols})
        .reindex(price_df.index)
        .ffill()
        .fillna(0.0)
    )
    weighted_positions = positions_df.mul(weights, axis=1).sum(axis=1)

    performance = _summarise_returns(
        weighted_returns,
        context.benchmark_returns,
        weighted_positions,
        ticker,
        "Ensemble Risk Parity",
        "Inverse-volatility blend of base systems inspired by RD-Agent portfolios.",
    )
    return performance


def run_quant_strategies(
    ticker: str,
    period: str = "2y",
    market_context: MarketContext | None = None,
) -> List[StrategyPerformance]:
    """Fetch data, compute indicators, and evaluate all strategies for a ticker."""

    price_df = prepare_indicators(fetch_price_data(ticker, period=period))
    context = market_context or build_market_context(period=period)

    results: List[StrategyPerformance] = []
    strategy_returns: Dict[str, pd.Series] = {}
    strategy_positions: Dict[str, pd.Series] = {}

    for definition in STRATEGY_DEFINITIONS:
        positions = definition.generator(price_df, context)
        performance, returns, aligned_positions = compute_performance(
            price_df,
            positions,
            context.benchmark_returns,
            ticker,
            definition.name,
            definition.description,
        )
        results.append(performance)
        strategy_returns[definition.name] = returns
        strategy_positions[definition.name] = aligned_positions

    ensemble_performance = build_strategy_ensemble(
        price_df,
        strategy_returns,
        strategy_positions,
        context,
        ticker,
    )
    if ensemble_performance is not None:
        results.append(ensemble_performance)

    return results


def rank_strategies_across_tickers(tickers: Iterable[str], period: str = "2y") -> pd.DataFrame:
    """Run the full suite of strategies across a list of tickers and rank by alpha."""

    performances: List[StrategyPerformance] = []
    context = build_market_context(period=period)
    for ticker in tickers:
        try:
            performances.extend(
                run_quant_strategies(ticker, period=period, market_context=context)
            )
        except ValueError as exc:
            print(f"⚠️ Skipping {ticker}: {exc}")
            continue

    if not performances:
        return pd.DataFrame()

    df = pd.DataFrame([perf.to_dict() for perf in performances])
    df.sort_values(by="alpha_vs_spy", ascending=False, inplace=True)
    return df.reset_index(drop=True)


__all__ = [
    "StrategyPerformance",
    "run_quant_strategies",
    "rank_strategies_across_tickers",
    "STRATEGY_DEFINITIONS",
]
