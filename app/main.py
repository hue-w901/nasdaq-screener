"""FastAPI application that exposes the quant strategies backend."""

from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from nasdaq_screener import (
    DEFAULT_PERIOD,
    get_fundamentally_strong_stocks,
    run_screener,
)
from quant_strategies import rank_strategies_across_tickers, run_quant_strategies

app = FastAPI(
    title="NASDAQ Quant Screener",
    description=(
        "API that screens fundamentally strong NASDAQ/NYSE tickers and ranks "
        "systematic strategies for alpha generation."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _parse_ticker_list(raw: str | None) -> List[str] | None:
    if raw is None:
        return None
    tickers = [ticker.strip().upper() for ticker in raw.split(",") if ticker.strip()]
    return tickers or None


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the single-page frontend."""

    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Frontend not found")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/screener")
async def screener(
    tickers: str | None = Query(
        default=None,
        description="Comma separated ticker symbols to screen. Uses defaults when omitted.",
    ),
    period: str = Query(default=DEFAULT_PERIOD, description="Lookback period accepted by yfinance"),
) -> dict[str, object]:
    ticker_list = _parse_ticker_list(tickers)
    if ticker_list is None:
        ticker_list = get_fundamentally_strong_stocks()

    results_df = run_screener(ticker_list, period=period)
    return {
        "tickers": ticker_list,
        "period": period,
        "results": results_df.to_dict(orient="records"),
    }


@app.get("/api/strategies/rank")
async def rank_strategies(
    tickers: str | None = Query(
        default=None,
        description="Comma separated ticker symbols to include in the ranking.",
    ),
    period: str = Query(default="2y", description="Lookback period accepted by yfinance"),
) -> dict[str, object]:
    ticker_list = _parse_ticker_list(tickers)
    if ticker_list is None:
        ticker_list = get_fundamentally_strong_stocks()

    ranking_df = rank_strategies_across_tickers(ticker_list, period=period)
    return {
        "tickers": ticker_list,
        "period": period,
        "results": ranking_df.to_dict(orient="records"),
    }


@app.get("/api/strategies/{ticker}")
async def strategies_for_ticker(
    ticker: str,
    period: str = Query(default="2y", description="Lookback period for strategy evaluation"),
) -> dict[str, object]:
    try:
        performances = run_quant_strategies(ticker.upper(), period=period)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "ticker": ticker.upper(),
        "period": period,
        "results": [performance.to_dict() for performance in performances],
    }


__all__ = ["app"]
