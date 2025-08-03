
import pandas as pd
import yfinance as yf
import talib
from datetime import datetime, timedelta

def get_fundamentally_strong_stocks():
    return ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "ADBE"]

def apply_technical_filters(tickers):
    screened = []
    for ticker in tickers:
        df = yf.download(ticker, period="6mo", interval="1d")
        if df.empty or len(df) < 100:
            continue

        df['SMA50'] = talib.SMA(df['Close'], timeperiod=50)
        df['SMA200'] = talib.SMA(df['Close'], timeperiod=200)
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['Volume_SMA20'] = talib.SMA(df['Volume'], timeperiod=20)

        latest = df.iloc[-1]

        conditions = (
            latest['SMA50'] > latest['SMA200'] and
            latest['Close'] > latest['SMA50'] and
            latest['RSI'] < 60 and
            latest['Volume'] > latest['Volume_SMA20']
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

def main():
    tickers = get_fundamentally_strong_stocks()
    results_df = apply_technical_filters(tickers)
    results_df.to_csv("screener_results.csv", index=False)
    print("✅ Screener results saved to screener_results.csv")

if __name__ == "__main__":
    main()
