#!/usr/bin/env python3
import sys
from dataclasses import dataclass
from typing import Optional, List

import pandas as pd
import yfinance as yf

from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator

SUFFIX = {
    "NSE": ".NS",
    "BSE": ".BO",
    "NASDAQ": "",
    "NYSE": "",
}

# Common shorthand -> official NSE symbols (extend as needed)
ALIASES = {
    "NALCO": "NATIONALUM",
    "VEDANTA": "VEDL",
    "VEDL": "VEDL",
    # add more if you want
}

@dataclass
class IndicatorPack:
    close: pd.Series
    ema20: pd.Series
    sma50: pd.Series
    sma200: pd.Series
    rsi: pd.Series
    macd: pd.Series
    macd_signal: pd.Series
    macd_hist: pd.Series
    stoch_k: pd.Series
    stoch_d: pd.Series
    obv: pd.Series

def nse_lookup(symbol_upper: str) -> Optional[str]:
    """Try to resolve symbol using NSE stock codes (requires nsetools)."""
    try:
        from nsetools import Nse
        codes = Nse().get_stock_codes()  # dict: { 'INFY': 'Infosys Limited', ... }

        if symbol_upper in codes:
            return symbol_upper

        # reverse search by company name substring
        for sym, name in codes.items():
            name_upper = str(name).upper()
            if symbol_upper in name_upper or symbol_upper == sym.upper():
                return sym.upper()

        return None
    except Exception:
        return None

def resolve_ticker_online(symbol: str, exchange: str) -> str:
    ex = exchange.strip().upper()
    if ex not in SUFFIX:
        raise ValueError(f"Exchange '{exchange}' not supported. Use NSE/BSE/NASDAQ/NYSE.")

    base = symbol.strip().upper()
    alias_base = ALIASES.get(base, base)

    bases = {base, alias_base}
    if ex == "NSE":
        official = nse_lookup(base)
        if official:
            bases.add(official)

    candidates = [b + SUFFIX[ex] for b in bases]

    for ticker in candidates:
        try:
            df = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
        except Exception:
            df = None

        if df is not None and not df.empty and "Close" in df and df["Close"].dropna().shape[0] > 0:
            return ticker

    raise ValueError(
        f"Could not resolve ticker for symbol='{symbol}', exchange='{exchange}'. "
        f"Tried: {candidates}"
    )

def download_ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No OHLCV data for '{ticker}'.")
    return df.dropna(subset=["Close"])

def compute_indicators(df: pd.DataFrame) -> IndicatorPack:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    ema20 = EMAIndicator(close, window=20).ema_indicator()
    sma50 = SMAIndicator(close, window=50).sma_indicator()
    sma200 = SMAIndicator(close, window=200).sma_indicator()

    rsi = RSIIndicator(close, window=14).rsi()

    macd_calc = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd_calc.macd()
    macd_sig = macd_calc.macd_signal()
    macd_hist = macd_calc.macd_diff()

    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    stoch_k = stoch.stoch()      # %K
    stoch_d = stoch.stoch_signal()  # %D

    obv = OnBalanceVolumeIndicator(close, vol).on_balance_volume()

    return IndicatorPack(
        close=close,
        ema20=ema20,
        sma50=sma50,
        sma200=sma200,
        rsi=rsi,
        macd=macd_line,
        macd_signal=macd_sig,
        macd_hist=macd_hist,
        stoch_k=stoch_k,
        stoch_d=stoch_d,
        obv=obv,
    )

def fib_levels(high: float, low: float) -> dict:
    diff = high - low
    return {
        "0%": low,
        "38.2%": high - diff * 0.382,
        "50%": high - diff * 0.5,
        "61.8%": high - diff * 0.618,
        "100%": high,
    }

def pick_swing(df: pd.DataFrame, lookback: int = 60) -> tuple:
    """Pick recent swing using lookback window; adjust as needed."""
    window = df.tail(lookback)
    return float(window["High"].max()), float(window["Low"].min())

def trend_label(price: float, ema20: float, sma50: float, sma200: float) -> str:
    if price > ema20 > sma50 > sma200:
        return "Uptrend"
    if price < ema20 < sma50 < sma200:
        return "Downtrend"
    return "Range/Consolidation"

def moving_average_cross(ema20: pd.Series, sma50: pd.Series, sma200: pd.Series) -> str:
    crosses = []
    if ema20.iloc[-1] > sma50.iloc[-1] and ema20.iloc[-2] <= sma50.iloc[-2]:
        crosses.append("20 EMA crossed above 50 SMA (bullish)")
    if ema20.iloc[-1] < sma50.iloc[-1] and ema20.iloc[-2] >= sma50.iloc[-2]:
        crosses.append("20 EMA crossed below 50 SMA (bearish)")

    # Golden/Death cross using 50 SMA vs 200 SMA
    if sma50.iloc[-1] > sma200.iloc[-1] and sma50.iloc[-2] <= sma200.iloc[-2]:
        crosses.append("Golden Cross (50 SMA > 200 SMA) (bullish)")
    if sma50.iloc[-1] < sma200.iloc[-1] and sma50.iloc[-2] >= sma200.iloc[-2]:
        crosses.append("Death Cross (50 SMA < 200 SMA) (bearish)")

    return "; ".join(crosses) if crosses else "No major MA cross on last bar"

def build_trade_setup(price: float, trend: str, fib: dict) -> dict:
    if trend == "Uptrend":
        buy_low = fib["38.2%"]
        buy_high = fib["61.8%"]
        stop = fib["0%"]
        target1 = fib["100%"]
        target2 = target1 + (target1 - stop) * 0.272  # Fibonacci extension
    elif trend == "Downtrend":
        # Mirror logic for short bias (simple; can refine)
        buy_low = fib["38.2%"]
        buy_high = fib["61.8%"]
        stop = fib["100%"]
        target1 = fib["0%"]
        target2 = target1 - (stop - target1) * 0.272
    else:
        return {
            "trend_bias": "Neutral",
            "buy_zone": None,
            "target1": None,
            "target2": None,
            "stop_loss": None,
            "rrr_target1": None,
        }

    entry = (buy_low + buy_high) / 2.0
    rr = (target1 - entry) / (entry - stop) if entry != stop else None

    return {
        "trend_bias": "Bullish" if trend == "Uptrend" else "Bearish",
        "buy_zone": (buy_low, buy_high),
        "target1": target1,
        "target2": target2,
        "stop_loss": stop,
        "rrr_target1": rr,
    }

def analyze_symbol(symbol: str, exchange: str, period: str = "1y") -> None:
    ticker = resolve_ticker_online(symbol, exchange)
    df = download_ohlcv(ticker, period=period)

    indicators = compute_indicators(df)
    high, low = pick_swing(df, lookback=60)
    fib = fib_levels(high, low)

    price = float(indicators.close.iloc[-1])
    ema20 = float(indicators.ema20.iloc[-1])
    sma50 = float(indicators.sma50.iloc[-1])
    sma200 = float(indicators.sma200.iloc[-1])
    rsi = float(indicators.rsi.iloc[-1])
    macd = float(indicators.macd.iloc[-1])
    macd_sig = float(indicators.macd_signal.iloc[-1])
    macd_hist = float(indicators.macd_hist.iloc[-1])
    stoch_k = float(indicators.stoch_k.iloc[-1])
    stoch_d = float(indicators.stoch_d.iloc[-1])
    obv = float(indicators.obv.iloc[-1])

    trend = trend_label(price, ema20, sma50, sma200)
    ma_cross = moving_average_cross(indicators.ema20, indicators.sma50, indicators.sma200)
    setup = build_trade_setup(price, trend, fib)

    print("\n==============================")
    print(f"STOCKGPT ANALYSIS: {symbol.upper()} ({ticker})")
    print(f"Exchange: {exchange.upper()} | Period: {period}")
    print("==============================")
    print("STEP 1: Trend & Structure")
    print(f"Current trend: {trend}")
    print(f"Key levels (fib swing from last 60 candles): High={high:.2f}, Low={low:.2f}")

    print("\nSTEP 2: Technical Indicators")
    print(f"Price: {price:.2f}")
    print(f"20 EMA: {ema20:.2f}")
    print(f"50 SMA: {sma50:.2f}")
    print(f"200 SMA: {sma200:.2f}")
    print(f"Golden/Death cross check: {ma_cross}")
    print(f"RSI(14): {rsi:.2f}")
    print(f"MACD(12,26,9): {macd:.2f} | Signal: {macd_sig:.2f} | Hist: {macd_hist:.2f}")
    print(f"Stochastic Oscillator: %K={stoch_k:.2f} | %D={stoch_d:.2f}")
    print(f"OBV: {obv:.2f}")
    print("Fibonacci retracement levels:")
    for lvl, val in fib.items():
        print(f"  {lvl}: {val:.2f}")

    print("\nSTEP 5: Trade Setup (derived from fib & trend)")
    if setup["buy_zone"] is None:
        print("No clear setup in range environment (trend = Neutral).")
    else:
        bz = setup["buy_zone"]
        print(f"Trend bias: {setup['trend_bias']}")
        print(f"Buy zone: {bz[0]:.2f} to {bz[1]:.2f}")
        print(f"Target 1: {setup['target1']:.2f}")
        print(f"Target 2: {setup['target2']:.2f}")
        print(f"Stop-loss: {setup['stop_loss']:.2f}")
        rr = setup['rrr_target1']
        print(f"Risk-reward (T1): {rr:.2f}" if rr is not None else "Risk-reward (T1): N/A")

    print("\nDISCLAIMER: Not financial advice. Do your own research.")

def main():
    symbols_in = input("Enter symbols separated by commas (e.g., INFY, RELIANCE): ").strip()
    exchange = input("Exchange (NSE/BSE/NASDAQ/NYSE): ").strip()
    symbols: List[str] = [s.strip() for s in symbols_in.split(",") if s.strip()]

    for sym in symbols:
        try:
            analyze_symbol(sym, exchange, period="1y")
        except Exception as e:
            print(f"\nERROR processing {sym}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
