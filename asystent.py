# asystent_ai.py
#!/usr/bin/env python3
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import yfinance as yf
import ta
import matplotlib.pyplot as plt

warnings.simplefilter("ignore", category=FutureWarning)

# ========= DANE =========
def fetch_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if df.empty or not {"Close", "High", "Low", "Volume"}.issubset(df.columns):
        raise RuntimeError(f"Brak danych dla {ticker}.")
    out = pd.DataFrame({
        "close": pd.to_numeric(df["Close"], errors="coerce"),
        "high":  pd.to_numeric(df["High"],  errors="coerce"),
        "low":   pd.to_numeric(df["Low"],   errors="coerce"),
        "vol":   pd.to_numeric(df["Volume"],errors="coerce"),
    }).dropna()
    if len(out) < 50:
        raise RuntimeError(f"Za mało danych dla {ticker} (min 50 sesji).")
    return out

def fetch_history_from(ticker: str, start: str = "2015-01-01") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
    if df.empty or "Close" not in df.columns:
        raise RuntimeError(f"Brak danych dla {ticker} od {start}.")
    out = pd.DataFrame({
        "close": pd.to_numeric(df["Close"], errors="coerce"),
        "high":  pd.to_numeric(df["High"],  errors="coerce"),
        "low":   pd.to_numeric(df["Low"],   errors="coerce"),
        "vol":   pd.to_numeric(df["Volume"],errors="coerce"),
    }).dropna()
    if len(out) < 250:
        raise RuntimeError(f"Za mało danych dla backtestu {ticker}.")
    return out

# ========= WSKAŹNIKI =========
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    sma20 = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
    rsi14 = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    macd_obj = ta.trend.MACD(df["close"])
    macd_line = macd_obj.macd()
    macd_sig  = macd_obj.macd_signal()
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    bb_high = bb.bollinger_hband()
    bb_low  = bb.bollinger_lband()
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14)\
            .average_true_range()

    feats = pd.concat(
        [df["close"], sma20, rsi14, macd_line, macd_sig, bb_high, bb_low, atr],
        axis=1
    )
    feats.columns = ["close", "sma20", "rsi14", "macd", "macd_sig", "bb_high", "bb_low", "atr"]
    feats = feats.dropna()
    return feats

# ========= OPIS / ANALIZA =========
def analyze_row(ticker: str, feats: pd.DataFrame) -> Dict[str, Any]:
    last = feats.iloc[-1]
    prev_close = feats["close"].iloc[-2] if len(feats) >= 2 else last["close"]
    day_chg = 100.0 * (last["close"] / prev_close - 1.0)

    trend_up = bool(last["close"] > last["sma20"])
    macd_up  = bool(last["macd"] > last["macd_sig"])

    return {
        "ticker": ticker.upper(),
        "price": round(float(last["close"]), 2),
        "day_chg_pct": round(float(day_chg), 2),
        "SMA20": round(float(last["sma20"]), 2),
        "RSI14": round(float(last["rsi14"]), 2),
        "MACD": round(float(last["macd"]), 4),
        "MACDsig": round(float(last["macd_sig"]), 4),
        "BB_low": round(float(last["bb_low"]), 2),
        "BB_high": round(float(last["bb_high"]), 2),
        "ATR_pct": round(float(100.0 * last["atr"] / last["close"]), 2),
        "trend_up": trend_up,
        "macd_up": macd_up,
    }

def passes_screen(row: Dict[str, Any], rsi_min: float, rsi_max: float) -> bool:
    return (row["trend_up"] and row["macd_up"] and (rsi_min <= row["RSI14"] <= rsi_max))

# ========= WYKRES =========
def save_chart(ticker: str, feats: pd.DataFrame, out_dir: Path, tail: int = 120) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cut = feats.tail(tail)

    plt.figure(figsize=(10, 4))
    plt.plot(cut.index, cut["close"], label="Cena")
    plt.plot(cut.index, cut["sma20"], label="SMA20")
    plt.plot(cut.index, cut["bb_high"], label="BB High")
    plt.plot(cut.index, cut["bb_low"], label="BB Low")
    plt.fill_between(cut.index, cut["bb_low"], cut["bb_high"], alpha=0.1)
    plt.title(f"{ticker.upper()} – cena, SMA20, pasy BB")
    plt.xlabel("Data"); plt.ylabel("Cena"); plt.grid(True); plt.legend()
    out_path = out_dir / f"{ticker.upper()}.png"
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    return out_path

# ========= BACKTEST: SMA(fast/slow) + filtr RSI =========
def backtest_sma_rsi(
    ticker: str,
    start: str = "2015-01-01",
    fast: int = 50,
    slow: int = 200,
    rsi_min: float = 35.0,
    rsi_max: float = 65.0,
    fee_bps: float = 5.0,  # 5 bps = 0.05% na wejście lub wyjście
) -> Dict[str, Any]:
    df = fetch_history_from(ticker, start=start)
    close = df["close"]

    sma_fast = ta.trend.SMAIndicator(close, window=fast).sma_indicator()
    sma_slow = ta.trend.SMAIndicator(close, window=slow).sma_indicator()
    rsi14    = ta.momentum.RSIIndicator(close, window=14).rsi()

    # Wspólny indeks bez NaN
    dat = pd.concat([close, sma_fast, sma_slow, rsi14], axis=1).dropna()
    dat.columns = ["close", "fast", "slow", "rsi"]

    # Sygnał: long gdy fast>slow i RSI w widełkach; inaczej 0 (cash)
    signal = ((dat["fast"] > dat["slow"]) & (dat["rsi"].between(rsi_min, rsi_max))).astype(int)

    # Pozycja = sygnał przesunięty o 1 dzień (brak „look-ahead”)
    position = signal.shift(1).fillna(0)

    # Zwroty dzienne
    ret = dat["close"].pct_change().fillna(0.0)

    # Koszt przy zmianie pozycji (wejście/wyjście)
    turns = position.diff().abs().fillna(position.iloc[0])
    cost = turns * (fee_bps / 10000.0)

    strat_ret = position * ret - cost
    bench_ret = ret  # kup-i-trzymaj

    # Kapitał skumulowany
    eq_strat = (1.0 + strat_ret).cumprod()
    eq_bench = (1.0 + bench_ret).cumprod()

    n = len(strat_ret)
    if n < 50:
        raise RuntimeError("Za mało punktów do sensownego backtestu.")

    # Metryki
    def annualize(series: pd.Series) -> float:
        return series.mean() * 252

    def vol_annual(series: pd.Series) -> float:
        return series.std(ddof=0) * np.sqrt(252)

    cagr_strat = eq_strat.iloc[-1] ** (252.0 / n) - 1.0
    cagr_bench = eq_bench.iloc[-1] ** (252.0 / n) - 1.0

    dd_series = eq_strat / eq_strat.cummax() - 1.0
    max_dd = dd_series.min()

    vol = vol_annual(strat_ret)
    sharpe = (annualize(strat_ret) / vol) if vol > 0 else np.nan

    # WinRate liczymy na dniach z ekspozycją
    active = strat_ret[position > 0]
    winrate = float((active > 0).mean()) if len(active) else 0.0

    gains = strat_ret[strat_ret > 0].sum()
    losses = strat_ret[strat_ret < 0].sum()
    profit_factor = float(gains / abs(losses)) if losses < 0 else np.nan

    entries = ((position == 1) & (position.shift(1) == 0)).sum()
    exposure_days = int(position.sum())
    avg_hold = float(exposure_days / entries) if entries > 0 else 0.0

    return {
        "ticker": ticker.upper(),
        "start": start,
        "fast": int(fast),
        "slow": int(slow),
        "rsi_min": float(rsi_min),
        "rsi_max": float(rsi_max),
        "fee_bps": float(fee_bps),
        "CAGR_strat": round(float(cagr_strat), 4),
        "CAGR_bench": round(float(cagr_bench), 4),
        "MaxDD": round(float(max_dd), 4),
        "Sharpe": round(float(sharpe), 3) if not np.isnan(sharpe) else None,
        "WinRate_days": round(float(winrate), 3),
        "ProfitFactor": round(float(profit_factor), 3) if not np.isnan(profit_factor) else None,
        "Trades": int(entries),
        "AvgHoldDays": round(avg_hold, 1),
    }

# ========= WATCHLISTY / CLI =========
def read_watchlist(path: Path) -> List[str]:
    tickers: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tickers.append(line.split()[0].upper())
    return tickers

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Asystent rynkowy: analizy, CSV, wykresy, backtest.")
    p.add_argument("tickers", nargs="*", help="Tickery (np. AAPL TSLA) lub 'AAPL,TSLA'.")
    p.add_argument("--watchlist", type=str, help="Plik z listą tickerów (np. watchlists/tech.txt).")
    p.add_argument("--csv", type=str, default=None, help="Zapis wyników analizy do CSV (np. raport.csv).")
    p.add_argument("--charts", type=str, default=None, help="Katalog na wykresy PNG (np. charts/).")
    p.add_argument("--period", type=str, default="6mo", help="Okres danych: 3mo, 6mo, 1y, 2y...")
    p.add_argument("--interval", type=str, default="1d", help="Interwał: 1d, 1h (dla krypto/intraday).")
    p.add_argument("--rsi-min", type=float, default=35.0, help="Min RSI do screenera (domyślnie 35).")
    p.add_argument("--rsi-max", type=float, default=65.0, help="Max RSI do screenera (domyślnie 65).")

    # BACKTEST
    p.add_argument("--backtest", type=str, default=None,
                   help="Tickery do backtestu (np. 'AAPL,TSLA' lub jeden ticker).")
    p.add_argument("--bt-start", type=str, default="2015-01-01", help="Data startu backtestu (YYYY-MM-DD).")
    p.add_argument("--bt-fast", type=int, default=50, help="Szybka SMA (domyślnie 50).")
    p.add_argument("--bt-slow", type=int, default=200, help="Wolna SMA (domyślnie 200).")
    p.add_argument("--bt-rsi-min", type=float, default=35.0, help="Min RSI w backteście (domyślnie 35).")
    p.add_argument("--bt-rsi-max", type=float, default=65.0, help="Max RSI w backteście (domyślnie 65).")
    p.add_argument("--bt-fee-bps", type=float, default=5.0, help="Koszt w bps na wejście/wyjście (np. 5=0.05%).")
    p.add_argument("--bt-csv", type=str, default=None, help="Zapis wyników backtestu do CSV.")

    return p.parse_args()

def expand_tickers(args: argparse.Namespace) -> List[str]:
    tickers: List[str] = []
    if args.watchlist:
        tickers += read_watchlist(Path(args.watchlist))
    for t in args.tickers:
        tickers += [x.strip().upper() for x in t.split(",") if x.strip()]
    return tickers or ["AAPL", "MSFT", "TSLA"]

# ========= MAIN =========
def main():
    args = parse_args()
    tickers = expand_tickers(args)

    # --- ANALIZA / SCREEN / CHARTS / CSV ---
    rows: List[Dict[str, Any]] = []
    charts_dir = Path(args.charts) if args.charts else None
    passed: List[str] = []

    for t in tickers:
        try:
            data = fetch_history(t, period=args.period, interval=args.interval)
            feats = compute_indicators(data)
            row = analyze_row(t, feats)
            rows.append(row)
            if (row and passes_screen(row, args.rsi_min, args.rsi_max)):
                passed.append(t.upper())
            if charts_dir:
                save_chart(t, feats, charts_dir)
        except Exception as e:
            print(f"❌ {t.upper()}: {e}")

    for r in rows:
        if "ticker" not in r: continue
        print(f"\n=== {r['ticker']} ===")
        print(f"📈 Cena: {r['price']}  (dziś {r['day_chg_pct']}%)")
        trend_txt = "krótko w górę (cena > SMA20)" if r["trend_up"] else "krótko w dół (cena < SMA20)"
        macd_txt = "impet rośnie (MACD > sygnał)" if r["macd_up"] else "impet słabnie (MACD < sygnał)"
        rsi_txt = (
            "RSI wysoko (często krótki oddech)" if r["RSI14"] > 70 else
            "RSI nisko (często krótkie odbicie)" if r["RSI14"] < 30 else
            "RSI w normie"
        )
        print(f"🧭 Trend: {trend_txt}")
        print(f"💪 Impet: {macd_txt}")
        print(f"📊 RSI(14): {r['RSI14']} → {rsi_txt}")
        print(f"🎯 Pasy BB: [{r['BB_low']}, {r['BB_high']}] (zakres „normalnych” wahań)")
        print(f"🌪️ Zmienność (ATR%): {r['ATR_pct']}%")

    if passed:
        print("\n✅ Screener (trend_up + MACD_up + RSI w widełkach): " + ", ".join(passed))
    else:
        print("\nℹ️ Brak spółek spełniających screener przy tych progach.")

    if args.csv and rows:
        out = Path(args.csv)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n📁 Zapisano CSV: {out}")

    # --- BACKTEST (opcjonalnie) ---
    if args.backtest:
        bt_tickers = [x.strip().upper() for x in args.backtest.split(",") if x.strip()]
        bt_rows = []
        print("\n=== BACKTEST: SMA(fast/slow) + filtr RSI ===")
        for t in bt_tickers:
            try:
                res = backtest_sma_rsi(
                    ticker=t,
                    start=args.bt_start,
                    fast=args.bt_fast,
                    slow=args.bt_slow,
                    rsi_min=args.bt_rsi_min,
                    rsi_max=args.bt_rsi_max,
                    fee_bps=args.bt_fee_bps,
                )
                bt_rows.append(res)
                print(f"{res['ticker']}: "
                      f"CAGR {res['CAGR_strat']*100:.2f}% "
                      f"(bench {res['CAGR_bench']*100:.2f}%) | "
                      f"MaxDD {res['MaxDD']*100:.1f}% | "
                      f"Sharpe {res['Sharpe']} | "
                      f"WinRate {res['WinRate_days']*100:.1f}% | "
                      f"PF {res['ProfitFactor']} | "
                      f"Trades {res['Trades']} | "
                      f"AvgHold {res['AvgHoldDays']}d")
            except Exception as e:
                print(f"❌ BT {t}: {e}")

        if args.bt_csv and bt_rows:
            Path(args.bt_csv).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(bt_rows).to_csv(args.bt_csv, index=False)
            print(f"\n📁 Zapisano wyniki backtestu: {args.bt_csv}")

if __name__ == "__main__":
    main()