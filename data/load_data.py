import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import yfinance as yf


DEFAULT_SYMBOLS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]
START_DATE = "2021-01-01"
END_DATE = "2025-01-01"
LOOKBACK = 60
TRAIN_END = "2022-12-31"
BACKTEST_START = "2023-01-01"
SENTIMENT_CACHE = Path("data/sentiment_cache.json")
MARKET_CACHE = Path("data/market_features_cache.parquet")


@dataclass
class MarketDataset:
    train_obs: List[Dict[str, np.ndarray]]
    train_next_returns: np.ndarray
    train_dates: List[str]
    test_obs: List[Dict[str, np.ndarray]]
    test_next_returns: np.ndarray
    test_dates: List[str]
    symbols: List[str]


def _safe_series(values: pd.Series) -> pd.Series:
    return values.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return _safe_series(rsi / 100.0)


def _headline_sentiment_score(text: str) -> float:
    positive_words = {
        "surge", "gain", "gains", "beat", "beats", "bullish", "growth", "upgrade",
        "strong", "profit", "profits", "record", "outperform", "rally", "improves",
    }
    negative_words = {
        "fall", "falls", "drop", "drops", "miss", "misses", "bearish", "downgrade",
        "weak", "loss", "losses", "decline", "slump", "underperform", "cuts",
    }
    tokens = [t.strip(".,:;!?()[]{}\"'`).-_").lower() for t in text.split()]
    if not tokens:
        return 0.0
    pos = sum(1 for t in tokens if t in positive_words)
    neg = sum(1 for t in tokens if t in negative_words)
    return float(np.tanh((pos - neg) / max(len(tokens), 1) * 8.0))


def _read_sentiment_cache() -> Dict[str, Dict[str, float]]:
    if not SENTIMENT_CACHE.exists():
        return {}
    return json.loads(SENTIMENT_CACHE.read_text(encoding="utf-8"))


def _write_sentiment_cache(cache: Dict[str, Dict[str, float]]) -> None:
    SENTIMENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SENTIMENT_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _scrape_monthly_sentiment(symbol: str, month_start: pd.Timestamp, month_end: pd.Timestamp) -> float:
    naked = symbol.replace(".NS", "")
    query = f"{naked} NSE stock after:{month_start.date()} before:{month_end.date()}"
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        scores = []
        for item in root.findall(".//item"):
            title = item.findtext("title") or ""
            if title:
                scores.append(_headline_sentiment_score(title))
        if not scores:
            return 0.0
        return float(np.mean(scores))
    except Exception:
        return 0.0


def _build_daily_sentiment(symbol: str, dates: pd.DatetimeIndex) -> pd.Series:
    cache = _read_sentiment_cache()
    cache_symbol = cache.get(symbol, {})

    month_starts = pd.date_range(dates.min().replace(day=1), dates.max(), freq="MS")
    monthly_values: Dict[str, float] = {}

    for month_start in month_starts:
        month_end = (month_start + pd.offsets.MonthEnd(1)) + pd.Timedelta(days=1)
        key = month_start.strftime("%Y-%m")
        if key in cache_symbol:
            monthly_values[key] = float(cache_symbol[key])
            continue
        score = _scrape_monthly_sentiment(symbol, month_start, month_end)
        monthly_values[key] = score
        cache_symbol[key] = score

    cache[symbol] = cache_symbol
    _write_sentiment_cache(cache)

    sentiment = pd.Series(index=dates, dtype=float)
    for d in dates:
        key = d.strftime("%Y-%m")
        sentiment.loc[d] = monthly_values.get(key, 0.0)
    return _safe_series(sentiment)


def _build_symbol_frame(symbol: str) -> pd.DataFrame:
    try:
        df = yf.download(symbol, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        # Fallback: synthetic random-walk OHLCV for stable offline demos.
        idx = pd.bdate_range(START_DATE, END_DATE)
        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
        base = 100.0 + np.cumsum(rng.normal(0.0, 1.0, len(idx)))
        close_synth = np.maximum(base, 1.0)
        vol_synth = rng.integers(1_000_000, 9_000_000, size=len(idx)).astype(float)
        df = pd.DataFrame({"Close": close_synth, "Volume": vol_synth}, index=idx)

    close_col = df["Close"]
    volume_col = df["Volume"]
    if isinstance(close_col, pd.DataFrame):
        close_col = close_col.iloc[:, 0]
    if isinstance(volume_col, pd.DataFrame):
        volume_col = volume_col.iloc[:, 0]

    close = _safe_series(close_col)
    volume = _safe_series(volume_col)
    ret_1d = _safe_series(close.pct_change())

    sma_10 = _safe_series(close.rolling(10).mean())
    sma_20 = _safe_series(close.rolling(20).mean())
    ema_12 = _safe_series(close.ewm(span=12, adjust=False).mean())
    ema_26 = _safe_series(close.ewm(span=26, adjust=False).mean())
    macd = _safe_series(ema_12 - ema_26)
    vol_z = _safe_series((volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-9))
    vol_chg = _safe_series(volume.pct_change())
    sentiment = _build_daily_sentiment(symbol, df.index)

    features = pd.DataFrame(
        {
            f"{symbol}_ret1d": ret_1d,
            f"{symbol}_sma10_ratio": _safe_series(close / (sma_10 + 1e-9) - 1.0),
            f"{symbol}_sma20_ratio": _safe_series(close / (sma_20 + 1e-9) - 1.0),
            f"{symbol}_rsi14": _compute_rsi(close, period=14),
            f"{symbol}_macd": _safe_series(macd / (close + 1e-9)),
            f"{symbol}_vol_z": vol_z,
            f"{symbol}_vol_chg": vol_chg,
            f"{symbol}_sentiment": sentiment,
            f"{symbol}_next_ret": _safe_series(ret_1d.shift(-1)),
        },
        index=df.index,
    )
    return features


def _zscore_fit_transform(train_values: np.ndarray, all_values: np.ndarray) -> np.ndarray:
    mu = train_values.mean(axis=0, keepdims=True)
    sigma = train_values.std(axis=0, keepdims=True) + 1e-9
    return (all_values - mu) / sigma


def _build_observations(frame: pd.DataFrame, symbols: List[str], lookback: int) -> MarketDataset:
    price_cols = []
    sentiment_cols = []
    volume_cols = []
    next_ret_cols = []
    for s in symbols:
        price_cols.extend(
            [
                f"{s}_ret1d",
                f"{s}_sma10_ratio",
                f"{s}_sma20_ratio",
                f"{s}_rsi14",
                f"{s}_macd",
            ]
        )
        volume_cols.extend([f"{s}_vol_z", f"{s}_vol_chg"])
        sentiment_cols.append(f"{s}_sentiment")
        next_ret_cols.append(f"{s}_next_ret")

    frame = frame.dropna().copy()
    train_mask = frame.index <= TRAIN_END

    frame[price_cols] = _zscore_fit_transform(frame.loc[train_mask, price_cols].values, frame[price_cols].values)
    frame[volume_cols] = _zscore_fit_transform(frame.loc[train_mask, volume_cols].values, frame[volume_cols].values)

    obs_list: List[Dict[str, np.ndarray]] = []
    next_returns: List[np.ndarray] = []
    obs_dates: List[str] = []

    for idx in range(lookback, len(frame) - 1):
        window = frame.iloc[idx - lookback : idx]
        row = frame.iloc[idx]

        obs_list.append(
            {
                "price_seq": window[price_cols].values.astype(np.float32),
                "sentiment": row[sentiment_cols].values.astype(np.float32),
                "volume": row[volume_cols].values.astype(np.float32),
            }
        )
        next_returns.append(row[next_ret_cols].values.astype(np.float32))
        obs_dates.append(frame.index[idx].strftime("%Y-%m-%d"))

    next_returns_arr = np.asarray(next_returns, dtype=np.float32)
    obs_dates_arr = np.asarray(obs_dates)
    split = obs_dates_arr < BACKTEST_START

    train_obs = [obs_list[i] for i in range(len(obs_list)) if split[i]]
    test_obs = [obs_list[i] for i in range(len(obs_list)) if not split[i]]

    return MarketDataset(
        train_obs=train_obs,
        train_next_returns=next_returns_arr[split],
        train_dates=list(obs_dates_arr[split]),
        test_obs=test_obs,
        test_next_returns=next_returns_arr[~split],
        test_dates=list(obs_dates_arr[~split]),
        symbols=symbols,
    )


def load_market_dataset(symbols: List[str] = None, lookback: int = LOOKBACK) -> MarketDataset:
    symbols = symbols or DEFAULT_SYMBOLS
    use_cache = MARKET_CACHE.exists()
    if use_cache:
        try:
            merged = pd.read_parquet(MARKET_CACHE)
            expected_cols = [f"{s}_next_ret" for s in symbols]
            if not all(c in merged.columns for c in expected_cols):
                use_cache = False
        except Exception:
            use_cache = False

    if not use_cache:
        per_symbol = [_build_symbol_frame(s) for s in symbols]
        merged = pd.concat(per_symbol, axis=1, join="inner").sort_index()
        MARKET_CACHE.parent.mkdir(parents=True, exist_ok=True)
        try:
            merged.to_parquet(MARKET_CACHE)
        except Exception:
            # Parquet engine may be unavailable; continue without persistent cache.
            pass

    return _build_observations(merged, symbols=symbols, lookback=lookback)


if __name__ == "__main__":
    dataset = load_market_dataset()
    print(f"Train observations: {len(dataset.train_obs)}")
    print(f"Test observations: {len(dataset.test_obs)}")
    print(f"Price sequence shape: {dataset.train_obs[0]['price_seq'].shape}")
    print(f"Sentiment shape: {dataset.train_obs[0]['sentiment'].shape}")
    print(f"Volume shape: {dataset.train_obs[0]['volume'].shape}")
    print(f"Returns shape: {dataset.train_next_returns.shape}")
