# khs_pyrofex_backtest.py
import datetime as dt
import pandas as pd
import numpy as np
import pyRofex
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
from dataclasses import field

# ==============================================================
# 🧩 Kalman Hull Supertrend
# ==============================================================
@dataclass
class KHSParams:
    price_col: str = "close"
    measurement_noise: float = 3.0
    process_noise: float = 0.01
    atr_period: int = 12
    factor: float = 1.7
    N: int = 5

def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

class _KalmanN:
    def __init__(self, N: int, process_noise: float, measurement_noise: float):
        self.N = N
        self.q = process_noise
        self.r = measurement_noise
        self.x = [np.nan]*N
        self.p = [100.0]*N
        self._inited = False

    def _init(self, z0: float):
        self.x = [z0]*self.N
        self.p = [1.0]*self.N
        self._inited = True

    def update(self, z: float):
        if not self._inited:
            self._init(z)
            return self.x[0]
        pred_x = self.x[:]
        pred_p = [p + self.q for p in self.p]
        new_x, new_p = [], []
        for i in range(self.N):
            k = pred_p[i] / (pred_p[i] + self.r)
            xi = pred_x[i] + k*(z - pred_x[i])
            pi = (1.0 - k)*pred_p[i]
            new_x.append(xi)
            new_p.append(pi)
        self.x, self.p = new_x, new_p
        return self.x[0]

def kalman_series(series: pd.Series, measurement_noise: float, process_noise: float, N: int = 5) -> pd.Series:
    kf = _KalmanN(N=N, process_noise=process_noise, measurement_noise=measurement_noise)
    return series.astype(float).apply(kf.update)

def khma(prices: pd.Series, length: float, process_noise: float, N: int = 5) -> pd.Series:
    k_full = kalman_series(prices, measurement_noise=length, process_noise=process_noise, N=N)
    k_half = kalman_series(prices, measurement_noise=length/2.0, process_noise=process_noise, N=N)
    hull_seed = 2.0*k_half - k_full
    m_noise2 = max(1.0, round(np.sqrt(length)))
    return kalman_series(hull_seed, measurement_noise=m_noise2, process_noise=process_noise, N=N)

def supertrend_from_src(df: pd.DataFrame, src: pd.Series, factor: float, atr_period: int):
    atr = _atr(df, atr_period)
    upper = src + factor * atr
    lower = src - factor * atr
    prev_lower = lower.shift(1).fillna(method="bfill")
    prev_upper = upper.shift(1).fillna(method="bfill")
    lower = np.where((lower > prev_lower) | (df["close"].shift(1) < prev_lower), lower, prev_lower)
    upper = np.where((upper < prev_upper) | (df["close"].shift(1) > prev_upper), upper, prev_upper)
    lower = pd.Series(lower, index=src.index)
    upper = pd.Series(upper, index=src.index)
    direction = pd.Series(index=src.index, dtype="float64")
    st = pd.Series(index=src.index, dtype="float64")
    direction.iloc[0] = 1
    st.iloc[0] = upper.iloc[0]
    for i in range(1, len(src)):
        prev_st = st.iloc[i - 1]
        if np.isnan(prev_st):
            direction.iloc[i] = 1
        elif prev_st == prev_upper.iloc[i - 1]:
            direction.iloc[i] = -1 if df["close"].iloc[i] > upper.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if df["close"].iloc[i] < lower.iloc[i] else -1
        st.iloc[i] = lower.iloc[i] if direction.iloc[i] == -1 else upper.iloc[i]
    dir_prev = direction.shift(1)
    long_sig = (dir_prev > 0) & (direction < 0)
    short_sig = (dir_prev < 0) & (direction > 0)
    return st, direction, long_sig.fillna(False), short_sig.fillna(False)

def kalman_hull_supertrend(df: pd.DataFrame, params: KHSParams = KHSParams()):
    price = df[params.price_col].astype(float)
    khma_series = khma(price, params.measurement_noise, params.process_noise, params.N)
    st, direction, long_sig, short_sig = supertrend_from_src(df, khma_series, params.factor, params.atr_period)
    out = df.copy()
    out["khma"] = khma_series
    out["supertrend"] = st
    out["direction"] = direction
    out["long"] = long_sig
    out["short"] = short_sig
    return out

# ==============================================================
# ⚙️ Integración con reMarkets
# ==============================================================


@dataclass
class BacktestConfig:
    user: str = ""         
    password: str = "$"
    account: str = ""
    instrument: str = "DLR/OCT25"
    bar_freq: str = "1H"
    params: KHSParams = field(default_factory=KHSParams)
    start_date: Optional[dt.date] = None
    end_date: Optional[dt.date] = None




def initialize_remarkets(cfg: BacktestConfig):
    import importlib
    import pyRofex.service as service

    # 🔄 Reset completo del estado interno de pyRofex
    importlib.reload(service)
    print(f"🧩 Intentando login con usuario={cfg.user}, cuenta={cfg.account}")

    try:
        pyRofex.initialize(
            user=cfg.user.strip(),
            password=cfg.password.strip(),
            account=cfg.account.strip(),
            environment=pyRofex.Environment.REMARKET
        )
        print("✅ Login correcto (reMarkets)")
    except Exception as e:
        print("❌ Error al autenticar:", e)
        raise


def _to_ohlc_from_trades(trades_json: dict, freq: str) -> pd.DataFrame:
    trades = trades_json.get("trades", [])
    if not trades:
        raise ValueError("No hay trades históricos.")
    df = pd.DataFrame(trades)
    df["ts"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("ts").sort_index()
    ohlc = df["price"].resample(freq).ohlc()
    vol = df["size"].resample(freq).sum().rename("volume")
    out = pd.concat([ohlc, vol], axis=1).dropna()
    out.columns = ["open", "high", "low", "close", "volume"]
    return out

# ==============================================================
# 📊 Backtest
# ==============================================================
def backtest(cfg: BacktestConfig):
    print("== Backtest Kalman Hull Supertrend ==")
    initialize_remarkets(cfg)
    end = cfg.end_date or dt.date.today()
    start = cfg.start_date or dt.date(year=end.year, month=1, day=1)
    trades = pyRofex.get_trade_history(ticker=cfg.instrument, start_date=start, end_date=end)
    ohlc = _to_ohlc_from_trades(trades, cfg.bar_freq)
    df = kalman_hull_supertrend(ohlc, cfg.params)

    # Estrategia: posición = -1 (short) si direction>0, +1 (long) si direction<0
    df["pos"] = np.where(df["direction"] < 0, 1, -1)
    df["ret"] = df["close"].pct_change()
    df["strat_ret"] = df["pos"].shift(1) * df["ret"]
    df["equity"] = (1 + df["strat_ret"].fillna(0)).cumprod()

    total_ret = df["equity"].iloc[-1] - 1
    wins = df.loc[df["strat_ret"] > 0, "strat_ret"]
    losses = df.loc[df["strat_ret"] < 0, "strat_ret"]
    win_rate = len(wins) / (len(wins) + len(losses)) * 100 if len(wins) + len(losses) > 0 else 0
    pf = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else np.nan

    print(f"\n📈 Retorno total: {total_ret:.2%}")
    print(f"✅ Win rate: {win_rate:.1f}%")
    print(f"💰 Profit factor: {pf:.2f}")
    print(f"📊 Barras analizadas: {len(df)} ({cfg.bar_freq})")

    df.to_csv("A3Bot_backtest.csv", index=True)
    print("💾 Guardado en A3Bot_backtest.csv")


    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["equity"], label="Equity Curve")
    plt.title(f"Kalman Hull Supertrend - {cfg.instrument}")
    plt.xlabel("Fecha")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.legend()
    plt.show()

    return df

# ==============================================================
# 🚀 MAIN
# ==============================================================
if __name__ == "__main__":
    cfg = BacktestConfig(
        user="",
        password="$",
        account="",
        instrument="DLR/OCT25",
        bar_freq="1H",
        params=KHSParams(
            price_col="close",
            measurement_noise=3.0,
            process_noise=0.01,
            atr_period=12,
            factor=1.7,
            N=5
        )
    )

    df = backtest(cfg)
