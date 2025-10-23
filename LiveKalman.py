# ==============================================================
# KALMAN HULL SUPERTREND - LIVE TRADER para reMarkets (Primary)
# ==============================================================

import pyRofex
import pandas as pd
import numpy as np
import datetime as dt
import csv
import os
from dataclasses import dataclass

# ==============================================================
# ⚙️ CONFIGURACIÓN
# ==============================================================

@dataclass
class Config:
    user: str = ""
    password: str = ""
    account: str = ""
    instrument: str = "DLR/OCT25"
    timeframe_minutes: int = 5           # cada 5 minutos calcula nueva vela
    window_size: int = 60                # ticks acumulados para formar una vela
    dry_run: bool = False                # True = no envía órdenes reales (simula)
    trade_log: str = "trades_log.csv"

cfg = Config()

# ==============================================================
# 📈 ESTRATEGIA: Kalman Hull Supertrend
# ==============================================================

def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

class _KalmanN:
    def __init__(self, N=5, q=0.01, r=3.0):
        self.N = N
        self.q = q
        self.r = r
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

def kalman_series(series, measurement_noise=3.0, process_noise=0.01, N=5):
    kf = _KalmanN(N=N, q=process_noise, r=measurement_noise)
    return series.astype(float).apply(kf.update)

def khma(prices, length=3.0, process_noise=0.01, N=5):
    k_full = kalman_series(prices, measurement_noise=length, process_noise=process_noise, N=N)
    k_half = kalman_series(prices, measurement_noise=length/2.0, process_noise=process_noise, N=N)
    hull_seed = 2.0*k_half - k_full
    m_noise2 = max(1.0, round(np.sqrt(length)))
    return kalman_series(hull_seed, measurement_noise=m_noise2, process_noise=process_noise, N=N)

def supertrend(df, src, factor=1.7, atr_period=12):
    atr = _atr(df, atr_period)
    upper = src + factor * atr
    lower = src - factor * atr
    prev_lower = lower.shift(1).bfill()
    prev_upper = upper.shift(1).bfill()
    lower = np.where((lower > prev_lower) | (df["close"].shift(1) < prev_lower), lower, prev_lower)
    upper = np.where((upper < prev_upper) | (df["close"].shift(1) > prev_upper), upper, prev_upper)
    lower, upper = pd.Series(lower, index=src.index), pd.Series(upper, index=src.index)

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

def kalman_hull_supertrend(df):
    price = df["close"].astype(float)
    khma_series = khma(price, 3.0, 0.01, 5)
    st, direction, long_sig, short_sig = supertrend(df, khma_series, 1.7, 12)
    out = df.copy()
    out["khma"] = khma_series
    out["supertrend"] = st
    out["direction"] = direction
    out["long"] = long_sig
    out["short"] = short_sig
    return out

# ==============================================================
# 🧠 FUNCIONES PRINCIPALES
# ==============================================================

print("== KALMAN LIVE TRADER ==")
print(f"🧩 Iniciando sesión como {cfg.user} ...")
pyRofex.initialize(
    user=cfg.user,
    password=cfg.password,
    account=cfg.account,
    environment=pyRofex.Environment.REMARKET
)
print("✅ Login correcto")

buffer = []
last_signal = None

def log_trade(side, price):
    """Guarda trade en CSV."""
    file_exists = os.path.isfile(cfg.trade_log)
    with open(cfg.trade_log, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp", "side", "price"])
        w.writerow([dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), side, price])

def send_order(side, price=None):
    if cfg.dry_run:
        print(f"[DRY-RUN] {side} @ {price}")
        log_trade(side, price)
        return
    response = pyRofex.send_order(
        ticker=cfg.instrument,
        side=pyRofex.Side.BUY if side == "BUY" else pyRofex.Side.SELL,
        size=1,
        order_type=pyRofex.OrderType.MARKET,
        time_in_force=pyRofex.TimeInForce.DAY
    )
    print(f"[ORDER] {side} enviada:", response)
    log_trade(side, price)

def market_data_handler(message):
    global buffer, last_signal
    if message["type"] != "md":
        return
    try:
        price = message["last"]["price"]
    except KeyError:
        return

    ts = dt.datetime.utcnow()
    buffer.append((ts, price))

    if len(buffer) >= cfg.window_size:
        df = pd.DataFrame(buffer, columns=["datetime", "close"])
        df["open"] = df["close"].iloc[0]
        df["high"] = df["close"].max()
        df["low"] = df["close"].min()
        df = df.set_index("datetime")

        sig = kalman_hull_supertrend(df)
        last = sig.iloc[-1]
        if last["long"] and last_signal != "LONG":
            send_order("BUY", last["close"])
            last_signal = "LONG"
        elif last["short"] and last_signal != "SHORT":
            send_order("SELL", last["close"])
            last_signal = "SHORT"
        else:
            print(f"[{ts:%H:%M:%S}] sin cambio de señal ({last_signal})")

        buffer.clear()

# ==============================================================
# 🚀 CONEXIÓN EN VIVO
# ==============================================================

pyRofex.init_websocket_connection(market_data_handler=market_data_handler)
pyRofex.market_data_subscription(
    tickers=[cfg.instrument],
    entries=[pyRofex.MarketDataEntry.LAST]
)

print(f"📡 Escuchando datos en vivo de {cfg.instrument}...")
print("CTRL+C para detener.")
