# 🤖 Kalman Hull Supertrend - Live Trader (reMarkets)

Este proyecto implementa un **bot de trading en tiempo real** para el mercado argentino (ROFEX) utilizando la API oficial de [Primary](https://apihub.primary.com.ar/).

La estrategia combina **filtro de Kalman**, **Hull Moving Average (HMA)** y **Supertrend**, generando señales automáticas de compra/venta sobre datos en vivo.

> ⚙️ Este bot se conecta al entorno **reMarkets (Paper Trading)** — es decir, simula operaciones reales sin riesgo.

---

## 🚀 Características

- Conexión en vivo mediante **WebSocket** de la API de Primary.
- Cálculo en tiempo real del **Kalman Hull Supertrend**.
- Envío automático de órdenes `BUY` / `SELL` (modo real o simulación).
- Registro de cada operación en `trades_log.csv`.
- Código 100% autónomo (sin módulos externos ni dependencias internas).

---

## 📦 Requisitos

- Python 3.10 o superior  
- Dependencias:
  ```bash
  pip install pyRofex pandas numpy
