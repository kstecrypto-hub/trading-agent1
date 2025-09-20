# ================================
# Trading Agent — Fortified Build
# ================================
from pathlib import Path
import hashlib, io, traceback, tempfile

# ---- Stdlib
import os, time, csv, math, json, uuid, threading, re, sys
from typing import Optional, List, Dict, Tuple
from collections import deque, defaultdict
from datetime import datetime, timezone, timedelta

# ---- Third-party
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, render_template_string, jsonify

# RL / Gym / SB3
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Alpaca (REST)
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

# Charts (headless-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================================
# SB3 / Gym compatibility determination
# ======================================
try:
    import stable_baselines3 as _sb3
    _SB3_VER = getattr(_sb3, "__version__", "1.0.0")
except Exception:
    _SB3_VER = "1.0.0"

_SB3_USES_GYMNASIUM = False
try:
    _SB3_USES_GYMNASIUM = int(_SB3_VER.split(".")[0]) >= 2
except Exception:
    _SB3_USES_GYMNASIUM = False

# ---- ENV
def load_env_safely() -> Dict[str, str]:
    vals = {}
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    vals["ALPACA_API_KEY"]    = (os.getenv("ALPACA_API_KEY") or "").strip()
    vals["ALPACA_API_SECRET"] = (os.getenv("ALPACA_API_SECRET") or "").strip()
    return vals

ENV = load_env_safely()

# ---- Small RL helpers
def rl_model_path(symbol: str) -> str:
    return f"rl_trading_model_{symbol.upper()}"

def _model_fingerprint(model) -> str:
    """Stable-Baselines3 model checksum via temp file (robust across versions)."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            model.save(tmp_path[:-4])  # SB3 appends .zip automatically
            with open(tmp_path, "rb") as f:
                data = f.read()
            return hashlib.sha256(data).hexdigest()[:16]
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    except Exception:
        return "fingerprint_err"

def _round_up_to_multiple(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple

# ---- Flask
app = Flask(__name__)
APP_TITLE = "Trading Agent — Fortified"

# Escaped Jinja inside f-string (double braces in f-string)
HTML = f"""
<!doctype html>
<title>{APP_TITLE}</title>
<style>
 body{{font-family:system-ui,Arial;margin:24px;max-width:1080px}}
 input[type=text]{{width:72%;padding:8px}}
 button, a.btn{{display:inline-block;padding:8px 12px;margin-left:6px;border:1px solid #334155;border-radius:6px;text-decoration:none;color:#0f172a;background:#e2e8f0}}
 button:hover, a.btn:hover{{background:#cbd5e1}}
 pre{{background:#0f172a;color:#e2e8f0;padding:12px;border-radius:8px;white-space:pre-wrap}}
 .hint{{color:#475569;margin-top:8px}}
 .toolbar{{margin:10px 0}}
</style>

<h2>{APP_TITLE}</h2>

<form method="post" style="margin-bottom:8px">
  <input name="cmd" placeholder="type a command e.g. 'help' or 'train model NVDA steps 30000'" />
  <button type="submit">Run</button>
  <a class="btn" href="/feed">Open Feed Page</a>
</form>

<div class="toolbar">
  <form method="post" style="display:inline"><input type="hidden" name="cmd" value="status" /><button type="submit">Status</button></form>
  <form method="post" style="display:inline"><input type="hidden" name="cmd" value="health" /><button type="submit">Health</button></form>
  <form method="post" style="display:inline"><input type="hidden" name="cmd" value="start live scores" /><button type="submit">Start Live Scores</button></form>
  <form method="post" style="display:inline"><input type="hidden" name="cmd" value="stop live scores" /><button type="submit">Stop Live Scores</button></form>
  <form method="post" style="display:inline"><input type="hidden" name="cmd" value="auto feed" /><button type="submit">Auto Feed</button></form>
</div>

<p class="hint">Quickstart: <code>set interval: 15m 60d</code> → <code>train model NVDA steps 30000</code> → <code>backtest model NVDA 60d 15m</code></p>

{{% if out %}}<pre>{{{{ out }}}}</pre>{{% endif %}}
"""

# =================
# Global Config
# =================
LIVE_SESSION_START_EQUITY: float = 0.0

RL_MODELS: Dict[str, PPO] = {}
WATCHLIST: List[str] = ["AAPL", "MSFT", "NVDA", "TSLA"]

INDICATOR_INTERVAL: str = "15m"
INDICATOR_PERIOD: str  = "60d"
POLL_INTERVAL_S: int   = 30
LIVE_SCORES_INTERVAL: int = 10
MAX_TRADE_USD: float   = 200.0

TREND_BUY_TH: int    = 65
TREND_SELL_TH: int   = 35
TREND_PERSIST: int   = 2
TRADE_COOLDOWN_SEC: int = 180

DAY_TRADE_MODE       = True
DAY_WINDOW_START_UTC = (13, 35)   # 09:35 ET
DAY_WINDOW_END_UTC   = (19, 55)   # 15:55 ET
EOD_FLATTEN          = True

MAX_TRADES_PER_DAY   = 20
MAX_TRADES_PER_SYM   = 6

STOP_LOSS_PCT        = 0.012     # 1.2%
TAKE_PROFIT_PCT      = 0.018     # 1.8%
MAX_HOLD_MIN         = 90

LONGING_ENABLED      = True
SHORTING_ENABLED     = True
CRYPTO_SHORTING_ENABLED = False  # default: block crypto shorts

# --- Model registry & autosave
MODEL_REG_PATH = os.path.join(os.getcwd(), "models.json")
GLOBAL_BASE_MODEL: Optional[str] = None
MODEL_AUTOSAVE_MIN = 15  # checkpoint cadence

def _exc(e: Exception) -> str:
    try:
        return "".join(traceback.format_exception(type(e), e, e.__traceback__))
    except Exception:
        return str(e)

def _save_model_registry(reg: Dict[str, Dict]) -> None:
    try:
        with open(MODEL_REG_PATH, "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2)
    except Exception as e:
        log_error("save_model_registry", e)

def _load_model_registry() -> Dict[str, Dict]:
    try:
        if os.path.exists(MODEL_REG_PATH):
            with open(MODEL_REG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log_error("load_model_registry", e)
    return {}

MODEL_REGISTRY: Dict[str, Dict] = _load_model_registry()

# =================
# Global State
# =================
TRADING_ACTIVE       = False
LIVE_SCORES_ACTIVE   = False

_auto_updates: deque = deque(maxlen=1000)
_sigbuf_trend        = defaultdict(lambda: deque(maxlen=TREND_PERSIST))
_last_trade_time: Dict[str, float] = {}
_last_side_by_sym: Dict[str, str]  = {}

_trades_today_total    = 0
_trades_today_by_sym   = defaultdict(int)
_last_reset_yyyy_mm_dd = None

_entry_time_by_sym: Dict[str, float] = {}

JOURNAL_ROOT = os.path.join(os.getcwd(), "journal")
JOURNAL_CSV  = os.path.join(JOURNAL_ROOT, "trades.csv")
os.makedirs(JOURNAL_ROOT, exist_ok=True)
if not os.path.exists(JOURNAL_CSV):
    with open(JOURNAL_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "trade_id","date_utc","symbol","side","qty","entry_time","entry_price",
            "exit_time","exit_price","pnl_abs","pnl_pct","outcome_label",
            "intervals","notes","snap_dir"
        ])
_open_trade_ids: Dict[str, str] = {}

DEBUG_TICKS = 0  # set via command to emit per-tick debug

# =================
# Logging helpers
# =================
def norm_cmd(s: str) -> str:
    return " ".join((s or "").lower().split())

def log_auto(msg: str) -> None:
    try:
        _auto_updates.append(msg)
    except Exception:
        pass

def log_error(context: str, err) -> None:
    try:
        emsg = err if isinstance(err, str) else _exc(err)
        _auto_updates.append(f"[ERROR] {context}: {emsg}")
    except Exception:
        pass

def dbg_reason(msg: str):
    try:
        _auto_updates.append("[DBG] " + msg)
    except Exception:
        pass

# =================
# Symbol & Market helpers
# =================
def norm_symbol(sym: str) -> str:
    s = (sym or "").upper().strip()
    alias = {"BRKB": "BRK-B", "BRK.B": "BRK-B", "BRK/B": "BRK-B"}
    return alias.get(s, s)

def is_crypto_symbol(sym: str) -> bool:
    s = (sym or "").upper()
    return ("-USD" in s) or ("/USD" in s) or s.endswith("USD")

def yf_symbol(sym: str) -> str:
    s = (sym or "").upper().replace("/", "-")
    if s.endswith("USD") and "-" not in s:
        s = s[:-3] + "-" + s[-3:]
    return s

def broker_symbol(sym: str) -> str:
    s = (sym or "").upper().replace("/", "")
    s = s.replace("-", "")
    return s

def _utc_now():
    return datetime.now(timezone.utc)

def intraday_window_open() -> bool:
    if not DAY_TRADE_MODE:
        return True
    now = _utc_now()
    if now.weekday() >= 5:
        return False
    h1, m1 = DAY_WINDOW_START_UTC
    h2, m2 = DAY_WINDOW_END_UTC
    start = now.replace(hour=h1, minute=m1, second=0, microsecond=0)
    end   = now.replace(hour=h2, minute=m2, second=0, microsecond=0)
    return start <= now <= end

def reset_intraday_counters_if_new_day():
    global _trades_today_total, _trades_today_by_sym, _last_reset_yyyy_mm_dd
    today = _utc_now().strftime("%Y-%m-%d")
    if _last_reset_yyyy_mm_dd != today:
        _trades_today_total = 0
        _trades_today_by_sym = defaultdict(int)
        _last_reset_yyyy_mm_dd = today

def can_trade_symbol(sym: str) -> bool:
    reset_intraday_counters_if_new_day()
    if not intraday_window_open() and not is_crypto_symbol(sym):
        return False
    if _trades_today_total >= MAX_TRADES_PER_DAY:
        return False
    if _trades_today_by_sym[sym] >= MAX_TRADES_PER_SYM:
        return False
    return True

def record_trade(sym: str):
    global _trades_today_total
    _trades_today_total += 1
    _trades_today_by_sym[sym] += 1
    _last_trade_time[sym] = time.time()

# =================
# Alpaca client & order utilities
# =================
def get_equity_value() -> float:
    try:
        c = get_trading_client()
        a = c.get_account()
        return float(getattr(a, "portfolio_value", getattr(a, "equity", 0.0)) or 0.0)
    except Exception as e:
        log_error("get_equity_value", e)
        return 0.0

def equity_snapshot_text(prefix: str = "") -> str:
    try:
        c = get_trading_client()
        a = c.get_account()
        pv = float(getattr(a, "portfolio_value", getattr(a, "equity", 0.0)) or 0.0)
        cash = float(getattr(a, "cash", 0.0) or 0.0)
        bp = float(getattr(a, "buying_power", 0.0) or 0.0)
        return f"{prefix}Equity=${pv:,.2f} | Cash=${cash:,.2f} | BP=${bp:,.2f}"
    except Exception as e:
        return f"{prefix}Equity snapshot error: {e}"

def get_trading_client() -> TradingClient:
    key = ENV.get("ALPACA_API_KEY", "")
    sec = ENV.get("ALPACA_API_SECRET", "")
    if not key or not sec:
        raise RuntimeError("Alpaca keys not set. Put ALPACA_API_KEY and ALPACA_API_SECRET in .env")
    return TradingClient(key, sec, paper=True)

def get_account_status_text() -> str:
    try:
        c = get_trading_client()
        a = c.get_account()
        return f"Status={a.status} | Cash=${a.cash} | BuyingPower=${a.buying_power}"
    except Exception as e:
        return f"Account error: {e}"

def list_open_orders():
    try:
        client = get_trading_client()
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        return client.get_orders(filter=req)
    except Exception as e:
        log_error("list_open_orders", e)
        return []

def cancel_order(order_id: str) -> str:
    try:
        client = get_trading_client()
        try:
            client.cancel_order_by_id(order_id)
        except Exception:
            client.cancel_order(order_id)
        return f"Cancelled {order_id}"
    except Exception as e:
        return f"Cancel error: {e}"

def cancel_all_orders() -> str:
    try:
        client = get_trading_client()
        try:
            client.cancel_orders()
            return "Cancelled all open orders."
        except Exception:
            ords = list_open_orders()
            errs = []
            for o in ords:
                try:
                    cancel_order(o.id)
                except Exception as ce:
                    errs.append(str(ce))
            return "Cancelled all open orders." if not errs else "Some cancels failed: " + "; ".join(errs)
    except Exception as e:
        return f"Cancel-all error: {e}"

# =================
# Market data / Indicators
# =================
def get_history(sym: str, interval: Optional[str] = None, period: Optional[str] = None,
                retries: int = 2, backoff_s: float = 1.0) -> Optional[pd.DataFrame]:
    itv = interval or INDICATOR_INTERVAL
    per = period or INDICATOR_PERIOD
    s   = yf_symbol(sym)
    last_err = None
    for attempt in range(retries + 1):
        try:
            df = yf.download(s, interval=itv, period=per, progress=False,
                             auto_adjust=True, threads=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.dropna().copy()
                return df
            else:
                last_err = ValueError("empty dataframe")
        except Exception as e:
            last_err = e
        time.sleep(backoff_s * (attempt + 1))
    log_error(f"get_history({s})", last_err or Exception("unknown"))
    return None

def to_close_series(df: pd.DataFrame) -> pd.Series:
    try:
        s = df["Close"]
        if isinstance(s, pd.DataFrame):
            for col in s.columns:
                col_s = pd.to_numeric(s[col], errors="coerce")
                col_s = col_s.dropna().astype(float)
                if not col_s.empty:
                    col_s.name = "Close"
                    return col_s
            return pd.Series(dtype=float)
        s = pd.to_numeric(s, errors="coerce").dropna().astype(float)
        s.name = "Close"
        return s
    except Exception as e:
        log_error("to_close_series", e)
        return pd.Series(dtype=float)

def _series1d(x) -> pd.Series:
    try:
        arr = np.asarray(x).reshape(-1).astype(float)
        return pd.Series(arr)
    except Exception:
        try:
            return pd.Series([float(x)])
        except Exception:
            return pd.Series(dtype=float)

def ema(series: pd.Series, span: int) -> pd.Series:
    try:
        s = _series1d(series)
        return s.ewm(span=span, adjust=False).mean()
    except Exception as e:
        log_error(f"ema(span={span})", e)
        return pd.Series(dtype=float)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    try:
        s = _series1d(series).dropna()
        if s.empty:
            return pd.Series(dtype=float)
        d = s.diff()
        gains = d.clip(lower=0.0)
        losses = (-d).clip(lower=0.0)
        roll_up = gains.rolling(period, min_periods=period).mean()
        roll_down = losses.rolling(period, min_periods=period).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val.fillna(50.0)
    except Exception as e:
        log_error("rsi", e)
        return pd.Series(dtype=float)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    try:
        s = _series1d(series)
        if s.empty:
            empty = pd.Series(dtype=float)
            return empty, empty, empty
        ema_fast = ema(s, fast)
        ema_slow = ema(s, slow)
        line = ema_fast - ema_slow
        signal_line = ema(line, signal)
        hist = line - signal_line
        return line, signal_line, hist
    except Exception as e:
        log_error("macd", e)
        empty = pd.Series(dtype=float)
        return empty, empty, empty

def trend_score(df: pd.DataFrame) -> int:
    try:
        c = to_close_series(df)
        if len(c) < 60:
            return 50
        e20  = ema(c, 20).iloc[-1]
        e50  = ema(c, 50).iloc[-1]
        e200 = ema(c, 200).iloc[-1]
        r    = rsi(c, 14).iloc[-1]
        _, _, h = macd(c)
        h_last = h.iloc[-1] if not h.empty else 0.0

        def scalar(x, fallback):
            try:
                return x.item() if hasattr(x, "item") else float(x)
            except Exception:
                return fallback

        e20v  = scalar(e20, c.iloc[-1])
        e50v  = scalar(e50, c.iloc[-1])
        e200v = scalar(e200, c.iloc[-1])
        rv    = scalar(r, 50.0)
        hv    = scalar(h_last, 0.0)

        score = 50
        score += 10 if e20v > e50v else -10
        score += 10 if e50v > e200v else -10
        if rv > 55: score += min(15, (rv - 55) * 0.6)
        if rv < 45: score -= min(15, (45 - rv) * 0.6)
        score += max(-15, min(15, float(hv) * 50.0))
        return int(max(0, min(100, round(score))))
    except Exception as e:
        log_error("trend_score", e)
        return 50

def trend_decision(score: int) -> str:
    try:
        if score >= TREND_BUY_TH:  return "BUY"
        if score <= TREND_SELL_TH: return "SELL"
        return "HOLD"
    except Exception as e:
        log_error("trend_decision", e)
        return "HOLD"

# =================
# Scalar helpers + Price & Positions
# =================
def _to_scalar(val) -> Optional[float]:
    try:
        if hasattr(val, "item"):
            return float(val.item())
        arr = np.asarray(val).reshape(-1)
        if arr.size >= 1:
            return float(arr[-1])
    except Exception:
        pass
    return None

def _scalar(x, default=0.0) -> float:
    try:
        if hasattr(x, "item"):
            return float(x.item())
        arr = np.asarray(x).reshape(-1)
        for v in arr[::-1]:
            try:
                fv = float(v)
                if np.isfinite(fv):
                    return fv
            except Exception:
                continue
        return float(default)
    except Exception:
        return float(default)

def get_price(sym: str) -> Optional[float]:
    try:
        df = get_history(sym)
        if df is not None and not df.empty:
            s = to_close_series(df)
            if not s.empty:
                v = _to_scalar(s.iloc[-1])
                if v is not None:
                    return v
    except Exception as e:
        log_error(f"get_price({sym}) primary", e)
    try:
        df2 = yf.download(yf_symbol(sym), interval="1d", period="5d",
                          progress=False, auto_adjust=True, threads=False)
        if isinstance(df2, pd.DataFrame) and not df2.empty:
            s2 = to_close_series(df2)
            if not s2.empty:
                v2 = _to_scalar(s2.iloc[-1])
                if v2 is not None:
                    return v2
    except Exception as e:
        log_error(f"get_price({sym}) fallback", e)
    return None

def portfolio_status() -> str:
    try:
        client = get_trading_client()
        positions = client.get_all_positions()
        if not positions:
            return "No positions."
        lines = []
        for p in positions:
            try:
                lines.append(f"{p.symbol} qty={p.qty} avg=${p.avg_entry_price} side={p.side}")
            except Exception:
                lines.append(f"{getattr(p,'symbol','?')} qty=?")
        return "\n".join(lines)
    except Exception as e:
        return f"Positions error: {e}"

def get_position_side_qty(sym: str) -> Tuple[str, float, float]:
    try:
        client = get_trading_client()
        p = client.get_open_position(broker_symbol(sym))
        side = str(getattr(p, "side", "")).lower()
        qty  = abs(float(getattr(p, "qty", 0) or 0.0))
        avg  = float(getattr(p, "avg_entry_price", 0) or 0.0)
        if qty <= 0:
            return "flat", 0.0, 0.0
        return ("long" if side == "long" else "short"), qty, avg
    except Exception:
        return "flat", 0.0, 0.0

def close_position_market(sym: str) -> str:
    side, qty, _ = get_position_side_qty(sym)
    if qty <= 0:
        return "No position to close."
    if side == "long":
        return place_order(sym, "sell", qty)
    else:
        return place_order(sym, "buy", qty)

# =================
# Journaling (PNG snapshots + CSV + metadata)
# =================
JOURNAL_FRAMES = [
    ("15m", "60d"),
    ("4h",  "2y"),
]

def _new_trade_id() -> str:
    return uuid.uuid4().hex[:12]

def _iso(ts: Optional[float]) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")

def _journal_dir_for(trade_id: str) -> str:
    d = os.path.join(JOURNAL_ROOT, trade_id)
    os.makedirs(d, exist_ok=True)
    return d

def _render_chart_png(sym: str, interval: str, period: str, out_path: str) -> Optional[str]:
    try:
        df = get_history(sym, interval=interval, period=period)
        if df is None or df.empty:
            return None
        s = to_close_series(df)
        if s.empty:
            return None

        e20  = ema(s, 20); e50 = ema(s, 50); e200 = ema(s, 200)
        r    = rsi(s, 14)
        _, _, h = macd(s)

        fig = plt.figure(figsize=(12, 8), dpi=120)

        ax1 = plt.subplot(3,1,1)
        ax1.plot(s.index, s.values, label="Close")
        ax1.plot(e20.index,  e20.values,  label="EMA20")
        ax1.plot(e50.index,  e50.values,  label="EMA50")
        ax1.plot(e200.index, e200.values, label="EMA200")
        ax1.set_title(f"{sym} — {interval}/{period}")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left")

        ax2 = plt.subplot(3,1,2, sharex=ax1)
        ax2.plot(r.index, r.values, label="RSI14")
        ax2.axhline(70, linestyle="--", alpha=0.4)
        ax2.axhline(30, linestyle="--", alpha=0.4)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper left")

        ax3 = plt.subplot(3,1,3, sharex=ax1)
        ax3.bar(h.index, h.values, width=0.8, align="center")
        ax3.axhline(0.0, color="black", linewidth=1)
        ax3.grid(True, alpha=0.3)
        ax3.set_title("MACD Histogram")

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path
    except Exception as e:
        log_error("_render_chart_png", e)
        return None

def journal_capture_entry(sym: str, side: str, qty: float, entry_price: float, entry_time: float) -> str:
    trade_id = _new_trade_id()
    snap_dir = _journal_dir_for(trade_id)
    meta = {
        "trade_id": trade_id,
        "symbol": broker_symbol(sym),
        "side": side.upper(),
        "qty": float(qty),
        "entry_time": entry_time,
        "entry_price": float(entry_price),
        "frames": JOURNAL_FRAMES,
        "snapshots": [],
    }
    for (itv, per) in JOURNAL_FRAMES:
        png_path = os.path.join(snap_dir, f"{broker_symbol(sym)}_{itv}_{per}.png")
        out = _render_chart_png(sym, itv, per, png_path)
        if out:
            meta["snapshots"].append({"interval": itv, "period": per, "path": out})
    with open(os.path.join(snap_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(JOURNAL_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            trade_id, datetime.utcnow().strftime("%Y-%m-%d"),
            broker_symbol(sym), side.upper(), qty,
            _iso(entry_time), f"{entry_price:.6f}",
            "", "", "", "", "",
            json.dumps(JOURNAL_FRAMES), "", snap_dir
        ])
    _entry_time_by_sym[sym] = entry_time
    return trade_id

def journal_finalize_exit(trade_id: str, exit_price: float, exit_time: float) -> Optional[str]:
    try:
        snap_dir = os.path.join(JOURNAL_ROOT, trade_id)
        meta_path = os.path.join(snap_dir, "metadata.json")
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        entry_price = float(meta.get("entry_price", 0.0) or 0.0)
        side = (meta.get("side") or "").upper()
        pnl_abs = (exit_price - entry_price) * (1 if side == "BUY" else -1)
        pnl_pct = (pnl_abs / entry_price) if entry_price else 0.0
        outcome = "WIN" if pnl_abs > 0 else ("LOSS" if pnl_abs < 0 else "BREAKEVEN")

        rows = []
        with open(JOURNAL_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        hdr = rows[0]
        idx = None
        for i in range(1, len(rows)):
            if rows[i][0] == trade_id:
                idx = i
                break
        if idx is None:
            return None

        def col(name): return hdr.index(name)

        rows[idx][col("exit_time")]     = _iso(exit_time)
        rows[idx][col("exit_price")]    = f"{exit_price:.6f}"
        rows[idx][col("pnl_abs")]       = f"{pnl_abs:.6f}"
        rows[idx][col("pnl_pct")]       = f"{pnl_pct:.6f}"
        rows[idx][col("outcome_label")] = outcome
        with open(JOURNAL_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(hdr); w.writerows(rows[1:])
        meta["exit_time"]  = exit_time
        meta["exit_price"] = float(exit_price)
        meta["pnl_abs"]    = pnl_abs
        meta["pnl_pct"]    = pnl_pct
        meta["outcome"]    = outcome
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return outcome
    except Exception as e:
        log_error("journal_finalize_exit", e)
        return None

def _safe_journal_entry(sym, side, qty, price, ts):
    try:
        return journal_capture_entry(sym, side, qty, price, ts)
    except Exception:
        return None

def _safe_journal_exit(tid, price, ts):
    try:
        return journal_finalize_exit(tid, price, ts)
    except Exception:
        return None

# =================
# Sizing
# =================
def calc_order_qty(sym: str, price: Optional[float]) -> Optional[float]:
    try:
        px = float(price or 0.0)
        if px <= 0:
            return None
        budget = float(MAX_TRADE_USD)
        if budget <= 0:
            return None
        if is_crypto_symbol(sym):
            return round(max(0.0001, budget / px), 4)
        q = int(math.floor(budget / px))
        return q if q >= 1 else None
    except Exception as e:
        log_error("calc_order_qty", e)
        return None

# =================
# Orders
# =================
def place_order(symbol: str, side: str, qty: float) -> str:
    try:
        client = get_trading_client()
        sd = (side or "").strip().lower()
        if sd not in ("buy", "sell"):
            return f"Invalid side: {side}"

        symN = norm_symbol(symbol)
        bro  = broker_symbol(symN)
        pos_side, pos_qty, _ = get_position_side_qty(symN)

        if pos_side == "flat" and sd == "buy" and not LONGING_ENABLED:
            return "Longing disabled. Use 'long on' to enable."
        if pos_side == "flat" and sd == "sell" and not SHORTING_ENABLED:
            return "Shorting disabled. Use 'short on' to enable."
        if pos_side == "flat" and sd == "sell" and is_crypto_symbol(symN) and not CRYPTO_SHORTING_ENABLED:
            return "Crypto shorting disabled. Use 'crypto short on' only if broker supports."

        broker_qty = qty if is_crypto_symbol(symN) else int(qty)
        req = MarketOrderRequest(
            symbol=bro,
            qty=str(broker_qty),
            side=(OrderSide.BUY if sd == "buy" else OrderSide.SELL),
            time_in_force=TimeInForce.DAY
        )
        order = client.submit_order(order_data=req)

        now = time.time()
        px  = get_price(symN) or 0.0

        if pos_side == "flat" and sd == "buy":
            tid = _safe_journal_entry(symN, "BUY", qty, float(px), now)
            if tid: _open_trade_ids[symN] = tid
        elif pos_side == "flat" and sd == "sell":
            tid = _safe_journal_entry(symN, "SELL", qty, float(px), now)
            if tid: _open_trade_ids[symN] = tid
        elif pos_side == "long" and sd == "sell":
            tid = _open_trade_ids.get(symN)
            if tid and px: _safe_journal_exit(tid, float(px), now)
            _open_trade_ids.pop(symN, None)
        elif pos_side == "short" and sd == "buy":
            tid = _open_trade_ids.get(symN)
            if tid and px: _safe_journal_exit(tid, float(px), now)
            _open_trade_ids.pop(symN, None)

        record_trade(symN)
        log_auto(f"[ORDER OK] {order.symbol} {order.side} qty={order.qty} id={order.id}")
        return f"OK order_id={order.id} {order.symbol} {order.side} qty={order.qty}"
    except Exception as e:
        msg = f"Order error: {e}"
        log_error("place_order", e)
        return msg

# =================
# Risk: forced exits (SL/TP + Max-Hold)
# =================
def evaluate_forced_exit(sym: str) -> bool:
    side, qty, avg = get_position_side_qty(sym)
    if side == "flat" or avg <= 0:
        return False

    px = get_price(sym)
    if not px:
        return False

    if side == "long":
        if px <= avg * (1 - STOP_LOSS_PCT) or px >= avg * (1 + TAKE_PROFIT_PCT):
            res = place_order(sym, "sell", qty)
            log_auto(f"[FORCED EXIT] {broker_symbol(sym)} long→flat @ {px:.4f} | {res}")
            return True
    else:
        if px >= avg * (1 + STOP_LOSS_PCT) or px <= avg * (1 - TAKE_PROFIT_PCT):
            res = place_order(sym, "buy", qty)
            log_auto(f"[FORCED EXIT] {broker_symbol(sym)} short→flat @ {px:.4f} | {res}")
            return True

    et = _entry_time_by_sym.get(sym)
    if et and (time.time() - et) >= (MAX_HOLD_MIN * 60):
        res = close_position_market(sym)
        log_auto(f"[MAX-HOLD EXIT] {broker_symbol(sym)} after {MAX_HOLD_MIN}m | {res}")
        return True
    return False

# =================
# Live-scores
# =================
def format_live_scores_line(sym: str) -> str:
    price_str = "?"
    tscore = 50
    try:
        p = get_price(sym)
        if isinstance(p, (int, float)):
            price_str = f"${p:.2f}"
    except Exception as e:
        log_error(f"format_live_scores_line.get_price({sym})", e)
    try:
        df = get_history(sym)
        if df is not None and not df.empty:
            tscore = trend_score(df)
    except Exception as e:
        log_error(f"format_live_scores_line.trend({sym})", e)
    return f"{broker_symbol(sym)}: Price {price_str} | TrendScore {tscore}/100"

def live_scores_loop():
    while True:
        try:
            if LIVE_SCORES_ACTIVE:
                lines = []
                for sym in WATCHLIST:
                    try:
                        lines.append(format_live_scores_line(sym))
                    except Exception as e:
                        log_error(f"live_scores_line {sym}", e)
                if lines:
                    log_auto("[AUTO LIVE SCORES]\n" + "\n".join(lines))
        except Exception as e:
            log_error("live_scores_loop", e)
        time.sleep(LIVE_SCORES_INTERVAL)

# =================
# RL helpers (training/backtest)
# =================
class TradingEnv(Env):
    """Single-symbol trading environment for PPO with API-compatible returns."""
    def __init__(self, df, initial_balance=10000):
        super().__init__()
        self.df = df.dropna().reset_index()
        self.initial_balance = float(initial_balance)
        self.current_step = 0
        self.balance = float(initial_balance)
        self.shares_held = 0.0
        self.avg_entry_price = 0.0
        self.previous_portfolio_value = float(initial_balance)
        self.action_space = Discrete(3)  # 0 HOLD, 1 BUY, 2 SELL
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    def _next_observation(self):
        price_series = _series1d(self.df['Close'].iloc[:self.current_step + 1]).dropna()
        if len(price_series) < 50:
            rsi_val = 50.0
            macd_hist_val = 0.0
            last_close = float(price_series.iloc[-1]) if len(price_series) else 0.0
            ema20_val = last_close
            ema50_val = last_close
        else:
            rsi_series = rsi(price_series, 14)
            rsi_val = _scalar(rsi_series.iloc[-1], 50.0) if not rsi_series.empty else 50.0
            m_tuple = macd(price_series)
            if not isinstance(m_tuple, tuple) or len(m_tuple) != 3:
                macd_hist_val = 0.0
            else:
                _, _, macd_hist_series = m_tuple
                macd_hist_val = _scalar(macd_hist_series.iloc[-1], 0.0) if not getattr(macd_hist_series, "empty", True) else 0.0
            last_close = float(price_series.iloc[-1]) if len(price_series) else 0.0
            ema20_val = _scalar(ema(price_series, 20).iloc[-1], last_close)
            ema50_val = _scalar(ema(price_series, 50).iloc[-1], last_close)

        last_close = float(price_series.iloc[-1]) if len(price_series) else 0.0
        obs = np.array([
            last_close, float(rsi_val), float(macd_hist_val),
            float(ema20_val), float(ema50_val),
            float(self.shares_held), float(self.balance)
        ], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        current_price = _scalar(self.df.loc[self.current_step, "Close"], 0.0)

        if action == 1 and self.balance > current_price:
            shares_to_buy = int(self.balance // current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                self.balance -= cost
                if self.shares_held + shares_to_buy > 0:
                    self.avg_entry_price = (
                        (self.shares_held * self.avg_entry_price) +
                        (shares_to_buy * current_price)
                    ) / (self.shares_held + shares_to_buy)
                self.shares_held += shares_to_buy

        elif action == 2 and self.shares_held > 0:
            revenue = self.shares_held * current_price
            self.balance += revenue
            self.shares_held = 0
            self.avg_entry_price = 0.0

        portfolio_value = self.balance + (self.shares_held * current_price)
        reward = float(portfolio_value - self.previous_portfolio_value)
        self.previous_portfolio_value = portfolio_value

        obs_next = self._next_observation().astype(np.float32)
        if self.balance < current_price and self.shares_held == 0:
            done = True

        if _SB3_USES_GYMNASIUM:
            return obs_next, float(reward), bool(done), False, {}
        else:
            return obs_next, float(reward), bool(done), {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = float(self.initial_balance)
        self.shares_held = 0.0
        self.avg_entry_price = 0.0
        self.previous_portfolio_value = float(self.initial_balance)
        obs0 = self._next_observation()
        if _SB3_USES_GYMNASIUM:
            return obs0, {}
        else:
            return obs0

# --- Training data
def get_training_data(symbol: str, period: Optional[str] = None, interval: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Fetches Close-only frame at the requested/global cadence."""
    per = period or INDICATOR_PERIOD
    itv = interval or INDICATOR_INTERVAL
    df = get_history(symbol, interval=itv, period=per)
    if df is None or df.empty:
        log_error("get_training_data", f"empty df for {symbol} with interval={itv}, period={per}")
        return None
    s = to_close_series(df)
    if s is None or s.empty:
        log_error("get_training_data", f"no Close series for {symbol} with interval={itv}, period={per}")
        return None
    return pd.DataFrame({"Close": s})

# --- KPI rollouts (deterministic)
def _rollout_metrics(df: pd.DataFrame, model: PPO, initial_balance: float = 10000.0) -> Dict[str, float]:
    """
    Deterministic pass over the dataset with forced liquidation at the end
    to compute Balance Δ, Trades, Wins, Hit-rate.
    Uses positional indexing (.iloc) to avoid DatetimeIndex KeyErrors.
    """
    # Safety: require at least 2 closes
    df = df.dropna().copy()
    if "Close" not in df or len(df) < 2:
        return {
            "start_balance": float(initial_balance),
            "end_balance": float(initial_balance),
            "delta": 0.0,
            "trades": 0.0,
            "wins": 0.0,
            "hit_rate": 0.0,
        }

    env = TradingEnv(df, initial_balance=initial_balance)
    if _SB3_USES_GYMNASIUM:
        obs, _ = env.reset()
    else:
        obs = env.reset()

    trades = 0
    wins = 0
    last_entry_px = None

    closes = df["Close"].reset_index(drop=True)  # ensure pure positional indexing

    while True:
        action, _ = model.predict(obs, deterministic=True)

        # If we intend to buy next step and have cash > next price, note current entry price
        next_idx = min(env.current_step + 1, len(closes) - 1)
        next_px = _scalar(closes.iloc[next_idx], 0.0)
        if action == 1 and env.balance > next_px:
            curr_idx = min(env.current_step, len(closes) - 1)
            last_entry_px = _scalar(closes.iloc[curr_idx], 0.0)

        if _SB3_USES_GYMNASIUM:
            obs, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
        else:
            obs, reward, done, _info = env.step(action)

        if done:
            break

    # FORCE LIQUIDATION to realize PnL
    term_idx = min(env.current_step, len(closes) - 1)
    term_price = _scalar(closes.iloc[term_idx], 0.0)
    if env.shares_held > 0:
        env.balance += env.shares_held * term_price
        trades += 1
        if last_entry_px is not None and term_price > last_entry_px:
            wins += 1
        env.shares_held = 0
        env.avg_entry_price = 0.0

    end_balance = float(env.balance)
    hit_rate = (wins / trades * 100.0) if trades > 0 else 0.0
    return {
        "start_balance": float(initial_balance),
        "end_balance": end_balance,
        "delta": end_balance - float(initial_balance),
        "trades": float(trades),
        "wins": float(wins),
        "hit_rate": float(hit_rate),
    }


# --- Progress callback
class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, symbol: str, log_every: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.total = max(1, int(total_timesteps))
        self.symbol = symbol
        self.every = max(1, int(log_every))
        self.last = 0

    def _on_step(self) -> bool:
        n = int(self.num_timesteps)
        if n - self.last >= self.every or n == self.total:
            pct = n / self.total * 100.0
            msg = f"[TRAIN] {self.symbol} {n}/{self.total} ({pct:.1f}%)"
            print(msg, file=sys.stdout, flush=True)
            log_auto(msg)
            self.last = n
        return True

# --- Train / Continue / Backtest
def train_rl_model(symbol: str,
                   period: Optional[str] = None,
                   interval: Optional[str] = None,
                   init_from: Optional[str] = None,
                   total_timesteps: int = 20000) -> str:
    sym = symbol.upper()
    df = get_training_data(sym, period=period, interval=interval)
    if df is None or df.empty:
        return f"No training data for {sym}."
    initial_balance = 10000.0

    def make_env():
        return TradingEnv(df, initial_balance=initial_balance)
    env = DummyVecEnv([make_env])

    seed_from = init_from or GLOBAL_BASE_MODEL
    try:
        if seed_from:
            model = PPO.load(rl_model_path(seed_from.upper()), env=env, device="cpu", verbose=1)
        else:
            model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    except Exception:
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")

    cb = ProgressCallback(total_timesteps=total_timesteps, symbol=sym, log_every=1000)
    model.learn(total_timesteps=total_timesteps, callback=cb)
    path = rl_model_path(sym)
    model.save(path)
    RL_MODELS[sym] = model

    MODEL_REGISTRY[sym] = {
        "path": path + ".zip",
        "fingerprint": _model_fingerprint(model),
        "trained_at": datetime.utcnow().isoformat(timespec="seconds"),
        "init_from": seed_from
    }
    _save_model_registry(MODEL_REGISTRY)

    metrics = _rollout_metrics(df, model, initial_balance=initial_balance)
    return (f"✅ Trained PPO for {sym} | steps={total_timesteps} | fp={MODEL_REGISTRY[sym]['fingerprint']}\n"
            f"Cadence: interval={interval or INDICATOR_INTERVAL}, period={period or INDICATOR_PERIOD}\n"
            f"Virtual balance: start=${metrics['start_balance']:,.2f} → end=${metrics['end_balance']:,.2f} "
            f"(Δ=${metrics['delta']:,.2f}) | Trades={int(metrics['trades'])} | Wins={int(metrics['wins'])} | "
            f"Hit rate={metrics['hit_rate']:.1f}%")

def continue_rl_model(symbol: str,
                      more_steps: int = 20480,
                      period: Optional[str] = None,
                      interval: Optional[str] = None) -> str:
    sym = symbol.upper()
    path = rl_model_path(sym)
    df = get_training_data(sym, period=period, interval=interval)
    if df is None or df.empty:
        return f"No training data for {sym}."
    initial_balance = 10000.0

    def make_env():
        return TradingEnv(df, initial_balance=initial_balance)
    env = DummyVecEnv([make_env])
    try:
        model = PPO.load(path, env=env, device="cpu", verbose=1)
    except Exception:
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")

    cb = ProgressCallback(total_timesteps=more_steps, symbol=sym, log_every=1000)
    model.learn(total_timesteps=more_steps, callback=cb)
    model.save(path)
    RL_MODELS[sym] = model

    MODEL_REGISTRY[sym] = {
        "path": path + ".zip",
        "fingerprint": _model_fingerprint(model),
        "continued_at": datetime.utcnow().isoformat(timespec="seconds"),
        "init_from": MODEL_REGISTRY.get(sym, {}).get("init_from")
    }
    _save_model_registry(MODEL_REGISTRY)

    metrics = _rollout_metrics(df, model, initial_balance=initial_balance)
    return (f"✅ Continued PPO for {sym} | +steps={more_steps} | fp={MODEL_REGISTRY[sym]['fingerprint']}\n"
            f"Cadence: interval={interval or INDICATOR_INTERVAL}, period={period or INDICATOR_PERIOD}\n"
            f"Virtual balance: start=${metrics['start_balance']:,.2f} → end=${metrics['end_balance']:,.2f} "
            f"(Δ=${metrics['delta']:,.2f}) | Trades={int(metrics['trades'])} | Wins={int(metrics['wins'])} | "
            f"Hit rate={metrics['hit_rate']:.1f}%")

def backtest_model_on_history(symbol: str, period: Optional[str] = None, interval: Optional[str] = None) -> str:
    sym = symbol.upper()
    df = get_training_data(sym, period=period, interval=interval)
    if df is None or df.empty:
        return f"No backtest data for {sym}."
    try:
        model = RL_MODELS.get(sym) or PPO.load(rl_model_path(sym), device="cpu")
    except Exception as e:
        # Cold stub model if loading fails
        log_error("backtest_load_model", e)
        model = PPO("MlpPolicy", DummyVecEnv([lambda: TradingEnv(df)]), verbose=0, device="cpu")

    metrics = _rollout_metrics(df, model, initial_balance=10000.0)
    return (f"Backtest {sym} @ interval={interval or INDICATOR_INTERVAL}, period={period or INDICATOR_PERIOD}\n"
            f"Virtual balance: start=${metrics['start_balance']:,.2f} → end=${metrics['end_balance']:,.2f} "
            f"(Δ=${metrics['delta']:,.2f}) | Trades={int(metrics['trades'])} | Wins={int(metrics['wins'])} | "
            f"Hit rate={metrics['hit_rate']:.1f}%")

# =================
# Trading loop
# =================
def trading_loop():
    while True:
        try:
            if TRADING_ACTIVE and intraday_window_open():
                for sym in list(WATCHLIST):
                    try:
                        if evaluate_forced_exit(sym):
                            continue

                        df = get_history(sym)
                        if df is None or df.empty:
                            continue
                        current_price = get_price(sym) or 0.0
                        if current_price <= 0:
                            continue

                        if not can_trade_symbol(sym):
                            continue
                        lt = _last_trade_time.get(sym, 0)
                        if (time.time() - lt) < TRADE_COOLDOWN_SEC:
                            continue

                        side_signal = "HOLD"
                        if sym in RL_MODELS:
                            try:
                                c_series = to_close_series(df)
                                macd_hist = macd(c_series)[2]
                                mh_last = float(macd_hist.iloc[-1]) if not macd_hist.empty else 0.0
                                obs_vector = np.array([
                                    float(current_price),
                                    float(rsi(c_series, 14).iloc[-1]),
                                    float(mh_last),
                                    float(ema(c_series, 20).iloc[-1]),
                                    float(ema(c_series, 50).iloc[-1]),
                                    0.0, 10000.0
                                ], dtype=np.float32)
                                action, _ = RL_MODELS[sym].predict(obs_vector)
                                action = int(action)
                                side_signal = "BUY" if action == 1 else ("SELL" if action == 2 else "HOLD")
                            except Exception as e:
                                log_error(f"RL prediction {sym}", e)
                                tscore = int(trend_score(df))
                                side_signal = trend_decision(tscore)
                        else:
                            tscore = int(trend_score(df))
                            side_signal = trend_decision(tscore)

                        buf = _sigbuf_trend[sym]
                        if buf.maxlen != TREND_PERSIST:
                            _sigbuf_trend[sym] = deque(buf, maxlen=TREND_PERSIST)
                            buf = _sigbuf_trend[sym]
                        buf.append(side_signal)
                        persistent = (len(buf) == TREND_PERSIST and all(x == side_signal for x in buf))
                        if not persistent or side_signal == "HOLD":
                            continue

                        qty = calc_order_qty(sym, current_price)
                        if not qty:
                            continue

                        res = place_order(sym, "buy" if side_signal == "BUY" else "sell", qty)
                        log_auto(f"[AUTO {side_signal}] {broker_symbol(sym)} qty={qty} -> {res}")

                    except Exception as e:
                        log_error(f"trading_loop per-symbol {sym}", e)
                        continue
        except Exception as e:
            log_error("trading_loop outer", e)

        time.sleep(POLL_INTERVAL_S)

# =================
# EOD flatten loop
# =================
def flatten_all() -> str:
    try:
        client = get_trading_client()
        poss = client.get_all_positions()
        if not poss:
            return "No positions to flatten."
        errs = []
        for p in poss:
            try:
                client.close_position(p.symbol)
            except Exception as pe:
                errs.append(f"{p.symbol}:{pe}")
        return "Flatten complete." if not errs else "Flatten partial: " + "; ".join(errs)
    except Exception as e:
        return f"Flatten error: {e}"

def eod_flatten_loop():
    while True:
        try:
            if EOD_FLATTEN and DAY_TRADE_MODE:
                now = _utc_now()
                h2, m2 = DAY_WINDOW_END_UTC
                cutoff = now.replace(hour=h2, minute=m2, second=0, microsecond=0) - timedelta(minutes=5)
                if now >= cutoff:
                    msg = flatten_all()
                    log_auto(f"[EOD FLATTEN] {msg}")
                    time.sleep(300)
        except Exception as e:
            log_error("eod_flatten_loop", e)
        time.sleep(15)

# =================
# Watchlist & Health
# =================
def add_to_watchlist(symbols: List[str]) -> List[str]:
    global WATCHLIST
    try:
        existing = set(WATCHLIST)
        for s in symbols:
            sym = norm_symbol(s)
            if sym and sym not in existing:
                WATCHLIST.append(sym)
                existing.add(sym)
        return WATCHLIST
    except Exception as e:
        log_error("add_to_watchlist", e)
        return WATCHLIST

def health_snapshot() -> str:
    try:
        lines = [
            f"WATCHLIST={WATCHLIST}",
            f"TRADING_ACTIVE={TRADING_ACTIVE} LIVE_SCORES_ACTIVE={LIVE_SCORES_ACTIVE}",
            f"INTERVAL={INDICATOR_INTERVAL} PERIOD={INDICATOR_PERIOD}",
            f"THRESHOLDS: BUY>={TREND_BUY_TH} SELL<={TREND_SELL_TH} PERSIST={TREND_PERSIST} COOLDOWN={TRADE_COOLDOWN_SEC}s",
            f"LIMITS: MAX_TRADE_USD=${MAX_TRADE_USD:.2f} MAX_TRADES_PER_DAY={MAX_TRADES_PER_DAY} MAX_TRADES_PER_SYM={MAX_TRADES_PER_SYM}",
            f"DayMode={DAY_TRADE_MODE} Window(UTC)={DAY_WINDOW_START_UTC}-{DAY_WINDOW_END_UTC} EOD_FLATTEN={EOD_FLATTEN}",
            f"Toggles: Longing={LONGING_ENABLED} Shorting={SHORTING_ENABLED} CryptoShorts={CRYPTO_SHORTING_ENABLED}",
            f"auto_updates={len(_auto_updates)}",
            f"SB3_VERSION={_SB3_VER} GymAPI={'Gymnasium' if _SB3_USES_GYMNASIUM else 'Classic Gym'}",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Health error: {e}"

# =================
# Flask UI / Command Parser
# =================
def _help_text() -> str:
    return """\
HELP — Command Catalog (deep explanations, examples, and notes)

GENERAL
 help
   • Shows this help. Try: `help` anytime.

 status
   • Alpaca account status and open portfolio positions (paper account when configured).
   • Example: `status`

 health
   • Internal config + runtime health snapshot (interval/period, thresholds, toggles).
   • Includes SB3 version and which Gym API is active.
   • Example: `health`

AUTO LOG / FEEDS
 auto feed | feed
   • Prints the last ~100 automated log lines (orders, training progress, errors).
   • Example: `auto feed`

 live scores
   • One-time snapshot of WATCHLIST prices and TrendScore.
   • Example: `live scores`

 start live scores / stop live scores
   • Start/stop background ticker that logs live scores every few seconds.
   • Example: `start live scores`

TRADING (LIVE)
 start trading / stop trading
   • Toggle the autonomous trading loop. Captures session start/end equity and logs P&L.
   • Example: `start trading`

 day mode on/off
   • Enable/disable intraday equities window (defaults to 09:35–15:55 ET).
   • When ON, equities trade only in that UTC window; crypto always allowed.
   • Example: `day mode off`

 set day window: HH:MM-HH:MM UTC
   • Set the equities trading window in UTC.
   • Example: `set day window: 13:35-19:55 UTC`

 eod flatten on/off
   • Toggle forced flatten five minutes before the end of the day window.
   • Example: `eod flatten on`

ORDERS & POSITIONS
 orders
   • List open broker orders.
 cancel all
   • Cancel all open orders.
 cancel <ID>
   • Cancel a specific order by ID.
 flatten all
   • Close all open positions at market.
 close <SYMBOL>
   • Close a specific position at market.
 buy <SYMBOL> [qty] / sell <SYMBOL> [qty]
   • Manual market orders; qty optional (auto-sized by MAX_TRADE_USD if omitted).
   • Examples: `buy NVDA`, `sell AAPL 3`

RISK / LIMITS / BUDGET
 set sl:X / set tp:X
   • Stop-loss / take-profit percentage. Input as %, e.g., `set sl:1.2` or decimal `0.012`.
 set hold:X
   • Max hold minutes before force-close. Example: `set hold:120`
 set cooldown:X
   • Seconds cooldown between same-symbol trades. Example: `set cooldown:180`
 set buy th:X | set sell th:X | set persist:X
   • TrendScore thresholds and required consecutive signals. Example: `set buy th:70`
 set max usd:X
   • Per-trade budget used for order sizing if qty not specified. Example: `set max usd:250`
 set max per day:X | set max per sym:X
   • Trade ceilings per UTC day / per symbol per day.

MARKET DATA CADENCE (LIVE & RL)
 set interval: I P
   • Sets global yfinance interval and period used everywhere (live signals, training, backtest).
   • Examples:
       `set interval: 15m 60d`
       `set interval: 1d 3y`
   • Supported intervals (yfinance): 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
   • Periods: 1d..10y etc.

WATCHLIST
 add <SYMBOLS>
   • Add one or more symbols (space/comma separated). Example: `add NVDA TSLA` or `add nvda, aapl`
 add btc
   • Quick add BTC-USD.

TOGGLES
 long on/off | short on/off | crypto short on/off
   • Enable/disable long entries, short entries, or crypto shorting (if broker supports).

RL (PPO) — TRAIN / CONTINUE / BACKTEST / MODELS
 train model SYMBOL [PERIOD INTERVAL] [steps N] [from SRC]
   • Train a PPO policy at the current/global cadence.
   • PERIOD/INTERVAL optional if you already set them globally via `set interval:`.
   • To seed weights from another model, append `from SRC`.
   • Examples:
       `train model NVDA steps 30000`
       `train model NVDA 60d 15m steps 50000`
       `train model AAPL steps 20000 from NVDA`
   • Shows progress in console and in Auto Feed every ~1000 steps.
   • On completion, prints virtual balance start→end, trades, wins, hit-rate.

 continue model SYMBOL [STEPS]
   • Resume training from the latest checkpoint for SYMBOL.
   • Uses current/global cadence unless you pass explicit [PERIOD INTERVAL] in `set interval:`.
   • Example: `continue model NVDA 20000`

 backtest model SYMBOL [PERIOD INTERVAL]
   • Deterministic walk with the trained model across the data window (no training).
   • Returns balance start→end, trades, wins, hit-rate.
   • Example: `backtest model NVDA 60d 15m` or just `backtest model NVDA`

 load model SYMBOL
   • Load a saved model from disk (rl_trading_model_SYMBOL.zip) into memory.

 list models
   • List loaded models, fingerprints, and origins (linked/seeded).

 snapshot models
   • Save snapshot copies of all loaded models with a UTC timestamp suffix.

 link model SRC DST
   • Copy SRC model weights into a DST environment (transplant), save as DST, and keep both loaded.

 set base model: SYMBOL
   • Specify a default seed model used when training without `from SRC`.
"""

@app.route("/", methods=["GET","POST"])
def index():
    global TRADING_ACTIVE, LIVE_SCORES_ACTIVE
    global LONGING_ENABLED, SHORTING_ENABLED, CRYPTO_SHORTING_ENABLED
    global DAY_TRADE_MODE, DAY_WINDOW_START_UTC, DAY_WINDOW_END_UTC, EOD_FLATTEN
    global STOP_LOSS_PCT, TAKE_PROFIT_PCT, MAX_HOLD_MIN
    global TREND_BUY_TH, TREND_SELL_TH, TREND_PERSIST, TRADE_COOLDOWN_SEC
    global MAX_TRADE_USD, INDICATOR_INTERVAL, INDICATOR_PERIOD, MAX_TRADES_PER_DAY, MAX_TRADES_PER_SYM
    global DEBUG_TICKS, _sigbuf_trend, RL_MODELS, WATCHLIST, LIVE_SESSION_START_EQUITY
    global GLOBAL_BASE_MODEL, MODEL_REGISTRY

    out = ""
    if request.method == "POST":
        raw = request.form.get("cmd", "")
        low = norm_cmd(raw)
        cmd = raw.strip()

        try:
            if low == "help":
                out = _help_text()

            elif low == "status":
                out = get_account_status_text() + "\n" + portfolio_status()
            elif low == "health":
                out = health_snapshot()

            elif low == "start trading":
                TRADING_ACTIVE = True
                try:
                    LIVE_SESSION_START_EQUITY = get_equity_value()
                    snap = equity_snapshot_text("[SESSION START] ")
                    log_auto(snap)
                    out = "Trading ENABLED\n" + snap
                except Exception as e:
                    out = "Trading ENABLED (equity capture failed.)\n" + _exc(e)

            elif low == "stop trading":
                TRADING_ACTIVE = False
                try:
                    end_eq = get_equity_value()
                    delta = end_eq - (LIVE_SESSION_START_EQUITY or end_eq)
                    pct = (delta / LIVE_SESSION_START_EQUITY * 100.0) if LIVE_SESSION_START_EQUITY else 0.0
                    snap = equity_snapshot_text("[SESSION END]  ")
                    summary = f"[SESSION P&L] Δ=${delta:,.2f} ({pct:.2f}%)"
                    log_auto(snap); log_auto(summary)
                    out = "Trading DISABLED\n" + snap + "\n" + summary
                except Exception as e:
                    out = "Trading DISABLED (equity capture failed.)\n" + _exc(e)

            elif low == "start live scores":
                LIVE_SCORES_ACTIVE = True
                try:
                    lines = [format_live_scores_line(s) for s in WATCHLIST]
                    if lines:
                        log_auto("[AUTO LIVE SCORES]\n" + "\n".join(lines))
                except Exception as e:
                    log_error("start live scores snapshot", e)
                out = "Live scores ENABLED"
            elif low == "stop live scores":
                LIVE_SCORES_ACTIVE = False
                out = "Live scores DISABLED"
            elif low in ("auto feed", "autofeed", "auto_feed", "feed"):
                out = "\n".join(list(_auto_updates)[-100:]) or "No auto updates yet."
            elif low == "live scores":
                out = "\n".join([format_live_scores_line(s) for s in WATCHLIST])

            # --- SETTERS / TOGGLES ---
            elif low.startswith("set sl:"):
                try:
                    val_raw = cmd.split("set sl:",1)[1].strip().replace("%","")
                    v = float(val_raw)
                    STOP_LOSS_PCT = v/100.0 if v > 1 else v
                    out = f"STOP_LOSS_PCT={STOP_LOSS_PCT:.4f}"
                except Exception as e:
                    out = f"Bad SL: {e}"

            elif low.startswith("set tp:"):
                try:
                    val_raw = cmd.split("set tp:",1)[1].strip().replace("%","")
                    v = float(val_raw)
                    TAKE_PROFIT_PCT = v/100.0 if v > 1 else v
                    out = f"TAKE_PROFIT_PCT={TAKE_PROFIT_PCT:.4f}"
                except Exception as e:
                    out = f"Bad TP: {e}"

            elif low.startswith("set hold:"):
                try:
                    MAX_HOLD_MIN = max(1, int(cmd.split("set hold:",1)[1].strip()))
                    out = f"MAX_HOLD_MIN={MAX_HOLD_MIN}"
                except Exception as e:
                    out = f"Bad hold: {e}"

            elif low.startswith("set cooldown:"):
                try:
                    TRADE_COOLDOWN_SEC = max(0, int(cmd.split("set cooldown:",1)[1].strip()))
                    out = f"TRADE_COOLDOWN_SEC={TRADE_COOLDOWN_SEC}s"
                except Exception as e:
                    out = f"Bad cooldown: {e}"

            elif low.startswith("set buy th:"):
                try:
                    TREND_BUY_TH = max(0, min(100, int(cmd.split("set buy th:",1)[1].strip())))
                    out = f"TREND_BUY_TH={TREND_BUY_TH}"
                except Exception as e:
                    out = f"Bad buy threshold: {e}"

            elif low.startswith("set sell th:"):
                try:
                    TREND_SELL_TH = max(0, min(100, int(cmd.split("set sell th:",1)[1].strip())))
                    out = f"TREND_SELL_TH={TREND_SELL_TH}"
                except Exception as e:
                    out = f"Bad sell threshold: {e}"

            elif low.startswith("set persist:"):
                try:
                    TREND_PERSIST = max(1, int(cmd.split("set persist:",1)[1].strip()))
                    for k in list(_sigbuf_trend.keys()):
                        _sigbuf_trend[k] = deque(_sigbuf_trend[k], maxlen=TREND_PERSIST)
                    out = f"TREND_PERSIST={TREND_PERSIST}"
                except Exception as e:
                    out = f"Bad persist: {e}"

            elif low.startswith("set max usd:"):
                try:
                    MAX_TRADE_USD = max(1.0, float(cmd.split("set max usd:",1)[1].strip()))
                    out = f"MAX_TRADE_USD=${MAX_TRADE_USD:.2f}"
                except Exception as e:
                    out = f"Bad max usd: {e}"

            elif low.startswith("set max per day:"):
                try:
                    MAX_TRADES_PER_DAY = max(1, int(cmd.split("set max per day:",1)[1].strip()))
                    out = f"MAX_TRADES_PER_DAY={MAX_TRADES_PER_DAY}"
                except Exception as e:
                    out = f"Bad max per day: {e}"

            elif low.startswith("set max per sym:"):
                try:
                    MAX_TRADES_PER_SYM = max(1, int(cmd.split("set max per sym:",1)[1].strip()))
                    out = f"MAX_TRADES_PER_SYM={MAX_TRADES_PER_SYM}"
                except Exception as e:
                    out = f"Bad max per sym: {e}"

            elif low.startswith("set interval:"):
                try:
                    tail = cmd.split("set interval:",1)[1].strip()
                    parts = tail.split()
                    if len(parts) < 2:
                        raise ValueError("Usage: set interval: 5m 60d")
                    INDICATOR_INTERVAL = parts[0]
                    INDICATOR_PERIOD  = parts[1]
                    out = f"Interval/Period → {INDICATOR_INTERVAL} / {INDICATOR_PERIOD} (LIVE & RL)"
                except Exception as e:
                    out = f"Bad interval/period: {e}"

            elif low == "day mode on":
                DAY_TRADE_MODE = True
                out = "Day trading window ENABLED"
            elif low == "day mode off":
                DAY_TRADE_MODE = False
                out = "Day trading window DISABLED"

            elif low.startswith("set day window:"):
                try:
                    m = re.search(r"set day window:\s*(\d{2}):(\d{2})-(\d{2}):(\d{2})\s*utc", low)
                    if not m:
                        raise ValueError("Format: set day window: HH:MM-HH:MM UTC")
                    h1, m1, h2, m2 = map(int, m.groups())
                    DAY_WINDOW_START_UTC = (h1, m1)
                    DAY_WINDOW_END_UTC = (h2, m2)
                    out = f"Day window (UTC): {DAY_WINDOW_START_UTC} → {DAY_WINDOW_END_UTC}"
                except Exception as e:
                    out = f"Bad day window: {e}"

            elif low == "eod flatten on":
                EOD_FLATTEN = True
                out = "EOD flatten ENABLED"
            elif low == "eod flatten off":
                EOD_FLATTEN = False
                out = "EOD flatten DISABLED"

            elif low == "long on":
                LONGING_ENABLED = True
                out = "Longing ENABLED"
            elif low == "long off":
                LONGING_ENABLED = False
                out = "Longing DISABLED"

            elif low == "short on":
                SHORTING_ENABLED = True
                out = "Shorting ENABLED"
            elif low == "short off":
                SHORTING_ENABLED = False
                out = "Shorting DISABLED"

            elif low == "crypto short on":
                CRYPTO_SHORTING_ENABLED = True
                out = "Crypto shorting ENABLED"
            elif low == "crypto short off":
                CRYPTO_SHORTING_ENABLED = False
                out = "Crypto shorting DISABLED"

            elif low.startswith("add "):
                tail = cmd.split("add ",1)[1].strip()
                if tail.lower() == "btc":
                    add_to_watchlist(["BTC-USD"])
                    out = f"WATCHLIST={WATCHLIST}"
                else:
                    tokens = [t.strip() for t in re.split(r"[,\s]+", tail) if t.strip()]
                    add_to_watchlist(tokens)
                    out = f"WATCHLIST={WATCHLIST}"

            # --- Orders / Positions ---
            elif low == "orders":
                try:
                    orders = list_open_orders()
                    if not orders:
                        out = "No open orders."
                    else:
                        out = "\n".join([f"{o.id} {o.symbol} {o.side} qty={o.qty} status={o.status}" for o in orders])
                except Exception as e:
                    out = f"Orders error: {e}"

            elif low == "cancel all":
                out = cancel_all_orders()

            elif low.startswith("cancel "):
                try:
                    oid = cmd.split("cancel ",1)[1].strip()
                    out = cancel_order(oid)
                except Exception as e:
                    out = f"Cancel error: {e}"

            elif low == "flatten all":
                out = flatten_all()

            elif low.startswith("close "):
                try:
                    sym = cmd.split("close ",1)[1].strip().upper()
                    out = close_position_market(sym)
                except Exception as e:
                    out = f"Close error: {e}"

            elif low.startswith("buy ") or low.startswith("sell "):
                try:
                    parts = cmd.split()
                    side = parts[0].lower()
                    sym  = parts[1].upper()
                    qty  = float(parts[2]) if len(parts) >= 3 else None
                    px = get_price(sym)
                    q = qty if qty else calc_order_qty(sym, px)
                    out = place_order(sym, side, q) if q else f"Cannot size order for {sym}."
                except Exception as e:
                    out = f"Manual order error: {e}"

            # --- Model utilities ---
            elif low == "snapshot models":
                try:
                    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
                    for sym, model in list(RL_MODELS.items()):
                        base = rl_model_path(sym)
                        model.save(base)
                        model.save(f"{base}.{ts}.zip")
                    out = f"Snapshot complete @ {ts}Z"
                except Exception as e:
                    out = f"Snapshot error: {e}"

            elif low.startswith("link model "):
                try:
                    parts = cmd.split()
                    if len(parts) < 4:
                        out = "Usage: link model SRC DST"
                    else:
                        src = parts[2].upper()
                        dst = parts[3].upper()
                        if src == dst:
                            out = "SRC and DST must differ."
                        else:
                            df = get_training_data(dst, period=INDICATOR_PERIOD, interval=INDICATOR_INTERVAL)
                            if df is None or df.empty:
                                out = f"No data to build DST env for {dst}."
                            else:
                                env = DummyVecEnv([lambda: TradingEnv(df)])
                                model = PPO.load(rl_model_path(src), env=env, device="cpu")
                                RL_MODELS[dst] = model
                                model.save(rl_model_path(dst))
                                MODEL_REGISTRY[dst] = {
                                    "path": rl_model_path(dst) + ".zip",
                                    "fingerprint": _model_fingerprint(model),
                                    "linked_from": src,
                                    "linked_at": datetime.utcnow().isoformat(timespec="seconds"),
                                    "init_from": src
                                }
                                _save_model_registry(MODEL_REGISTRY)
                                out = f"✅ {dst} initialized from {src}."
                except Exception as e:
                    out = f"Link error:\n{_exc(e)}"

            elif low.startswith("set base model:"):
                try:
                    GLOBAL_BASE_MODEL = cmd.split("set base model:",1)[1].strip().upper()
                    out = f"GLOBAL_BASE_MODEL={GLOBAL_BASE_MODEL}"
                except Exception as e:
                    out = f"Set base model error: {e}"

            # --- RL COMMANDS ---
            elif low.startswith("train model "):
                try:
                    # train model SYMBOL [PERIOD INTERVAL] [steps N] [from SRC]
                    tokens = cmd.split()
                    symbol = tokens[2].upper() if len(tokens) >= 3 else None
                    period = INDICATOR_PERIOD
                    interval = INDICATOR_INTERVAL
                    steps = 20000
                    init_from = None

                    m_steps = re.search(r"\bstep[s]?\s+(\d+)", cmd, flags=re.I)
                    if m_steps:
                        steps = int(m_steps.group(1))
                    m_from = re.search(r"\bfrom\s+([A-Za-z0-9\-\._]+)", cmd, flags=re.I)
                    if m_from:
                        init_from = m_from.group(1).upper()

                    if len(tokens) >= 4 and re.match(r"^\d+[dwmy]$", tokens[3], flags=re.I):
                        period = tokens[3]
                    if len(tokens) >= 5:
                        interval = tokens[4]

                    if not symbol:
                        out = "Usage: train model SYMBOL [PERIOD INTERVAL] [steps N] [from SRC]"
                    else:
                        out = train_rl_model(symbol, period=period, interval=interval, init_from=init_from, total_timesteps=steps)
                except Exception as e:
                    tb = _exc(e)
                    log_auto(f"[TRAIN ERROR] {tb}")
                    out = f"Training error:\n{tb}"

            elif low.startswith("continue model "):
                try:
                    tokens = cmd.split()
                    symbol = tokens[2].upper() if len(tokens) >= 3 else None
                    more_steps = 20480
                    m_steps = re.search(r"\b(\d+)\b$", cmd.strip())
                    if m_steps:
                        more_steps = int(m_steps.group(1))
                    if not symbol:
                        out = "Usage: continue model SYMBOL [STEPS]"
                    else:
                        out = continue_rl_model(symbol, more_steps=more_steps, period=INDICATOR_PERIOD, interval=INDICATOR_INTERVAL)
                except Exception as e:
                    tb = _exc(e)
                    log_auto(f"[CONTINUE ERROR] {tb}")
                    out = f"Continue error:\n{tb}"

            elif low.startswith("backtest model "):
                try:
                    tokens = cmd.split()
                    symbol = tokens[2].upper() if len(tokens) >= 3 else None
                    period = tokens[3] if len(tokens) >= 4 else INDICATOR_PERIOD
                    interval = tokens[4] if len(tokens) >= 5 else INDICATOR_INTERVAL
                    if not symbol:
                        out = "Usage: backtest model SYMBOL [PERIOD INTERVAL]"
                    else:
                        out = backtest_model_on_history(symbol, period=period, interval=interval)
                except Exception as e:
                    tb = _exc(e)
                    log_auto(f"[BACKTEST ERROR] {tb}")
                    out = f"Backtest error:\n{tb}"

            elif low.startswith("load model "):
                try:
                    symbol = cmd.split("load model ",1)[1].strip().upper()
                    def make_env():
                        df = get_training_data(symbol, period=INDICATOR_PERIOD, interval=INDICATOR_INTERVAL) or pd.DataFrame({"Close":[1.0]})
                        return TradingEnv(df if not df.empty else pd.DataFrame({"Close":[1.0]}))
                    env = DummyVecEnv([make_env])
                    model = PPO.load(rl_model_path(symbol), env=env, device="cpu")
                    RL_MODELS[symbol] = model
                    out = f"✅ RL model for {symbol} loaded and active!"
                except Exception as e:
                    out = f"Error loading model:\n{_exc(e)}"

            elif low == "list models":
                try:
                    lines = []
                    for sym in sorted(RL_MODELS.keys()):
                        meta = MODEL_REGISTRY.get(sym, {})
                        fp = meta.get("fingerprint", "n/a")
                        origin = meta.get("init_from") or meta.get("linked_from")
                        lines.append(f"{sym} fp={fp}" + (f" from={origin}" if origin else ""))
                    out = "Loaded RL models:\n" + ("\n".join(lines) if lines else "(none)")
                except Exception as e:
                    out = f"List models error:\n{_exc(e)}"

            else:
                out = f"Unknown command: {cmd}"

        except Exception as e:
            tb = _exc(e)
            log_auto(f"[CMD ERROR] {tb}")
            out = f"Command error:\n{tb}"

    return render_template_string(HTML, out=out)

# =================
# Dedicated feed endpoint
# =================
@app.route("/feed")
def feed():
    return "<pre>" + ("\n".join(list(_auto_updates)[-100:]) or "No auto updates yet.") + "</pre>"

# =================
# JSON APIs
# =================
@app.route("/api/feed")
def api_feed():
    return jsonify({"feed": list(_auto_updates)[-100:]})

@app.route("/api/health")
def api_health():
    return jsonify({
        "status": get_account_status_text(),
        "watchlist": WATCHLIST,
        "rl_models": list(RL_MODELS.keys()),
        "trading_active": TRADING_ACTIVE,
        "live_scores_active": LIVE_SCORES_ACTIVE,
        "interval": INDICATOR_INTERVAL,
        "period": INDICATOR_PERIOD,
        "sb3_version": _SB3_VER,
        "gym_api": "gymnasium" if _SB3_USES_GYMNASIUM else "classic-gym",
    })

# =================
# Model autoload/autosave + Server boot
# =================
def _autoload_models_on_boot():
    try:
        for fn in os.listdir(os.getcwd()):
            if not fn.startswith("rl_trading_model_") or not fn.endswith(".zip"):
                continue
            sym = fn[len("rl_trading_model_"):-4].upper()
            def make_env():
                df = get_training_data(sym, period=INDICATOR_PERIOD, interval=INDICATOR_INTERVAL) or pd.DataFrame({"Close":[1.0]})
                return TradingEnv(df if not df.empty else pd.DataFrame({"Close":[1.0]}))
            env = DummyVecEnv([make_env])
            try:
                model = PPO.load(fn, env=env, device="cpu")
                RL_MODELS[sym] = model
                MODEL_REGISTRY[sym] = {
                    "path": fn,
                    "fingerprint": _model_fingerprint(model),
                    "loaded_at": datetime.utcnow().isoformat(timespec="seconds"),
                    "init_from": MODEL_REGISTRY.get(sym, {}).get("init_from")
                }
            except Exception as e:
                log_error(f"autoload_model {sym}", e)
        _save_model_registry(MODEL_REGISTRY)
        if RL_MODELS:
            log_auto(f"[AUTOLOAD] Loaded models: {', '.join(sorted(RL_MODELS.keys()))}")
    except Exception as e:
        log_error("autoload_models_on_boot", e)

def _autosave_models_loop():
    while True:
        try:
            if RL_MODELS:
                ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
                for sym, model in list(RL_MODELS.items()):
                    try:
                        base = rl_model_path(sym)
                        model.save(base)
                        model.save(f"{base}.{ts}.zip")
                        MODEL_REGISTRY[sym] = {
                            "path": base + ".zip",
                            "fingerprint": _model_fingerprint(model),
                            "checkpoint_utc": ts,
                            "init_from": MODEL_REGISTRY.get(sym, {}).get("init_from")
                        }
                    except Exception as e:
                        log_error(f"autosave {sym}", e)
                _save_model_registry(MODEL_REGISTRY)
                log_auto(f"[AUTOSAVE] Snapshotted {len(RL_MODELS)} model(s) @ {ts}Z")
        except Exception as e:
            log_error("autosave_models_loop", e)
        time.sleep(MODEL_AUTOSAVE_MIN * 60)

def start_background_threads():
    try:
        _autoload_models_on_boot()
        threading.Thread(target=trading_loop,     daemon=True).start()
        threading.Thread(target=live_scores_loop, daemon=True).start()
        threading.Thread(target=eod_flatten_loop, daemon=True).start()
        threading.Thread(target=_autosave_models_loop, daemon=True).start()
        _auto_updates.append("🚀 agent booted; use 'help', 'start live scores', and 'start trading'")
    except Exception as e:
        log_error("start_background_threads", e)

start_background_threads()

if __name__ == "__main__":
    print("🚀 Trading Agent running on http://127.0.0.1:7860")
    app.run(host="0.0.0.0", port=7860, debug=False)
