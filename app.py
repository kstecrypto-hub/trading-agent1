# ================================
# Trading Agent — Fortified Build
# ================================

# ---- Stdlib
import os, time, csv, math, json, uuid, threading
from typing import Optional, List, Dict
from collections import deque, defaultdict
from datetime import datetime, timezone, timedelta

# ---- Third-party
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, render_template_string

# Alpaca (REST)
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

# Charts (headless-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# ---- Flask
app = Flask(__name__)
APP_TITLE = "Trading Agent — Fortified"

# Escaped Jinja inside f-string
HTML = f"""
<!doctype html>
<title>{APP_TITLE}</title>
<style>
 body{{font-family:system-ui,Arial;margin:24px;max-width:1024px}}
 input[type=text]{{width:70%;padding:8px}}
 button, a.btn{{display:inline-block;padding:8px 12px;margin-left:6px;border:1px solid #334155;border-radius:6px;text-decoration:none;color:#0f172a;background:#e2e8f0}}
 button:hover, a.btn:hover{{background:#cbd5e1}}
 pre{{background:#0f172a;color:#e2e8f0;padding:12px;border-radius:8px;white-space:pre-wrap}}
 .hint{{color:#475569;margin-top:8px}}
 .toolbar{{margin:10px 0}}
</style>

<h2>{APP_TITLE}</h2>

<!-- Command input -->
<form method="post" style="margin-bottom:8px">
  <input name="cmd" placeholder="type a command e.g. 'help'" />
  <button type="submit">Run</button>
  <a class="btn" href="/feed">Open Feed Page</a>
</form>

<!-- Quick actions -->
<div class="toolbar">
  <form method="post" style="display:inline"><input type="hidden" name="cmd" value="status" /><button type="submit">Status</button></form>
  <form method="post" style="display:inline"><input type="hidden" name="cmd" value="health" /><button type="submit">Health</button></form>
  <form method="post" style="display:inline"><input type="hidden" name="cmd" value="start live scores" /><button type="submit">Start Live Scores</button></form>
  <form method="post" style="display:inline"><input type="hidden" name="cmd" value="stop live scores" /><button type="submit">Stop Live Scores</button></form>
  <form method="post" style="display:inline"><input type="hidden" name="cmd" value="auto feed" /><button type="submit">Auto Feed</button></form>
</div>

<p class="hint">Try: <code>help</code>, <code>live scores</code>, <code>start trading</code>, <code>day mode on</code></p>

{{% if out %}}<pre>{{{{ out }}}}</pre>{{% endif %}}
"""

# =================
# Global Config (baseline)
# =================
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
# Utilities & logging
# =================
def norm_cmd(s: str) -> str:
    return " ".join((s or "").lower().split())

def log_auto(msg: str) -> None:
    try:
        _auto_updates.append(msg)
    except Exception:
        pass

def log_error(context: str, err: Exception) -> None:
    try:
        _auto_updates.append(f"[ERROR] {context}: {err}")
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
    """yfinance symbol, e.g., BTCUSD -> BTC-USD."""
    s = (sym or "").upper().replace("/", "-")
    if s.endswith("USD") and "-" not in s:
        s = s[:-3] + "-" + s[-3:]
    return s

def broker_symbol(sym: str) -> str:
    """Broker (Alpaca) symbol, e.g., BTC-USD -> BTCUSD."""
    s = (sym or "").upper().replace("/", "")
    s = s.replace("-", "")
    return s

def is_crypto_symbol_yf(sym: str) -> bool:
    return is_crypto_symbol(sym)

def _utc_now():
    return datetime.now(timezone.utc)

def intraday_window_open() -> bool:
    """Equities gate: open only within configured UTC window on weekdays."""
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
    """Resets trade ceilings once per UTC day."""
    global _trades_today_total, _trades_today_by_sym, _last_reset_yyyy_mm_dd
    today = _utc_now().strftime("%Y-%m-%d")
    if _last_reset_yyyy_mm_dd != today:
        _trades_today_total = 0
        _trades_today_by_sym = defaultdict(int)
        _last_reset_yyyy_mm_dd = today

def can_trade_symbol(sym: str) -> bool:
    """Policy gate: window for equities, daily ceilings for all."""
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

# =================
# Alpaca client & order utilities
# =================
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
    """
    Normalize yfinance Close column to a clean 1-D float Series.
    Handles cases where Close is a DataFrame (multiindex) or contains arrays.
    """
    try:
        s = df["Close"]
        # If yfinance returns multi-ticker or multiindex columns, pick the first Close column
        if isinstance(s, pd.DataFrame):
            # first non-empty column
            for col in s.columns:
                col_s = pd.to_numeric(s[col], errors="coerce")
                col_s = col_s.dropna().astype(float)
                if not col_s.empty:
                    col_s.name = "Close"
                    return col_s
            return pd.Series(dtype=float)

        # Single Series path
        s = pd.to_numeric(s, errors="coerce").dropna().astype(float)
        s.name = "Close"
        return s
    except Exception as e:
        log_error("to_close_series", e)
        return pd.Series(dtype=float)

def ema(series: pd.Series, span: int) -> pd.Series:
    try:
        return series.ewm(span=span, adjust=False).mean()
    except Exception as e:
        log_error(f"ema(span={span})", e)
        return pd.Series(index=series.index, dtype=float)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    try:
        s = pd.Series(series).astype(float).dropna()
        if s.empty:
            return pd.Series(index=getattr(series, "index", None), dtype=float).fillna(50.0)
        d = s.diff()
        gains = d.clip(lower=0.0)
        losses = (-d).clip(lower=0.0)
        roll_up = gains.rolling(period, min_periods=period).mean()
        roll_down = losses.rolling(period, min_periods=period).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val.reindex(series.index).fillna(50.0)
    except Exception as e:
        log_error("rsi", e)
        return pd.Series(index=getattr(series, "index", None), dtype=float).fillna(50.0)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    try:
        ema_fast = ema(series, fast)
        ema_slow = ema(series, slow)
        line = ema_fast - ema_slow
        signal_line = ema(line, signal)
        hist = line - signal_line
        return line, signal_line, hist
    except Exception as e:
        log_error("macd", e)
        empty = pd.Series(index=series.index, dtype=float)
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
# Price & Positions
# =================
def _to_scalar(val) -> Optional[float]:
    try:
        # Prefer .item() for 0-d numpy/pandas scalars
        if hasattr(val, "item"):
            return float(val.item())
        # Fall back to last element of a 1-D view
        arr = np.asarray(val).reshape(-1)
        if arr.size >= 1:
            return float(arr[-1])
    except Exception:
        pass
    return None

def get_price(sym: str) -> Optional[float]:
    """Robust last-close fetch with safe scalarization and fallback."""
    # Primary: current interval/period
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

    # Fallback: daily bars
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

def get_position_side_qty(sym: str) -> tuple[str, float, float]:
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
            "", "", "", "", "",  # exit fields TBD
            json.dumps(JOURNAL_FRAMES), "", snap_dir
        ])
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
                idx = i; break
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
    """Return broker-valid quantity or None if cannot size."""
    try:
        px = float(price or 0.0)
        if px <= 0:
            return None
        budget = float(MAX_TRADE_USD)
        if budget <= 0:
            return None
        if is_crypto_symbol(sym):
            # fractional allowed; 4dp granularity is safe for BTC on Alpaca
            q = round(max(0.0001, budget / px), 4)
            return q
        # equities: integer shares
        q = int(math.floor(budget / px))
        return q if q >= 1 else None
    except Exception as e:
        log_error("calc_order_qty", e)
        return None

# =================
# Orders (side-aware, crypto-aware, journaling)
# =================
def place_order(symbol: str, side: str, qty: float) -> str:
    """
    Market order submitter with:
      - Long/short entry toggles
      - Crypto shorting policy
      - Journaling on open/close
      - Reversal-friendly (handled by caller)
    """
    try:
        client = get_trading_client()
        sd = (side or "").strip().lower()
        if sd not in ("buy", "sell"):
            return f"Invalid side: {side}"

        symN = norm_symbol(symbol)
        bro  = broker_symbol(symN)
        pos_side, pos_qty, _ = get_position_side_qty(symN)

        # Entry toggles
        if sd == "buy" and pos_side == "flat" and not LONGING_ENABLED:
            return "Longing disabled. Use 'long on' to enable."
        if sd == "sell" and pos_side == "flat" and not SHORTING_ENABLED:
            return "Shorting disabled. Use 'short on' to enable."

        # Crypto shorting policy (spot brokers typically disallow short)
        if sd == "sell" and pos_side == "flat" and is_crypto_symbol(symN) and not CRYPTO_SHORTING_ENABLED:
            return "Crypto shorting disabled. Use 'crypto short on' only if broker supports."

        # Normalize qty for broker
        broker_qty = qty
        if not is_crypto_symbol(symN):
            broker_qty = int(qty) if float(qty).is_integer() else int(qty)

        req = MarketOrderRequest(
            symbol=bro,
            qty=str(broker_qty),
            side=(OrderSide.BUY if sd == "buy" else OrderSide.SELL),
            time_in_force=TimeInForce.DAY
        )
        order = client.submit_order(order_data=req)

        now = time.time()
        px  = get_price(symN) or 0.0

        # Journal hooks
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
# Risk: forced exits (SL/TP)
# =================
def evaluate_forced_exit(sym: str) -> bool:
    """Execute hard exits when SL/TP thresholds hit; return True if exited."""
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
    else:  # short
        if px >= avg * (1 + STOP_LOSS_PCT) or px <= avg * (1 - TAKE_PROFIT_PCT):
            res = place_order(sym, "buy", qty)
            log_auto(f"[FORCED EXIT] {broker_symbol(sym)} short→flat @ {px:.4f} | {res}")
            return True
    return False

# =================
# Live-scores utilities & loop
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
# Trading loop (day-trade; long/short; reversals; cooldown)
# =================
def trading_loop():
    global DEBUG_TICKS
    while True:
        try:
            reset_intraday_counters_if_new_day()

            if TRADING_ACTIVE:
                for sym in WATCHLIST:
                    # ceilings/window
                    if not can_trade_symbol(sym):
                        if DEBUG_TICKS > 0: dbg_reason(f"{sym} skip: can_trade_symbol=False")
                        continue

                    # risk exits first
                    if evaluate_forced_exit(sym):
                        if DEBUG_TICKS > 0: dbg_reason(f"{sym}: forced exit")
                        continue

                    # equities obey window; crypto 24/7
                    if (not is_crypto_symbol(sym)) and (not intraday_window_open()):
                        if DEBUG_TICKS > 0: dbg_reason(f"{sym} skip: outside day window")
                        continue

                    # data & score
                    df = get_history(sym)
                    if df is None or df.empty:
                        if DEBUG_TICKS > 0: dbg_reason(f"{sym} skip: history empty")
                        continue

                    tscore = int(trend_score(df))
                    side_signal = trend_decision(tscore)

                    # toggles for new entries
                    cur_side, cur_qty, _ = get_position_side_qty(sym)
                    if side_signal == "BUY" and cur_side == "flat" and not LONGING_ENABLED:
                        if DEBUG_TICKS > 0: dbg_reason(f"{sym} HOLD: long disabled (score={tscore})")
                        side_signal = "HOLD"
                    if side_signal == "SELL" and cur_side == "flat" and not SHORTING_ENABLED:
                        if DEBUG_TICKS > 0: dbg_reason(f"{sym} HOLD: short disabled (score={tscore})")
                        side_signal = "HOLD"
                    if side_signal == "SELL" and cur_side == "flat" and is_crypto_symbol(sym) and not CRYPTO_SHORTING_ENABLED:
                        if DEBUG_TICKS > 0: dbg_reason(f"{sym} HOLD: crypto short disabled")
                        side_signal = "HOLD"

                    # persistence debounce
                    buf = _sigbuf_trend[sym]
                    buf.append(side_signal)
                    persistent = (len(buf) == TREND_PERSIST and all(x == side_signal for x in buf))
                    if not persistent or side_signal == "HOLD":
                        if DEBUG_TICKS > 0: dbg_reason(f"{sym} skip: persist/HOLD (score={tscore}, buf={list(buf)})")
                        continue

                    # size & cooldown
                    px = get_price(sym)
                    qty = calc_order_qty(sym, px)
                    if qty is None:
                        if DEBUG_TICKS > 0: dbg_reason(f"{sym} skip: cannot size qty (price={px})")
                        continue

                    now = time.time()
                    last_side = _last_side_by_sym.get(sym)
                    last_t    = _last_trade_time.get(sym, 0.0)
                    if last_side == side_signal and (now - last_t) < TRADE_COOLDOWN_SEC:
                        if DEBUG_TICKS > 0: dbg_reason(f"{sym} skip: cooldown {int(now-last_t)}<{TRADE_COOLDOWN_SEC}s")
                        continue

                    # execute + reversals
                    if side_signal == "BUY":
                        if cur_side == "short":
                            place_order(sym, "buy", cur_qty)    # cover
                            res = place_order(sym, "buy", qty)  # open long
                            log_auto(f"[REVERSAL] {broker_symbol(sym)} short→long qty={qty} TrendScore={tscore}/100 | {res}")
                        elif cur_side == "flat":
                            res = place_order(sym, "buy", qty)
                            log_auto(f"[AUTO BUY] {broker_symbol(sym)} qty={qty} px={px:.4f} TrendScore={tscore}/100 | {res}")
                        _last_side_by_sym[sym] = "BUY"; _last_trade_time[sym] = now

                    elif side_signal == "SELL":
                        if cur_side == "long":
                            place_order(sym, "sell", cur_qty)   # close long
                            res = place_order(sym, "sell", qty) # open short
                            log_auto(f"[REVERSAL] {broker_symbol(sym)} long→short qty={qty} TrendScore={tscore}/100 | {res}")
                        elif cur_side == "flat":
                            res = place_order(sym, "sell", qty)
                            log_auto(f"[AUTO SHORT] {broker_symbol(sym)} qty={qty} px={px:.4f} TrendScore={tscore}/100 | {res}")
                        _last_side_by_sym[sym] = "SELL"; _last_trade_time[sym] = now

                    if DEBUG_TICKS > 0:
                        DEBUG_TICKS -= 1

        except Exception as e:
            log_error("trading_loop", e)

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
    """Force flat five minutes before configured intraday window end (equities)."""
    while True:
        try:
            if EOD_FLATTEN and DAY_TRADE_MODE:
                now = _utc_now()
                h2, m2 = DAY_WINDOW_END_UTC
                cutoff = now.replace(hour=h2, minute=m2, second=0, microsecond=0) - timedelta(minutes=5)
                if now >= cutoff:
                    msg = flatten_all()
                    log_auto(f"[EOD FLATTEN] {msg}")
                    time.sleep(300)  # avoid repeats
        except Exception as e:
            log_error("eod_flatten_loop", e)
        time.sleep(15)
# =================
# Watchlist & Health
# =================
def add_to_watchlist(symbols: List[str]) -> List[str]:
    """Add symbols without removing existing; returns updated WATCHLIST."""
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
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Health error: {e}"

# =================
# Flask UI / Command Parser
# =================
@app.route("/", methods=["GET","POST"])
def index():
    # --- declare globals once, up-front ---
    global TRADING_ACTIVE, LIVE_SCORES_ACTIVE
    global LONGING_ENABLED, SHORTING_ENABLED, CRYPTO_SHORTING_ENABLED
    global DAY_TRADE_MODE, DAY_WINDOW_START_UTC, DAY_WINDOW_END_UTC
    global STOP_LOSS_PCT, TAKE_PROFIT_PCT, MAX_HOLD_MIN
    global TREND_BUY_TH, TREND_SELL_TH, TREND_PERSIST, TRADE_COOLDOWN_SEC
    global MAX_TRADE_USD, INDICATOR_INTERVAL, INDICATOR_PERIOD
    global DEBUG_TICKS, _sigbuf_trend  # we reassign _sigbuf_trend on persist change

    out = ""
    if request.method == "POST":
        raw = request.form.get("cmd", "")
        low = norm_cmd(raw)   # normalized for matching
        cmd = raw.strip()     # original for params/display

        try:
            # --- HELP ---
            if low == "help":
                out = """Available commands:
 help                  → this menu
 status                → account + portfolio status
 health                → internal config + state

 start trading         → activate auto trading
 stop trading          → deactivate auto trading
 start live scores     → turn on auto score feed
 stop live scores      → turn off auto score feed
 auto feed             → show last auto log messages
 live scores           → snapshot scores for WATCHLIST

 long on/off           → enable or disable long entries
 short on/off          → enable or disable short entries
 crypto short on/off   → enable/disable crypto shorts (if broker supports)

 day mode on/off       → enable or disable day-trading window
 set day window: HH:MM-HH:MM UTC

 set sl:X              → set stop-loss % (e.g., set sl:1.2)
 set tp:X              → set take-profit % (e.g., set tp:1.8)
 set hold:X            → set max-hold minutes (placeholder)

 set buy th:X          → TrendScore buy threshold (int)
 set sell th:X         → TrendScore sell threshold (int)
 set persist:X         → consecutive identical signals required
 set cooldown:X        → cooldown seconds between same-side trades
 set max usd:X         → per-trade budget in USD

 set interval: i p     → yfinance interval/period (e.g., set interval:5m 60d)
 add SYMBOLS           → add symbols (comma/space separated)
 add btc               → quick add BTC-USD

 orders                → list open orders
 cancel all            → cancel all open orders
 cancel ID             → cancel order by ID
 flatten all           → close all positions
 close SYMBOL          → close a specific symbol market

 buy SYMBOL [qty]      → manual buy (qty optional; auto-sized if omitted)
 sell SYMBOL [qty]     → manual sell (qty optional; auto-sized if omitted)

 debug ticks:N         → log per-tick decisions N times
"""

            # --- STATUS / HEALTH ---
            elif low == "status":
                out = get_account_status_text() + "\n" + portfolio_status()
            elif low == "health":
                out = health_snapshot()

            # --- TRADING TOGGLES ---
            elif low == "start trading":
                TRADING_ACTIVE = True; out = "Trading ENABLED"
            elif low == "stop trading":
                TRADING_ACTIVE = False; out = "Trading DISABLED"

            # --- LIVE SCORES / FEED ---
            elif low == "start live scores":
                LIVE_SCORES_ACTIVE = True
                # immediate snapshot so /feed isn’t empty
                try:
                    lines = [format_live_scores_line(s) for s in WATCHLIST]
                    if lines:
                        log_auto("[AUTO LIVE SCORES]\n" + "\n".join(lines))
                except Exception as e:
                    log_error("start live scores snapshot", e)
                out = "Live scores ENABLED"
            elif low == "stop live scores":
                LIVE_SCORES_ACTIVE = False; out = "Live scores DISABLED"
            elif low in ("auto feed", "autofeed", "auto_feed", "feed"):
                out = "\n".join(list(_auto_updates)[-100:]) or "No auto updates yet."
            elif low == "live scores":
                out = "\n".join([format_live_scores_line(s) for s in WATCHLIST])

            # --- LONG/SHORT & CRYPTO SHORT TOGGLES ---
            elif low == "long on":
                LONGING_ENABLED = True; out = "📈 Longing ENABLED"
            elif low == "long off":
                LONGING_ENABLED = False; out = "📈 Longing DISABLED"
            elif low == "short on":
                SHORTING_ENABLED = True; out = "📉 Shorting ENABLED"
            elif low == "short off":
                SHORTING_ENABLED = False; out = "📉 Shorting DISABLED"
            elif low == "crypto short on":
                CRYPTO_SHORTING_ENABLED = True; out = "₿ Crypto shorting ENABLED (check broker support)"
            elif low == "crypto short off":
                CRYPTO_SHORTING_ENABLED = False; out = "₿ Crypto shorting DISABLED"

            # --- DAY MODE / WINDOW ---
            elif low == "day mode on":
                DAY_TRADE_MODE = True; out = "🕒 Day-trading mode ENABLED"
            elif low == "day mode off":
                DAY_TRADE_MODE = False; out = "🕒 Day-trading mode DISABLED"
            elif low.startswith("set day window:"):
                try:
                    payload = cmd.split(":",1)[1].strip().split()[0]
                    start, end = payload.split("-")
                    sh, sm = [int(x) for x in start.split(":")]
                    eh, em = [int(x) for x in end.split(":")]
                    DAY_WINDOW_START_UTC = (sh, sm); DAY_WINDOW_END_UTC = (eh, em)
                    out = f"✅ Day window set to {sh:02d}:{sm:02d}-{eh:02d}:{em:02d} UTC"
                except Exception:
                    out = "Usage: set day window: HH:MM-HH:MM UTC"

            # --- RISK & RUNTIME KNOBS ---
            elif low.startswith("set sl:"):
                try:
                    STOP_LOSS_PCT = float(cmd.split(":",1)[1].strip())/100.0
                    out = f"✅ SL set to {STOP_LOSS_PCT*100:.2f}%"
                except: out = "Usage: set sl: <percent>"
            elif low.startswith("set tp:"):
                try:
                    TAKE_PROFIT_PCT = float(cmd.split(":",1)[1].strip())/100.0
                    out = f"✅ TP set to {TAKE_PROFIT_PCT*100:.2f}%"
                except: out = "Usage: set tp: <percent>"
            elif low.startswith("set hold:"):
                try:
                    MAX_HOLD_MIN = int(cmd.split(":",1)[1].strip())
                    out = f"✅ Max hold set to {MAX_HOLD_MIN} min"
                except: out = "Usage: set hold: <minutes>"

            elif low.startswith("set buy th:"):
                try:
                    TREND_BUY_TH = int(cmd.split(":",1)[1].strip())
                    out = f"✅ Buy threshold = {TREND_BUY_TH}"
                except: out = "Usage: set buy th: <int>"
            elif low.startswith("set sell th:"):
                try:
                    TREND_SELL_TH = int(cmd.split(":",1)[1].strip())
                    out = f"✅ Sell threshold = {TREND_SELL_TH}"
                except: out = "Usage: set sell th: <int>"
            elif low.startswith("set persist:"):
                try:
                    TREND_PERSIST = int(cmd.split(":",1)[1].strip())
                    # rebuild signal buffers with new maxlen
                    old_keys = list(_sigbuf_trend.keys())
                    _sigbuf_trend = defaultdict(lambda: deque(maxlen=TREND_PERSIST))
                    for k in old_keys:
                        _sigbuf_trend[k]  # touch to create
                    out = f"✅ Signal persistence = {TREND_PERSIST}"
                except: out = "Usage: set persist: <bars>"
            elif low.startswith("set cooldown:"):
                try:
                    TRADE_COOLDOWN_SEC = int(cmd.split(":",1)[1].strip())
                    out = f"✅ Cooldown = {TRADE_COOLDOWN_SEC}s"
                except: out = "Usage: set cooldown: <seconds>"
            elif low.startswith("set max usd:"):
                try:
                    MAX_TRADE_USD = float(cmd.split(":",1)[1].strip())
                    out = f"✅ MAX_TRADE_USD = ${MAX_TRADE_USD:.2f}"
                except: out = "Usage: set max usd: <dollars>"

            # --- INTERVAL / WATCHLIST ---
            elif low.startswith("set interval:"):
                try:
                    _, payload = cmd.split(":",1)
                    parts = payload.strip().split()
                    INDICATOR_INTERVAL = parts[0]; INDICATOR_PERIOD = parts[1]
                    out = f"✅ Interval set to {INDICATOR_INTERVAL} {INDICATOR_PERIOD}"
                except Exception:
                    out = "Usage: set interval: <interval> <period>"
            elif low.startswith("add "):
                payload = cmd.split(" ",1)[1]
                syms = [s.strip() for s in payload.replace(","," ").split()]
                add_to_watchlist(syms)
                out = f"✅ Added {syms}. WATCHLIST={WATCHLIST}"
            elif low == "add btc":
                add_to_watchlist(["BTC-USD"])
                out = f"✅ Added BTC-USD. WATCHLIST={WATCHLIST}"

            # --- ORDERS / POSITIONS ---
            elif low == "orders":
                ords = list_open_orders()
                out = "\n".join([f"{o.id} {o.symbol} {o.qty} {o.side} status={o.status}" for o in ords]) or "No open orders"
            elif low == "cancel all":
                out = cancel_all_orders()
            elif low.startswith("cancel "):
                oid = cmd.split(" ",1)[1].strip()
                out = cancel_order(oid)
            elif low == "flatten all":
                out = flatten_all()
            elif low.startswith("close "):
                sym = cmd.split(" ",1)[1].strip()
                out = close_position_market(sym)

            # --- MANUAL ORDERS (for E2E sanity) ---
            elif low.startswith("buy "):
                try:
                    payload = cmd.split(" ",1)[1].strip()
                    parts = payload.split()
                    sym = parts[0]
                    q   = float(parts[1]) if len(parts) > 1 else None
                    px  = get_price(sym)
                    if q is None:
                        q = calc_order_qty(sym, px)
                    out = place_order(sym, "buy", q)
                except Exception as e:
                    out = f"Buy error: {e}"
            elif low.startswith("sell "):
                try:
                    payload = cmd.split(" ",1)[1].strip()
                    parts = payload.split()
                    sym = parts[0]
                    q   = float(parts[1]) if len(parts) > 1 else None
                    px  = get_price(sym)
                    if q is None:
                        q = calc_order_qty(sym, px)
                    out = place_order(sym, "sell", q)
                except Exception as e:
                    out = f"Sell error: {e}"

            # --- DEBUG ---
            elif low.startswith("debug ticks:"):
                try:
                    DEBUG_TICKS = int(cmd.split(":",1)[1].strip())
                    out = f"✅ Debug logging for next {DEBUG_TICKS} loop ticks"
                except:
                    out = "Usage: debug ticks: <count>"

            else:
                out = f"Unknown command: {cmd}"
        except Exception as e:
            out = f"Command error: {e}"

    return render_template_string(HTML, out=out)


# Dedicated feed endpoint (bypass parser)
@app.route("/feed")
def feed():
    return "<pre>" + ("\n".join(list(_auto_updates)[-100:]) or "No auto updates yet.") + "</pre>"

# =================
# Thread startup + Server boot
# =================
def start_background_threads():
    try:
        threading.Thread(target=trading_loop,     daemon=True).start()
        threading.Thread(target=live_scores_loop, daemon=True).start()
        threading.Thread(target=eod_flatten_loop, daemon=True).start()
        _auto_updates.append("🚀 agent booted; use 'start live scores' and 'start trading'")
    except Exception as e:
        log_error("start_background_threads", e)

start_background_threads()

if __name__ == "__main__":
    print("🚀 Trading Agent running on http://127.0.0.1:7860")
    app.run(host="0.0.0.0", port=7860, debug=False)
