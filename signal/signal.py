#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 SIGNAL BOT (GOLD ONLY) — แจ้งเตือนเข้า/ออก (นายกดเทรดเอง) รันบน Mac ได้
=====================================================================
ดึงกราฟ 1h จาก TwelveData -> logic ที่ผ่าน walk-forward -> ส่ง Telegram
บันทึกทุกไม้เป็น R-multiple (ไม่เดากำไรเป็นเงิน = ไม่หลอกตัวเองแบบตัวเก่า)

⚠️ สิ่งที่บอทนี้ 'เป็น' และ 'ไม่เป็น':
  เป็น   : ตัวจับจังหวะ บอกว่า 'ควรเข้า/ออกตรงไหน' ตาม logic
  ไม่เป็น: ตัววัดกำไรจริง — ราคาอ้างอิง TwelveData ยังไม่รวมสเปรด/คอม/
           slippage ผลจริงดูจากบัญชี Exness ที่นายกดเทรดเอง
  R-multiple ในบอท = 'จังหวะดีแค่ไหน' ไม่ใช่ 'ได้เงินเท่าไหร่'

ตั้งค่า (.env ในโฟลเดอร์เดียวกัน):
  TWELVEDATA_API_KEY=xxxx
  TELEGRAM_BOT_TOKEN=xxxx          # จาก @BotFather (ควร /revoke อันเก่าถ้าเคยหลุด)
  TELEGRAM_CHAT_ID=                # เว้นว่างได้ บอทหาให้เอง (ทักบอท 1 ข้อความก่อนรัน)

รัน:
  1) pip install requests pandas numpy python-dotenv
  2) เปิด Telegram ทักบอทตัวเอง พิมพ์อะไรก็ได้ 1 ข้อความ
  3) python signal_bot.py
=====================================================================
"""
import os
import json
import time
from datetime import datetime
import requests
import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

TD_KEY = os.getenv("TWELVEDATA_API_KEY", "")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

# ---------------- CONFIG ----------------
# source: "td" = TwelveData, "yf" = yfinance (ฟรี ไม่ต้อง key)
# โฟกัสทองคำตัวเดียว
SYMBOLS = [
    {"source": "td", "ticker": "XAU/USD", "label": "ทองคำ", "validated": True},
]

INTERVAL   = "1h"
OUTPUTSIZE = 600          # ต้องพอสำหรับ ema200 + atr_ratio(100) + warmup
BAR_HOURS  = 1            # ความยาวแท่ง (ชม.) ใช้ตัดแท่งที่ยังไม่ปิดทิ้ง
# ---- v4: พรีเซ็ตความถี่การเทรด (เลือกอันเดียว) ----
#   STANDARD = เข้มสุด  ~2.9 ไม้/เดือน  (edge ต่อไม้สูงสุด)
#   ACTIVE   = กลาง     ~3.9 ไม้/เดือน
#   MAX      = เยอะสุด  ~5.3 ไม้/เดือน  (edge ต่อไม้บางลง ไวต่อสเปรด)
PRESET = "STANDARD"   # เลือกอันกำไรรวมสูงสุดใน backtest (เปลี่ยนเป็น ACTIVE / MAX ถ้าอยากเทรดถี่ขึ้น)
PRESETS = {
    "STANDARD": dict(ADX_MIN=20, Z_LIMIT=2.5, ATR_SL_MULT=1.0, RR=2.0,  ER_MIN=0.40, ATR_RATIO_MIN=0.8),  # ปรับจาก SL3/RR3 -> SL1.0/RR2.0
    "ACTIVE":   dict(ADX_MIN=15, Z_LIMIT=3.5, ATR_SL_MULT=3.0, RR=1.25, ER_MIN=0.45, ATR_RATIO_MIN=0.8),
    "MAX":      dict(ADX_MIN=15, Z_LIMIT=3.5, ATR_SL_MULT=3.0, RR=1.5,  ER_MIN=0.40, ATR_RATIO_MIN=0.8),
}
VOL_LIM    = 2.0
ADX_RISING    = False     # เปิด = เข้มขึ้น เทรดน้อยลง
ADX_SLOPE_N   = 5
ER_N          = 30        # Kaufman Efficiency Ratio ย้อนหลัง N แท่ง (ราคาไปทางเดียวได้แค่ไหน)
ATR_RATIO_N   = 100       # ATR ปัจจุบัน / ค่าเฉลี่ย 100 แท่ง (ตลาดนิ่งมาก -> ไม่เข้า)
# ER_MIN = ตัวหลักที่กัน sideway: < ER_MIN = ราคาแกว่งไปมา -> ไม่เข้า (ห้ามลดต่ำกว่า 0.40 — backtest พบ drawdown พุ่ง)


def apply_preset(name):
    """ตั้งค่า ADX_MIN/Z_LIMIT/ATR_SL_MULT/RR/ER_MIN/ATR_RATIO_MIN ตามพรีเซ็ต"""
    global PRESET
    PRESET = name
    globals().update(PRESETS[name])


apply_preset(PRESET)

# ---- v2: เบรกเกอร์ป้องกันพอร์ตเล็ก ----
COOL_AFTER_LOSSES = 0     # 0 = ปิดการพักเทรด (backtest: ตัดไม้ที่ชนะทิ้งไปบางส่วน กำไรลด) ; ตั้ง 3 = พักหลังแพ้ติด 3 ไม้
COOL_BARS         = 24    # พัก N แท่ง (24 ชม.)
DD_HALT_PCT       = 50.0  # ทุนตกจาก peak เกินกี่ % -> หยุดบอท (ต้องตั้ง halted เป็น false ใน state file เอง)
POLL_SEC   = 600          # เช็กทุก 10 นาที (แท่ง 4h ปิดค่อยมีสัญญาณ)
REQ_GAP    = 8            # หน่วงระหว่าง request (free tier ~8/นาที)
STATE_FILE = "signal_state_gold.json"
LOG_FILE   = "signals_log_gold.csv"

# ---- พอร์ต (ตั้งให้ตรงกับ Exness demo ของนาย) ----
PORTFOLIO_SIZE = 400.0    # ทุนเริ่มต้น (USD)
RISK_PCT       = 2.0      # เสี่ยงต่อไม้ (%) เป้าหมาย
MAX_RISK_PCT   = 100.0    # ปิดเพดานแล้ว (เดิม 15) = ส่งสัญญาณทุกครั้งที่ผ่านครบ ข้ามเฉพาะกรณี SL ใหญ่กว่าทุนทั้งพอร์ต
SPREAD_USD     = 0.30     # สเปรดโดยประมาณ USD/oz ต่อไม้ (ใช้หักในยอดประมาณ)
# contract size ต่อ 1.0 lot (ค่ามาตรฐาน Exness — เช็กใน Specification ของนายอีกที)
CONTRACT_SIZE = {
    "XAU/USD": 100,    # ทอง 100 oz/lot  (0.01 lot = 1 oz)
}
MIN_LOT = 0.01            # lot ต่ำสุดของโบรกเกอร์ (เช็กใน Specification)
LOT_STEP = 0.01


# ---------------- INDICATORS ----------------
def rma(s, n): return s.ewm(alpha=1.0 / n, adjust=False).mean()


def add_ind(df):
    h, l, c = df['high'], df['low'], df['close']
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df['atr'] = rma(tr, 14)
    up, dn = h.diff(), -l.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = rma(tr, 14).replace(0, np.nan)
    df['dmp'] = 100 * rma(plus, 14) / a
    df['dmn'] = 100 * rma(minus, 14) / a
    ds = (df['dmp'] + df['dmn']).replace(0, np.nan)
    df['adx'] = rma(100 * (df['dmp'] - df['dmn']).abs() / ds, 14)
    df['ema200'] = c.ewm(span=200, adjust=False).mean()
    m, sd = c.rolling(20).mean(), c.rolling(20).std()
    df['z'] = (c - m) / sd.replace(0, np.nan)
    df['vol_spike'] = 1.0
    df['adx_slope'] = df['adx'] - df['adx'].shift(ADX_SLOPE_N)
    df['er'] = (c - c.shift(ER_N)).abs() / c.diff().abs().rolling(ER_N).sum().replace(0, np.nan)
    df['atr_ratio'] = df['atr'] / df['atr'].rolling(ATR_RATIO_N).mean()
    return df


def structure_last(df, w=5):
    high, low = df['high'].values, df['low'].values
    n = len(df); sh, sl = [], []
    for t in range(w, n - w):
        if high[t] == high[t - w:t + w + 1].max(): sh.append(high[t])
        if low[t] == low[t - w:t + w + 1].min(): sl.append(low[t])
    if len(sh) >= 2 and len(sl) >= 2:
        if sl[-1] > sl[-2] and sh[-1] > sh[-2]: return 'UPTREND'
        if sl[-1] < sl[-2] and sh[-1] < sh[-2]: return 'DOWNTREND'
    return 'SIDEWAY'


def signal(df):
    r = df.iloc[-1]
    if any(pd.isna(r[k]) for k in ['atr', 'adx', 'dmp', 'dmn', 'z', 'ema200', 'adx_slope', 'er', 'atr_ratio']):
        return None
    bias = 'BUY' if r['close'] > r['ema200'] else 'SELL'
    di = 'BUY' if r['dmp'] > r['dmn'] else 'SELL'
    st = structure_last(df)
    st_ok = (bias == 'BUY' and st in ['UPTREND', 'SIDEWAY']) or \
            (bias == 'SELL' and st in ['DOWNTREND', 'SIDEWAY'])
    ok = (r['adx'] > ADX_MIN and di == bias and r['vol_spike'] < VOL_LIM and
          abs(r['z']) < Z_LIMIT and st_ok)
    # ---- v2 กรอง sideway ----
    if ADX_RISING and not r['adx_slope'] > 0:
        ok = False
    if r['er'] < ER_MIN or r['atr_ratio'] < ATR_RATIO_MIN:
        ok = False
    return bias if ok else None


LAST_PRICE = {}   # ticker -> ราคาปิดล่าสุด (ใช้คำนวณกำไร/ขาดทุนลอยในหน้ารายงาน)
CLEAR_SCREEN = True   # True = เคลียร์จอก่อนเริ่มทุกรอบ ให้หน้ารายงานอยู่ในจอเดียวพอดีสำหรับแคป (False = เลื่อนต่อเนื่อง)
LAST_CHECK = {}   # ticker -> (bar_time, bias, rows)  เก็บผลเช็กลิสต์ล่าสุดไว้แสดงตอนรอสัญญาณ


def checklist(df):
    """เงื่อนไขเข้าไม้ทีละข้อ ณ แท่งล่าสุด -> (bias, [(ชื่อ, ผ่านไหม, ค่าปัจจุบัน, ข้อความขาด)])
    ใช้เงื่อนไขชุดเดียวกับ signal() ทุกข้อ (ผ่านครบ = signal() ให้ BUY/SELL)"""
    r = df.iloc[-1]
    g = lambda k: None if pd.isna(r[k]) else float(r[k])
    close, ema, dmp, dmn, adx = g('close'), g('ema200'), g('dmp'), g('dmn'), g('adx')
    z, er, ar, slope = g('z'), g('er'), g('atr_ratio'), g('adx_slope')
    if close is None or ema is None:
        return None, [("ข้อมูลยังไม่พอคำนวณ", False, "n/a", "")]
    bias = 'BUY' if close > ema else 'SELL'
    st = structure_last(df)
    st_ok = (bias == 'BUY' and st in ['UPTREND', 'SIDEWAY']) or (bias == 'SELL' and st in ['DOWNTREND', 'SIDEWAY'])
    rows = []
    def add(name, ok, val, need=""):
        rows.append((name, bool(ok), val, "" if ok else need))
    add(f"DI ตรงทิศ {bias}  (+DI {'>' if bias == 'BUY' else '<'} -DI)",
        dmp is not None and dmn is not None and ((dmp > dmn) == (bias == 'BUY')),
        "n/a" if dmp is None or dmn is None else f"+DI {dmp:.1f} / -DI {dmn:.1f}", "DI ยังสวนทิศ")
    add(f"ADX > {ADX_MIN:g}", adx is not None and adx > ADX_MIN,
        "n/a" if adx is None else f"{adx:.1f}", "" if adx is None else f"ขาดอีก {ADX_MIN - adx:.1f}")
    if ADX_RISING:
        add("ADX กำลังเพิ่ม", slope is not None and slope > 0, "n/a" if slope is None else f"{slope:+.1f}", "ADX กำลังลด")
    add(f"ER ≥ {ER_MIN:g}  (ไม่ sideway)", er is not None and er >= ER_MIN,
        "n/a" if er is None else f"{er:.2f}", "" if er is None else f"ขาดอีก {ER_MIN - er:.2f}")
    add(f"ATR ratio ≥ {ATR_RATIO_MIN:g}", ar is not None and ar >= ATR_RATIO_MIN,
        "n/a" if ar is None else f"{ar:.2f}", "" if ar is None else f"ขาดอีก {ATR_RATIO_MIN - ar:.2f}")
    add(f"|z| < {Z_LIMIT:g}  (ไม่วิ่งไกลเกิน)", z is not None and abs(z) < Z_LIMIT,
        "n/a" if z is None else f"{abs(z):.2f}", "" if z is None else f"เกินอยู่ {abs(z) - Z_LIMIT:.2f}")
    add("โครงสร้างราคาไม่สวนทิศ", st_ok, st, f"{st} สวน {bias}")
    return bias, rows


def checklist_text(state, html=False):
    """ข้อความเช็กลิสต์ของทุกตลาดที่ยังไม่มีไม้เปิด (✅ ผ่านแล้ว / ❌ ยังไม่ผ่าน)"""
    import html as _h
    out = []
    for key, (bar_t, bias, rows) in LAST_CHECK.items():
        if key in state.get("positions", {}):
            continue
        rows = list(rows)
        if state.get("halted"):
            rows.append(("บอทหยุดเพราะทุนตกเกินกำหนด (DD halt)", False, "halted", "ลบ halted ใน state"))
        pu = state.get("pause_until")
        if pu and pd.Timestamp(bar_t) < pd.Timestamp(pu):
            rows.append(("ไม่อยู่ในช่วงพักเทรด", False, f"พักถึง {pu}", ""))
        n_ok = sum(1 for x in rows if x[1])
        head = f"🔎 {key} | แท่ง {bar_t} | ผ่าน {n_ok}/{len(rows)} ข้อ" + ("  → ผ่านครบ (เปิดไม้จริงหรือไม่ ขึ้นกับเพดานความเสี่ยง) " + str(bias) if n_ok == len(rows) else "")
        lines = [head]
        for name, ok, val, need in rows:
            lines.append(f"   {'✅' if ok else '❌'} {name}: {val}" + (f"  ({need})" if need else ""))
        out.append("\n".join(lines))
    txt = "\n".join(out)
    return _h.escape(txt) if html else txt


# ---------------- DATA SOURCES ----------------
def fetch_td(td_symbol):
    url = "https://api.twelvedata.com/time_series"
    p = {"symbol": td_symbol, "interval": INTERVAL, "outputsize": OUTPUTSIZE,
         "apikey": TD_KEY, "timezone": "UTC", "order": "ASC"}
    try:
        r = requests.get(url, params=p, timeout=20).json()
        if "values" not in r:
            return None, r.get("message", "no data")
        df = pd.DataFrame(r["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col])
        return add_ind(df), None
    except Exception as e:
        return None, str(e)


def fetch_yf(ticker):
    """yfinance ไม่มี 4h -> ดึง 1h แล้ว resample เป็น 4h"""
    try:
        import yfinance as yf
    except ImportError:
        return None, "ยังไม่ได้ลง yfinance (pip install yfinance)"
    try:
        raw = yf.download(ticker, period="90d", interval="1h",
                          progress=False, auto_adjust=False)
        if raw is None or raw.empty:
            return None, "yf empty"
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [str(c).lower() for c in raw.columns]
        if raw.index.tz is not None:
            raw.index = raw.index.tz_convert("UTC").tz_localize(None)
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        df = raw.resample("4h").agg(agg).dropna(subset=["open", "high", "low", "close"])
        if len(df) < 250:
            return None, f"ข้อมูลน้อยไป ({len(df)} แท่ง)"
        return add_ind(df), None
    except Exception as e:
        return None, str(e)


def drop_unclosed(df):
    """ตัดแท่งสุดท้ายทิ้งถ้ายังไม่ปิด — กันสัญญาณ/เช็ก TP-SL จากแท่งที่เพิ่งเปิดไม่กี่นาที"""
    if len(df) and df.index[-1] + pd.Timedelta(hours=BAR_HOURS) > pd.Timestamp.now("UTC").tz_localize(None):
        return df.iloc[:-1]
    return df


def get_data(sym):
    if sym["source"] == "yf":
        df, err = fetch_yf(sym["ticker"])
    else:
        df, err = fetch_td(sym["ticker"])
    if df is not None:
        df = drop_unclosed(df)
    return df, err


# ---------------- TELEGRAM ----------------
def tg_send(text):
    if not TG_TOKEN or not TG_CHAT:
        print("   [no telegram]", text.replace("\n", " ")[:80]); return
    try:
        _r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                           data={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"}, timeout=15)
        if _r.status_code != 200:
            print(f"   telegram ส่งไม่สำเร็จ: HTTP {_r.status_code} {_r.text[:200]}")
        else:
            print("   telegram ส่งแล้ว ✓")
    except Exception as e:
        print("   telegram error:", e)


def resolve_chat_id():
    global TG_CHAT
    if TG_CHAT:
        return
    try:
        r = requests.get(f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates", timeout=15).json()
        ups = r.get("result", [])
        if ups:
            TG_CHAT = str(ups[-1]["message"]["chat"]["id"])
            print(f"✅ เจอ chat id: {TG_CHAT}  (ใส่ TELEGRAM_CHAT_ID={TG_CHAT} ใน .env ไว้ก็ได้)")
        else:
            print("⚠️ ยังไม่เจอ chat id — เปิด Telegram ทักบอทตัวเอง 1 ข้อความ แล้วรันใหม่")
    except Exception as e:
        print("getUpdates error:", e)


# ---------------- STATE / LOG ----------------
def load_state():
    default = {"positions": {}, "last_bar": {}, "equity": PORTFOLIO_SIZE,
               "peak": PORTFOLIO_SIZE, "loss_streak": 0, "pause_until": "", "halted": False}
    if not os.path.exists(STATE_FILE):
        return default
    try:
        s = json.load(open(STATE_FILE))
        if not isinstance(s, dict):
            return default
        # กันไฟล์ state เก่า/เพี้ยน: บังคับให้ positions และ last_bar เป็น dict เสมอ
        if not isinstance(s.get("positions"), dict):
            s["positions"] = {}
        if not isinstance(s.get("last_bar"), dict):
            s["last_bar"] = {}
        if not isinstance(s.get("equity"), (int, float)):
            s["equity"] = PORTFOLIO_SIZE
        s.setdefault("peak", max(s["equity"], PORTFOLIO_SIZE))
        s.setdefault("loss_streak", 0)
        s.setdefault("pause_until", "")
        s.setdefault("halted", False)
        return s
    except Exception:
        return default


def save_state(s):
    json.dump(s, open(STATE_FILE, "w"), indent=2, default=str)


def log_row(row):
    new = not os.path.exists(LOG_FILE)
    import csv
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "event", "symbol", "validated", "type",
                        "price", "sl", "R_multiple", "note", "pnl_usd"])
        w.writerow(row)


# ---------------- MAIN ----------------
def calc_lot_risk(ticker, sl_dist, equity=None):
    """คืน (lot, risk จริงเป็น $, เป้า risk เป็น $)
    lot ปัดลงเป็นสเต็ป 0.01 และไม่ต่ำกว่า MIN_LOT -> ทุนน้อยอาจทำให้ risk จริง > RISK_PCT"""
    equity = PORTFOLIO_SIZE if equity is None else equity
    target = equity * RISK_PCT / 100.0
    cs = CONTRACT_SIZE.get(ticker, 1)
    loss_per_lot = sl_dist * cs
    lot = target / loss_per_lot if loss_per_lot > 0 else 0
    lot = max(MIN_LOT, int(lot / LOT_STEP + 1e-9) * LOT_STEP)
    lot = round(lot, 2)
    return lot, lot * loss_per_lot, target


def process(sym, state):
    df, err = get_data(sym)
    if df is None:
        print(f"   ⚠️ {sym['ticker']} ดึงไม่ได้: {err} (ข้าม)")
        return
    bar_t = str(df.index[-1])
    key = sym["ticker"]
    pos = state["positions"].get(key)
    tag = "✅" if sym["validated"] else "⚠️ backtest ขาดทุน"
    try:
        _b, _rows = checklist(df)
        LAST_CHECK[key] = (bar_t, _b, _rows)
        LAST_PRICE[key] = float(df.iloc[-1]["close"])
    except Exception as _e:
        print("   checklist error:", _e)

    r = df.iloc[-1]
    atr = r["atr"]; price = r["close"]

    # ---- มีไม้เปิดอยู่: เช็ก TP/SL (เฉพาะแท่งใหม่) ----
    if pos:
        if bar_t != pos.get("last_seen_bar"):
            pos["last_seen_bar"] = bar_t
            hi, lo = r["high"], r["low"]
            e, sd, typ = pos["entry"], pos["sl_dist"], pos["type"]
            if typ == "BUY":
                hit_sl, hit_tp = lo <= pos["sl"], hi >= pos["tp"]
            else:
                hit_sl, hit_tp = hi >= pos["sl"], lo <= pos["tp"]

            reason = None
            if hit_sl and hit_tp:          # แท่งเดียวโดนทั้งคู่ -> นับ SL (อนุรักษ์นิยม)
                reason, exitp, R = "SL", pos["sl"], -1.0
            elif hit_tp:
                reason, exitp, R = "TP", pos["tp"], float(RR)
            elif hit_sl:
                reason, exitp, R = "SL", pos["sl"], -1.0

            if reason:
                risk_amt = pos.get("risk_usd", PORTFOLIO_SIZE * RISK_PCT / 100.0)
                est = R * risk_amt - SPREAD_USD * pos.get("lot", MIN_LOT) * CONTRACT_SIZE.get(key, 1)
                state["equity"] = state.get("equity", PORTFOLIO_SIZE) + est
                state["peak"] = max(state.get("peak", PORTFOLIO_SIZE), state["equity"])
                state["loss_streak"] = state.get("loss_streak", 0) + 1 if est <= 0 else 0
                brk = ""
                if COOL_AFTER_LOSSES and state["loss_streak"] >= COOL_AFTER_LOSSES:
                    state["pause_until"] = str(df.index[-1] + pd.Timedelta(hours=COOL_BARS * BAR_HOURS))
                    state["loss_streak"] = 0
                    brk = f"\n⏸️ แพ้ติด {COOL_AFTER_LOSSES} ไม้ -> พักเข้าไม้ใหม่ {COOL_BARS} ชม."
                if state["equity"] < state["peak"] * (1 - DD_HALT_PCT / 100.0):
                    state["halted"] = True
                    brk += f"\n🛑 ทุนตกจากจุดสูงสุดเกิน {DD_HALT_PCT:g}% -> บอทหยุดหาสัญญาณ (ลบ halted ใน {STATE_FILE} เพื่อเริ่มใหม่)"
                ico = "✅ WIN" if R > 0 else "❌ LOSS"
                bar = "━" * 18
                tg_send(
                    f"🔴 <b>ปิดไม้ | {sym['label']} ({key})</b>\n{bar}\n"
                    f"ผล: <b>{ico}  {R:+.1f}R</b>  ({reason})\n"
                    f"💰 เข้า {e:.4f} → ออก {exitp:.4f}\n"
                    f"📏 ประมาณ: <b>{est:+.2f} USD</b>  <i>(หักสเปรดโดยประมาณ)</i>\n"
                    f"💼 ทุนประมาณ: <b>{state['equity']:.2f} USD</b>\n{bar}\n"
                    f"👉 (ถ้าตั้ง SL/TP ใน Exness ไว้แล้ว มันปิดเองอัตโนมัติ){brk}")
                log_row([datetime.now(), "EXIT", key, sym["validated"], typ,
                         round(exitp, 4), "", R, f"fixed {reason}, no cost", round(est, 2)])
                del state["positions"][key]
                return

    # ---- ไม่มีไม้: หาสัญญาณเข้า (เฉพาะแท่งใหม่) ----
    if not pos:
        if bar_t == state["last_bar"].get(key):
            print(f"   [debug] แท่ง {bar_t} เช็กไปแล้ว (last_bar ใน {os.path.abspath(STATE_FILE)}) -> ข้าม รอแท่งใหม่")
            return
        state["last_bar"][key] = bar_t
        if state.get("halted"):
            print("   [debug] halted=True ใน state -> บอทหยุดหาสัญญาณ")
            return
        pu = state.get("pause_until")
        if pu and df.index[-1] < pd.Timestamp(pu):
            print(f"   [debug] พักเทรดถึง {pu}")
            return
        bias = signal(df)
        if not bias:
            print("   [debug] signal() ไม่ให้ทิศ -> ไม่เข้า")
            return
        sl_dist = atr * ATR_SL_MULT
        sl = price - sl_dist if bias == "BUY" else price + sl_dist
        tp = price + sl_dist * RR if bias == "BUY" else price - sl_dist * RR
        equity = state.get("equity", PORTFOLIO_SIZE)
        lot, risk_amt, target_risk = calc_lot_risk(key, sl_dist, equity)
        if risk_amt > equity * MAX_RISK_PCT / 100.0 or \
                risk_amt + SPREAD_USD * lot * CONTRACT_SIZE.get(key, 1) >= equity:
            print(f"   ข้ามสัญญาณ {bias}: SL ${sl_dist:.1f} -> เสี่ยง ${risk_amt:.1f} "
                  f"({risk_amt / equity * 100:.0f}% ของทุน) เกินเพดาน {MAX_RISK_PCT:g}%")
            # แจ้งสัญญาณผ่าน Telegram แม้ไม่เปิดไม้ (ผ่านครบทุกข้อ แต่เสี่ยงเกินเพดานของพอร์ต)
            ico = "🚀" if bias == "BUY" else "🔻"
            bar = "━" * 18
            tg_send(
                f"{ico} <b>สัญญาณผ่านครบ | {sym['label']} ({key})</b> {tag}\n{bar}\n"
                f"📈 ทิศ: <b>{bias}</b>\n"
                f"💰 ราคา: ~{price:.4f}\n"
                f"🛡️ SL: {sl:.4f}  <i>(ระยะ ${sl_dist:.1f})</i>\n"
                f"🎯 TP: {tp:.4f}   RR 1:{RR:g}\n{bar}\n"
                f"⚠️ <b>บอทไม่ได้เปิดไม้ให้</b> เพราะ lot {lot} เสี่ยง ${risk_amt:.1f} "
                f"({risk_amt / equity * 100:.0f}% ของทุน) เกินเพดาน {MAX_RISK_PCT:g}%\n"
                f"<i>แจ้งเป็นข้อมูลเท่านั้น ตัดสินใจเข้าเองได้ถ้ารับความเสี่ยงไหว</i>")
            log_row([datetime.now(), "SIGNAL_ONLY", key, sym["validated"], bias,
                     round(price, 4), round(sl, 4), "", f"skipped: risk {risk_amt:.1f} > cap", ""])
            return
        reward_amt = risk_amt * RR
        over = risk_amt > target_risk * 1.25
        warn = (f"\n⚠️ lot ขั้นต่ำ {MIN_LOT} ทำให้เสี่ยงจริง {risk_amt / equity * 100:.0f}% "
                f"ของพอร์ต (เป้า {RISK_PCT:g}%)") if over else ""
        ico = "🚀" if bias == "BUY" else "🔻"
        bar = "━" * 18
        state["positions"][key] = {
            "type": bias, "entry": price, "sl": sl, "tp": tp, "sl_dist": sl_dist,
            "entry_time": bar_t, "last_seen_bar": bar_t, "lot": lot, "risk_usd": risk_amt}
        tg_send(
            f"{ico} <b>สัญญาณเข้า | {sym['label']} ({key})</b> {tag}\n{bar}\n"
            f"📈 ทิศ: <b>{bias}</b>\n"
            f"💰 เข้า: ~{price:.4f}\n"
            f"📦 Lot: <b>{lot}</b>\n"
            f"🛡️ SL: {sl:.4f}  <i>(−${risk_amt:.0f})</i>\n"
            f"🎯 TP: {tp:.4f}  <i>(+${reward_amt:.0f})</i>   RR 1:{RR:g}{warn}\n{bar}\n"
            f"<i>อ้างอิง TwelveData/yfinance — ราคาจริงต่างนิดหน่อย</i>\n"
            f"👉 ตั้ง SL/TP ใน Exness ครั้งเดียวจบ")
        log_row([datetime.now(), "ENTRY", key, sym["validated"], bias,
                 round(price, 4), round(sl, 4), "", f"fixed TP, no cost, lot {lot}", ""])


def report_text(state, cycle=0):
    """หน้ารายงานแบบกระชับ (~10 บรรทัด) แคปจอเดียวพอดี: ไม้ที่เปิด + กำไรลอย + เช็กลิสต์ + ผลสะสม"""
    W = 46
    now = datetime.now().strftime("%H:%M:%S")
    L = ["═" * W, f" รายงาน {now} | รอบ #{cycle}", "═" * W]
    short = (("DI", "DI"), ("ADX กำลัง", "ADXขึ้น"), ("ADX", "ADX"), ("ER", "ER"),
             ("ATR", "ATR"), ("|z|", "z"), ("โครงสร้าง", "โครง"))
    for key, (bar_t, bias, rows) in LAST_CHECK.items():
        px = LAST_PRICE.get(key)
        L.append(f" {key} | แท่ง {str(bar_t)[5:16]} | ราคา {px:.2f}" if px else f" {key} | แท่ง {bar_t}")
        pos = state.get("positions", {}).get(key)
        if pos:
            d = 1 if pos["type"] == "BUY" else -1
            L.append(f" 📌 {pos['type']} @ {pos['entry']:.2f}  lot {pos.get('lot', MIN_LOT)}")
            L.append(f"    SL {pos['sl']:.2f}  |  TP {pos['tp']:.2f}")
            if px:
                fl = (px - pos["entry"]) * d * pos.get("lot", MIN_LOT) * CONTRACT_SIZE.get(key, 1)
                risk = pos.get("risk_usd") or 1
                to_tp = abs(pos["tp"] - px); to_sl = abs(px - pos["sl"])
                L.append(f"    ลอยตอนนี้ {fl:+.2f}$ ({fl / risk:+.2f}R)")
                L.append(f"    ห่าง TP {to_tp:.1f}$ | ห่าง SL {to_sl:.1f}$")
        else:
            L.append(f" สถานะ: {status_line(state)}")
        items = []
        for name, ok, val, need in rows:
            lab = next((s for pre, s in short if name.startswith(pre)), name[:6])
            v = "" if lab == "DI" else f" {val}"
            items.append(f"{lab}{v}{'✅' if ok else '❌'}")
        n_ok = sum(1 for r in rows if r[1])
        L.append(f" เช็กลิสต์ {n_ok}/{len(rows)} (ทิศ {bias}):")
        for i in range(0, len(items), 3):
            L.append("   " + "  ".join(items[i:i + 3]))
    entries, exits = count_trades()
    eq = state.get("equity", PORTFOLIO_SIZE)
    L.append(f" ทุน {eq:.2f}$ | เข้า {entries} ออก {exits} ไม้")
    if state.get("halted"):
        L.append(" 🛑 บอทหยุด (DD halt) — ลบ halted ใน state")
    L.append("═" * W)
    return "\n".join(L)


def status_line(state):
    """สรุปสถานะปัจจุบันเป็นข้อความสั้น"""
    pos = state.get("positions", {})
    if not pos:
        return "ไม่มีไม้เปิดอยู่ (กำลังรอสัญญาณ)"
    parts = []
    for k, p in pos.items():
        parts.append(f"{k} {p['type']} @ {p['entry']:.2f} SL {p['sl']:.2f}")
    return f"เปิดอยู่ {len(pos)} ไม้: " + " | ".join(parts)


def count_trades():
    """นับไม้ทั้งหมดจาก log (เข้า/ออกกี่ครั้งแล้ว)"""
    if not os.path.exists(LOG_FILE):
        return 0, 0
    try:
        df = pd.read_csv(LOG_FILE)
        return int((df["event"] == "ENTRY").sum()), int((df["event"] == "EXIT").sum())
    except Exception:
        return 0, 0


def perf_summary():
    """สรุปผลรวมจาก log: ปิดไปกี่ไม้ ชนะกี่ไม้ รวมกี่ R + ประมาณ $"""
    if not os.path.exists(LOG_FILE):
        return "ยังไม่มีไม้ปิด"
    try:
        df = pd.read_csv(LOG_FILE)
        ex = df[df["event"] == "EXIT"].copy()
        if ex.empty:
            return "ยังไม่มีไม้ปิด (รอไม้แรกจบ)"
        ex["R_multiple"] = pd.to_numeric(ex["R_multiple"], errors="coerce")
        n = len(ex); wins = int((ex["R_multiple"] > 0).sum())
        totR = ex["R_multiple"].sum()
        wr = wins / n * 100
        est = pd.to_numeric(ex["pnl_usd"], errors="coerce").sum()
        return (f"ปิดแล้ว {n} ไม้ | ชนะ {wins} ({wr:.0f}%)\n"
                f"รวม <b>{totR:+.2f}R</b>  ≈ {est:+.0f} USD <i>(ยังไม่รวมต้นทุน)</i>")
    except Exception as e:
        return f"อ่าน log ไม่ได้: {e}"


def main():
    if not TD_KEY:
        print("⚠️ ไม่มี TWELVEDATA_API_KEY — ทองจะดึงไม่ได้ (ตัวอื่นใช้ yfinance ยังได้)")
    if not TG_TOKEN:
        print("⚠️ ไม่มี TELEGRAM_BOT_TOKEN — จะพิมพ์สัญญาณลง console แทน")
    resolve_chat_id()
    tg_send(f"🤖 Signal bot เริ่มทำงาน — เฝ้าทองคำ (XAU/USD) ทุน {PORTFOLIO_SIZE:.0f} USD\n"
            "จะแจ้งเตือนเมื่อมีสัญญาณเข้า/ออก + สรุปสถานะเป็นระยะ")
    state = load_state()
    print(f">> state: {os.path.abspath(STATE_FILE)} | last_bar={state.get('last_bar')} | equity={state.get('equity')}")
    print(f">> เริ่มลูป | {len(SYMBOLS)} ตลาด | เช็กทุก {POLL_SEC}s | Ctrl+C หยุด\n")

    cycle = 0
    last_summary = time.time()
    SUMMARY_EVERY = 6 * 3600   # ส่งสรุปเข้า Telegram ทุก 6 ชม.

    while True:
        try:
            cycle += 1
            if CLEAR_SCREEN and cycle > 1:
                os.system("cls" if os.name == "nt" else "clear")
            for sym in SYMBOLS:
                process(sym, state)
                save_state(state)
                time.sleep(REQ_GAP)

            # --- heartbeat ลง console ทุกรอบ (รู้ว่ายังรันอยู่) ---
            now = datetime.now().strftime("%H:%M:%S")
            print(report_text(state, cycle))

            # --- สรุปเข้า Telegram ทุก 6 ชม. (รู้ว่ายังไม่ตายจากในมือถือ) ---
            if time.time() - last_summary >= SUMMARY_EVERY:
                bar = "━" * 18
                tg_send(f"📊 <b>สรุปสถานะ</b> ({now})\n{bar}\n"
                        f"บอทยังทำงานปกติ ✅\n"
                        f"{status_line(state)}\n{bar}\n"
                        f"{checklist_text(state, html=True)}\n{bar}\n"
                        f"📈 <b>ผลสะสม:</b>\n{perf_summary()}")
                last_summary = time.time()

            time.sleep(POLL_SEC)
        except KeyboardInterrupt:
            print("\n>> หยุดแล้ว"); save_state(state); break
        except Exception as e:
            print("   loop error:", e); time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()