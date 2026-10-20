import time
import requests
import pandas as pd
import numpy as np
import warnings
import os
import json
import csv
import joblib
from dotenv import load_dotenv
from twelvedata import TDClient
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import pandas_ta as ta

warnings.filterwarnings('ignore')

# ========= CONFIG & ENV =========
load_dotenv()

API_KEY = os.getenv("TWELVEDATA_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not API_KEY or not TELEGRAM_TOKEN or not CHAT_ID:
    raise ValueError("❌ Missing API Keys! Please check your .env file.")

# 🛠️ [FIX 1] เปลี่ยนจาก 15min เป็น 1h ตามสถิติ Backtest ที่ชนะตลาด
SYMBOL = "XAU/USD"
INTERVAL = "1h" 

INITIAL_BALANCE = 3000
# 🛠️ [FIX 2] เพิ่มความเสี่ยงเป็น 2% เพราะ Max Drawdown เราต่ำแค่ 13%
RISK_PER_TRADE_PERCENT = 2.0 
MAX_DAILY_LOSS_PERCENT = 4.0 # ขยับขึ้นเล็กน้อยเพื่อให้สอดคล้องกับ Risk 2%
GOLD_CONTRACT_SIZE = 100
MAX_LOT_SIZE = 0.5  

STATE_FILE = "state.json"
MODEL_FILE = "rf_model.joblib"

START_HOUR = 14
END_HOUR = 2

# 🛠️ [FIX 3] ปรับจูนค่าฟิลเตอร์ต่างๆ ตามผล Backtest ล่าสุด
ADX_MIN = 22
VOL_SPIKE_LIMIT = 2.0
AI_CONF_MIN = 0.58  # ความมั่นใจ AI ที่จุดสมดุล
Z_SCORE_LIMIT = 1.5

# ========= TELEGRAM =========
def send_telegram(msg):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
        )
    except:
        pass

def send_entry_telegram(t_type, price, lot, sl, tp):
    risk = abs(price - sl) * lot * GOLD_CONTRACT_SIZE
    reward = abs(tp - price) * lot * GOLD_CONTRACT_SIZE
    rr = reward / risk if risk != 0 else 0
    emoji = "🚀 BUY" if t_type == "BUY" else "🔻 SELL"
    msg = (
        f"{emoji} | *XAU PURE TREND*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Entry: `{price:.2f}`\n"
        f"📦 Lot: `{lot:.2f}`\n\n"
        f"🛡️ SL: `{sl:.2f}` (-${risk:.2f})\n"
        f"🎯 TP: `{tp:.2f}` (+${reward:.2f})\n"
        f"📊 RR: `{rr:.2f}`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━"
    )
    send_telegram(msg)

def send_close_telegram(pnl):
    emoji = "✅ WIN" if pnl > 0 else "❌ LOSS"
    send_telegram(
        f"🔔 *TRADE CLOSED*\n━━━━━━━━━━━━━━━━━━━━━━\n{emoji}\n💵 `{pnl:+.2f} USD`\n━━━━━━━━━━━━━━━━━━━━━━"
    )

# ========= DATA LOGGING =========
def save_metrics(s):
    total = s["total_trades"]
    wr = (s["wins"] / total * 100) if total > 0 else 0.0
    lr = (s["losses"] / total * 100) if total > 0 else 0.0
    pf = (s["gross_profit"] / s["gross_loss"]) if s["gross_loss"] > 0 else (float('inf') if s["gross_profit"] > 0 else 0.0)
    metrics = {
        "Last_Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Current_Balance": round(s["current_balance"], 2),
        "Total_Trades": total,
        "Wins": s["wins"],
        "Losses": s["losses"],
        "Win_Rate_%": round(wr, 2),
        "Loss_Rate_%": round(lr, 2),
        "Profit_Factor": round(pf, 2),
        "Max_Drawdown_%": round(s["max_drawdown"], 2)
    }
    with open("performance_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

# ========= STATE =========
def load_state():
    default_state = {
        "is_open": False, "daily_loss": 0, "last_day": "", "be_done": False,
        "current_balance": INITIAL_BALANCE,
        "start_day_balance": INITIAL_BALANCE,
        "peak_balance": INITIAL_BALANCE,
        "total_trades": 0, "wins": 0, "losses": 0,
        "gross_profit": 0.0, "gross_loss": 0.0,
        "max_drawdown": 0.0,
        "sl_dist_cache": 0.0
    }
    if os.path.exists(STATE_FILE):
        try:
            s = json.load(open(STATE_FILE))
            for k, v in default_state.items():
                if k not in s:
                    s[k] = v
            return s
        except:
            return default_state
    return default_state

def save_state(s):
    json.dump(s, open(STATE_FILE, "w"))

# ========= LOT =========
def calc_lot(balance, risk, sl):
    if sl <= 0:
        return 0.01
    calculated_lot = round((balance * (risk / 100)) / (sl * GOLD_CONTRACT_SIZE), 2)
    return max(min(calculated_lot, MAX_LOT_SIZE), 0.01)

# ========= STANDARD ADX (pandas_ta) =========
def calculate_indicators(df):
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx_df is not None:
        df['adx']    = adx_df['ADX_14']
        df['dmp']    = adx_df['DMP_14']  
        df['dmn']    = adx_df['DMN_14']  
    else:
        df['adx'] = np.nan; df['dmp'] = np.nan; df['dmn'] = np.nan

    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['z'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['ema20'] = ta.ema(df['close'], length=20)
    df['ema50'] = ta.ema(df['close'], length=50)
    df['ema20_dist'] = (df['close'] - df['ema20']) / df['ema20'] * 100

    return df

# ========= MAIN =========
def run():
    try:
        td = TDClient(apikey=API_KEY)
        s = load_state()
        now = datetime.now()

        # Reset day
        if s["last_day"] != now.strftime("%Y-%m-%d"):
            s["daily_loss"] = 0
            s["last_day"] = now.strftime("%Y-%m-%d")
            s["start_day_balance"] = s["current_balance"]

        # Kill switch
        daily_loss_limit = s["start_day_balance"] * (MAX_DAILY_LOSS_PERCENT / 100)
        if s["daily_loss"] <= -daily_loss_limit:
            print(f"🛑 STOP DAILY LOSS LIMIT REACHED (-${daily_loss_limit:.2f})")
            return

        # คำนวณช่วงเวลาที่อนุญาตให้เทรด (รองรับการข้ามเที่ยงคืน)
        current_hour = now.hour
        if START_HOUR <= END_HOUR:
            is_time = (START_HOUR <= current_hour <= END_HOUR)
        else:
            is_time = (current_hour >= START_HOUR or current_hour <= END_HOUR)

        df = td.time_series(symbol=SYMBOL, interval=INTERVAL, outputsize=200).as_pandas().sort_index()
        # ใช้ timeframe เดิม (1h) เพื่อเทียบ EMA200 
        h1 = td.time_series(symbol=SYMBOL, interval="1h", outputsize=250).as_pandas().sort_index()

        price = float(td.price(symbol=SYMBOL).as_json()['price'])

        # ===== INDICATORS =====
        df = calculate_indicators(df)

        atr      = df['atr'].iloc[-1]
        cur_adx  = df['adx'].iloc[-1]
        cur_dmp  = df['dmp'].iloc[-1]   
        cur_dmn  = df['dmn'].iloc[-1]   
        cur_z    = df['z'].iloc[-1]

        if any(v is None or (isinstance(v, float) and np.isnan(v)) 
               for v in [atr, cur_adx, cur_dmp, cur_dmn, cur_z]):
            print("⚠️  Indicator not ready (not enough data), skipping...")
            return

        di_trend = "BUY" if cur_dmp > cur_dmn else "SELL"

        h1['ema200'] = ta.ema(h1['close'], length=200)
        ema200_h1 = h1['ema200'].iloc[-1]

        if ema200_h1 is None or (isinstance(ema200_h1, float) and np.isnan(ema200_h1)):
            print("⚠️  EMA200 H1 not ready (not enough H1 data), skipping...")
            return

        trend = "BUY" if price > ema200_h1 else "SELL"

        vol_spike = 1
        if 'volume' in df.columns:
            vol_spike = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]

        # ===== MARKET STRUCTURE =====
        window_size = 5
        df['swing_high'] = df['high'][df['high'] == df['high'].rolling(window=window_size*2+1, center=True).max()]
        df['swing_low']  = df['low'][df['low'] == df['low'].rolling(window=window_size*2+1, center=True).min()]
        recent_highs = df['swing_high'].dropna().tail(2).values
        recent_lows  = df['swing_low'].dropna().tail(2).values

        market_structure = "SIDEWAY"
        if len(recent_highs) == 2 and len(recent_lows) == 2:
            if recent_lows[1] > recent_lows[0] and recent_highs[1] > recent_highs[0]:
                market_structure = "UPTREND"
            elif recent_lows[1] < recent_lows[0] and recent_highs[1] < recent_highs[0]:
                market_structure = "DOWNTREND"

        # ===== AI =====
        ML_FEATURES = ['adx', 'dmp', 'dmn', 'z', 'rsi', 'ema20_dist']
        df_ml = df[ML_FEATURES + ['close']].dropna()
        X = df_ml[ML_FEATURES]
        y = np.where(df_ml['close'].shift(-1) > df_ml['close'], 1, 0)

        train_model = True
        is_new_model = False
        if os.path.exists(MODEL_FILE):
            mtime = os.path.getmtime(MODEL_FILE)
            if (time.time() - mtime) < 86400:
                train_model = False

        if train_model:
            model = RandomForestClassifier(n_estimators=80, max_depth=4, random_state=42)
            model.fit(X[:-1], y[:-1])
            joblib.dump(model, MODEL_FILE)
            is_new_model = True
        else:
            model = joblib.load(MODEL_FILE)

        prob  = model.predict_proba(X.iloc[[-1]])[0][1]
        bias  = "BUY" if prob > 0.5 else "SELL"
        conf  = prob if bias == "BUY" else (1 - prob)

        # ===== FILTER =====
        structure_valid = (bias == "BUY" and market_structure in ["UPTREND", "SIDEWAY"]) or \
                          (bias == "SELL" and market_structure in ["DOWNTREND", "SIDEWAY"])

        # 🛠️ [FIX 4] อัปเดตเงื่อนไขให้ตรงกับที่ Backtest ชนะ
        valid = (
            cur_adx > ADX_MIN and          
            di_trend == bias and            
            vol_spike < VOL_SPIKE_LIMIT and
            bias == trend and
            conf >= AI_CONF_MIN and
            abs(cur_z) < Z_SCORE_LIMIT and
            structure_valid
        )

        # ===== DASHBOARD =====
        os.system('cls' if os.name == 'nt' else 'clear')
        print("╔══════════════════════════════════════╗")
        print(f"║ 🤖 QUANT ULITE v6.4 (PROD READY)     ║")
        print("╠══════════════════════════════════════╣")
        print(f"║ 🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"║ 💰 BALANCE: ${s['current_balance']:.2f}")
        print(f"║ 🪙 PRICE:   {price:.2f}")
        print(f"║ 📊 TREND (1H EMA200): {trend}")
        print(f"║ 📐 STRUCTURE: {market_structure}")
        print(f"║ 🧠 AI: {bias} ({conf*100:.1f}%) {'[NEW MODEL]' if is_new_model else '[CACHED]'}")
        print(f"║ 📈 ADX: {cur_adx:.1f} | +DI: {cur_dmp:.1f} | -DI: {cur_dmn:.1f}")
        print(f"║ 📉 Z: {cur_z:.2f} | ATR: {atr:.2f}")
        print("╠══════════════════════════════════════╣")
        print("║ 🎯 FILTERS (INSTITUTIONAL GRADE)")
        print(f"║ TIME ALIGN (14-02): {'✅' if is_time else '❌'}")
        print(f"║ ADX (>{ADX_MIN}):       {'✅' if cur_adx > ADX_MIN else '❌'} ({cur_adx:.1f})")
        print(f"║ DI Direction:    {'✅' if di_trend == bias else '❌'} (+DI={cur_dmp:.1f} -DI={cur_dmn:.1f})")
        print(f"║ VOL Spike:       {'✅' if vol_spike < VOL_SPIKE_LIMIT else '❌'} ({vol_spike:.2f}x)")
        print(f"║ EMA200 Trend:    {'✅' if bias == trend else '❌'} ({trend})")
        print(f"║ Structure:       {'✅' if structure_valid else '❌'} ({market_structure})")
        print(f"║ AI Conf (>={AI_CONF_MIN*100}%): {'✅' if conf >= AI_CONF_MIN else '❌'} ({conf*100:.1f}%)")
        print(f"║ Z-Score (<{Z_SCORE_LIMIT}):      {'✅' if abs(cur_z) < Z_SCORE_LIMIT else '❌'} ({cur_z:.2f})")
        print("╠══════════════════════════════════════╣")

        # ===== ENTRY =====
        if not s["is_open"]:
            print(f"║ STATUS: {'✅ READY TO TRADE' if (valid and is_time) else '⏳ WAITING...'}")

            if is_time and valid:
                sl_dist = atr * 1.5
                rr      = 1.5  # 🛠️ [FIX 5] ปรับ RR เป็น 1.5 เพื่อให้ชนกำไรไวขึ้น Win Rate แตะ 40%+
                
                lot     = calc_lot(s["current_balance"], RISK_PER_TRADE_PERCENT, sl_dist)
                sl      = price - sl_dist if bias == "BUY" else price + sl_dist
                tp      = price + sl_dist * rr if bias == "BUY" else price - sl_dist * rr

                s.update({
                    "is_open": True, "type": bias, "entry": price,
                    "sl": sl, "tp": tp, "lot": lot,
                    "be_done": False, "sl_dist_cache": sl_dist
                })
                send_entry_telegram(bias, price, lot, sl, tp)

        else:
            entry = s["entry"]
            lot   = s["lot"]
            pnl   = ((price - entry) if s["type"] == "BUY" else (entry - price)) * lot * GOLD_CONTRACT_SIZE
            price_move = (price - entry) if s["type"] == "BUY" else (entry - price)

            print(f"║ 📦 OPEN: {s['type']} @ {s['entry']:.2f}")
            print(f"║ 💵 PnL: ${pnl:+.2f}")
            print(f"║ 🎯 TP: {s['tp']:.2f} | 🛡️ SL: {s['sl']:.2f}")

            # Break-even ถูกปิดใช้งานแล้ว (ตามคำขอ) — SL/TP จะคงค่าเดิมตามตอนเปิดออเดอร์
            # ไม้จะปิดที่ TP จริง หรือ SL จริงเท่านั้น ไม่มีการเลื่อน SL มาที่จุดเข้าอีกต่อไป

            # --- ตรรกะการเช็กชน TP/SL แบบ Real-time + ไส้เทียน ---
            candle_high = df['high'].iloc[-1]
            candle_low  = df['low'].iloc[-1]

            if s["type"] == "BUY":
                hit_tp = (price >= s["tp"]) or (candle_high >= s["tp"])
                hit_sl = (price <= s["sl"]) or (candle_low <= s["sl"])
            else:
                hit_tp = (price <= s["tp"]) or (candle_low <= s["tp"])
                hit_sl = (price >= s["sl"]) or (candle_high >= s["sl"])
            # --------------------------------------------------

            if hit_tp or hit_sl:
                # คำนวณราคาปิดตามเป้าหมายระบบ
                exit_price = s["tp"] if hit_tp else s["sl"]
                pnl = ((exit_price - entry) if s["type"] == "BUY" else (entry - exit_price)) * lot * GOLD_CONTRACT_SIZE
                
                s["daily_loss"]    += pnl
                s["current_balance"] += pnl
                s["total_trades"]  += 1

                if pnl > 0:
                    s["wins"] += 1
                    s["gross_profit"] += pnl
                else:
                    s["losses"] += 1
                    s["gross_loss"] += abs(pnl)

                if s["current_balance"] > s["peak_balance"]:
                    s["peak_balance"] = s["current_balance"]

                dd = (s["peak_balance"] - s["current_balance"]) / s["peak_balance"] * 100
                if dd > s["max_drawdown"]:
                    s["max_drawdown"] = dd

                # บันทึก trade history
                file_exists = os.path.isfile('trade_history.csv')
                with open('trade_history.csv', 'a', newline='', encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    if not file_exists:
                        writer.writerow(['Timestamp', 'Type', 'Entry', 'Exit', 'Lot', 'PnL', 'Balance'])
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        s["type"], round(entry, 2), round(exit_price, 2),
                        lot, round(pnl, 2), round(s["current_balance"], 2)
                    ])

                save_metrics(s)
                send_close_telegram(pnl)
                s["is_open"]       = False
                s["sl_dist_cache"] = 0.0

        print("╚══════════════════════════════════════╝")
        save_state(s)

    except Exception as e:
        error_msg = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR: {str(e)}"
        print(f"\n{error_msg}")
        with open("error_log.txt", "a", encoding="utf-8") as f:
            f.write(error_msg + "\n")

# ========= LOOP =========
while True:
    run()
    # หน่วงเวลา 5 นาที (300 วินาที) ตื่นมาเฝ้าราคาชน SL/TP และส่งสัญญาณความไวสูง
    time.sleep(300)