#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BACKTEST ทองคำ (XAU/USD 4h) — ใช้ logic เดียวกับ Signal.py
ทุนเริ่มต้น 150 USD, เสี่ยง 2%/ไม้ (lot ขั้นต่ำ 0.01 ทำให้ risk จริงอาจสูงกว่านี้ — สคริปต์นับให้)

วิธีรัน (วางไฟล์นี้ไว้โฟลเดอร์เดียวกับ Signal.py):
  python backtest_gold.py --csv xauusd_4h.csv        # ใช้ไฟล์ราคาของนายเอง
  python backtest_gold.py                            # ดึงจาก TwelveData (ต้องมี TWELVEDATA_API_KEY ใน .env)
  python backtest_gold.py --csv xauusd_4h.csv --sweep   # ลองหลาย ATR_SL_MULT x RR เทียบกัน

CSV ต้องมีคอลัมน์เวลา (datetime/time/date/timestamp) + open, high, low, close
ค่าเริ่มต้น (timeframe, SL/RR, เพดาน risk, สเปรด) ดึงจาก Signal.py ให้ตรงกับบอทเสมอ
  เพิ่ม --rolling เพื่อดูการกระจายของผลจากหลายจุดเริ่ม (ทุนเล็กผลขึ้นกับโชคช่วงแรกมาก)
ข้อสมมติ (ตั้งแก้ได้ด้านล่าง / ผ่าน argument):
  - เข้าที่ราคาปิดแท่งที่เกิดสัญญาณ, เช็ก TP/SL ตั้งแต่แท่งถัดไปด้วย high/low
  - แท่งเดียวโดนทั้ง TP และ SL = นับ SL ; เปิดแท่งใหม่ทะลุ SL (gap) = ออกที่ราคาเปิด
  - หักสเปรดต่อไม้ (USD/oz) — ค่าเริ่มต้น 0.30 ปรับตามบัญชีจริงของนาย
  - ไม่รวม swap/commission/slippage อื่น ๆ
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd

import Signal as sig   # ใช้ indicator + เงื่อนไข + ค่า ADX_MIN/Z_LIMIT เดียวกับบอทจริง

CAPITAL_DEFAULT = 150.0


# ---------------- DATA ----------------
def load_csv(path, tf=4):
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    tcol = next((c for c in ["datetime", "time", "date", "timestamp", "gmt time", "local time"]
                 if c in df.columns), None)
    if tcol is None:
        tcol = df.columns[0]
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=[tcol]).set_index(tcol).sort_index()
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            sys.exit(f"CSV ไม่มีคอลัมน์ {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["open", "high", "low", "close"]].dropna()
    df = df[~df.index.duplicated(keep="last")]
    gap = df.index.to_series().diff().median()
    if gap < pd.Timedelta(hours=tf):     # ไฟล์ถี่กว่า timeframe ที่ขอ -> รวมแท่ง
        print(f"ไฟล์เป็นแท่ง {gap} -> resample เป็น {tf}h")
        df = df.resample(f"{tf}h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return df


def load_td(size=5000, tf=4):
    import requests
    if not sig.TD_KEY:
        sys.exit("ไม่มี TWELVEDATA_API_KEY — ใส่ใน .env หรือใช้ --csv")
    p = {"symbol": "XAU/USD", "interval": f"{tf}h", "outputsize": size,
         "apikey": sig.TD_KEY, "timezone": "UTC", "order": "ASC"}
    r = requests.get("https://api.twelvedata.com/time_series", params=p, timeout=30).json()
    if "values" not in r:
        sys.exit(f"TwelveData error: {r.get('message', r)}")
    df = pd.DataFrame(r["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c])
    return df[["open", "high", "low", "close"]]


# ---------------- SIGNALS (ไม่มี look-ahead) ----------------
def structure_series(high, low, w=5):
    """เหมือน sig.structure_last แต่คำนวณทุกแท่งในรอบเดียว
    pivot ที่ index p จะ 'ยืนยัน' ได้ตอนแท่ง p+w เท่านั้น (ตรงกับที่บอทเห็นตอนนั้น)"""
    n = len(high)
    is_sh = np.zeros(n, bool)
    is_sl = np.zeros(n, bool)
    for p in range(w, n - w):
        if high[p] == high[p - w:p + w + 1].max():
            is_sh[p] = True
        if low[p] == low[p - w:p + w + 1].min():
            is_sl[p] = True
    out = np.array(["SIDEWAY"] * n, dtype=object)
    sh, sl = [], []
    for t in range(n):
        p = t - w
        if p >= w:
            if is_sh[p]:
                sh.append(high[p])
            if is_sl[p]:
                sl.append(low[p])
        if len(sh) >= 2 and len(sl) >= 2:
            if sl[-1] > sl[-2] and sh[-1] > sh[-2]:
                out[t] = "UPTREND"
            elif sl[-1] < sl[-2] and sh[-1] < sh[-2]:
                out[t] = "DOWNTREND"
    return out


def signal_series(df):
    """คืน array ของ 'BUY'/'SELL'/None ต่อแท่ง ด้วยเงื่อนไขเดียวกับ sig.signal()"""
    st = structure_series(df["high"].values, df["low"].values)
    bias = np.where(df["close"] > df["ema200"], "BUY", "SELL")
    di = np.where(df["dmp"] > df["dmn"], "BUY", "SELL")
    st_ok = ((bias == "BUY") & np.isin(st, ["UPTREND", "SIDEWAY"])) | \
            ((bias == "SELL") & np.isin(st, ["DOWNTREND", "SIDEWAY"]))
    valid = df[["atr", "adx", "dmp", "dmn", "z", "ema200"]].notna().all(axis=1).values
    ok = (valid & (df["adx"].values > sig.ADX_MIN) & (di == bias) &
          (df["vol_spike"].values < sig.VOL_LIM) & (np.abs(df["z"].values) < sig.Z_LIMIT) & st_ok)
    return np.where(ok, bias, None), st


def verify_equivalence(df, sigs, n_samples=25, seed=1):
    """เทียบผลกับ sig.signal() ตัวจริงบนแท่งสุ่ม — ถ้าไม่ตรง แปลว่า backtest ไม่เหมือนบอท"""
    rnd = random.Random(seed)
    idx = rnd.sample(range(250, len(df)), min(n_samples, len(df) - 250))
    bad = 0
    for t in idx:
        ref = sig.signal(df.iloc[:t + 1])
        if ref != sigs[t]:
            bad += 1
    return bad, len(idx)


# ---------------- SIMULATION ----------------
def simulate(df, sigs, capital=CAPITAL_DEFAULT, risk_pct=2.0, atr_mult=1.5, rr=2.0,
             spread=0.30, contract=100, min_lot=0.01, lot_step=0.01, leverage=500,
             warmup=250, start=None, end=None, max_risk_pct=None, hours=None):
    """start/end = ช่วงแท่งที่ให้เปิดไม้ (index) ; max_risk_pct = ข้ามสัญญาณถ้า risk จริง (หลังติด lot ขั้นต่ำ)
    เกิน % นี้ของทุน ; hours = ชุดชั่วโมง UTC ของแท่งสัญญาณที่อนุญาตให้เข้า"""
    o, h, l, c = (df[k].values for k in ["open", "high", "low", "close"])
    atr = df["atr"].values
    times = df.index
    n = len(df)
    eq = capital
    peak = capital
    trades = []
    skipped = 0
    gated = 0
    ruined = False
    t = max(warmup, start or 0)
    t_end = min(n - 1, end if end is not None else n - 1)
    while t < t_end:
        side = sigs[t]
        if side is None or np.isnan(atr[t]):
            t += 1
            continue
        if hours is not None and times[t].hour not in hours:
            t += 1
            continue
        entry = c[t]
        sl_dist = atr[t] * atr_mult
        loss_per_lot = sl_dist * contract
        target = eq * risk_pct / 100.0
        lot = max(min_lot, int(target / loss_per_lot / lot_step + 1e-9) * lot_step)
        lot = round(lot, 2)
        if max_risk_pct is not None and lot * loss_per_lot > eq * max_risk_pct / 100.0:
            gated += 1
            t += 1
            continue
        worst = lot * loss_per_lot + lot * contract * spread
        margin = lot * contract * entry / leverage
        if worst >= eq or margin >= eq:      # ถ้าโดน SL แล้วทุนไม่พอ/ margin ไม่พอ = เทรดไม่ได้
            skipped += 1
            t += 1
            continue
        sl = entry - sl_dist if side == "BUY" else entry + sl_dist
        tp = entry + sl_dist * rr if side == "BUY" else entry - sl_dist * rr
        exit_i, exit_p, reason = None, None, None
        for j in range(t + 1, n):
            if side == "BUY":
                if o[j] <= sl:
                    exit_i, exit_p, reason = j, o[j], "SL(gap)"; break
                hit_sl, hit_tp = l[j] <= sl, h[j] >= tp
            else:
                if o[j] >= sl:
                    exit_i, exit_p, reason = j, o[j], "SL(gap)"; break
                hit_sl, hit_tp = h[j] >= sl, l[j] <= tp
            if hit_sl:
                exit_i, exit_p, reason = j, sl, "SL"; break
            if hit_tp:
                exit_i, exit_p, reason = j, tp, "TP"; break
        if exit_i is None:       # จบข้อมูลก่อนไม้ปิด -> ไม่นับ
            break
        gross = (exit_p - entry) * lot * contract if side == "BUY" else (entry - exit_p) * lot * contract
        cost = spread * lot * contract
        pnl = gross - cost
        risk_usd = lot * loss_per_lot
        eq += pnl
        peak = max(peak, eq)
        trades.append({
            "entry_time": times[t], "exit_time": times[exit_i], "side": side, "lot": lot,
            "entry": round(entry, 2), "exit": round(exit_p, 2), "sl_dist": round(sl_dist, 2),
            "reason": reason, "R": round(pnl / risk_usd, 3), "risk_usd": round(risk_usd, 2),
            "risk_pct_of_equity": round(risk_usd / (eq - pnl) * 100, 1),
            "pnl_usd": round(pnl, 2), "equity": round(eq, 2),
        })
        if eq <= 0:
            ruined = True
            break
        t = exit_i       # ไม้ถัดไปหาจากแท่งที่ปิดไม้ (บอทมีได้ทีละไม้)
    out = pd.DataFrame(trades)
    out.attrs["gated"] = gated
    return out, eq, skipped, ruined


def stats(tr, capital, eq_end, skipped, ruined, risk_pct):
    if tr.empty:
        return {"trades": 0, "end_equity": round(eq_end, 2), "skipped": skipped, "ruined": ruined}
    curve = np.concatenate([[capital], tr["equity"].values])
    peaks = np.maximum.accumulate(curve)
    dd = ((curve - peaks) / peaks * 100).min()
    wins = tr[tr["pnl_usd"] > 0]
    losses = tr[tr["pnl_usd"] <= 0]
    streak = best = 0
    for v in (tr["pnl_usd"] <= 0):
        streak = streak + 1 if v else 0
        best = max(best, streak)
    pf = wins["pnl_usd"].sum() / abs(losses["pnl_usd"].sum()) if len(losses) and losses["pnl_usd"].sum() != 0 else float("inf")
    return {
        "trades": len(tr), "win_rate_%": round(len(wins) / len(tr) * 100, 1),
        "total_R": round(tr["R"].sum(), 2), "avg_R": round(tr["R"].mean(), 3),
        "profit_factor": round(pf, 2),
        "end_equity": round(eq_end, 2), "return_%": round((eq_end / capital - 1) * 100, 1),
        "max_dd_%": round(dd, 1), "max_losing_streak": best,
        "avg_risk_%_of_equity": round(tr["risk_pct_of_equity"].mean(), 1),
        "trades_risk_over_%d%%" % (risk_pct * 1.25 // 1): int((tr["risk_pct_of_equity"] > risk_pct * 1.25).sum()),
        "skipped_cant_afford": skipped, "ruined": ruined,
    }


def print_stats(title, s):
    print(f"\n=== {title} ===")
    for k, v in s.items():
        print(f"  {k:28s} {v}")


# ---------------- MAIN ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="ไฟล์ราคา XAUUSD (ถี่กว่าหรือเท่ากับ --tf)")
    ap.add_argument("--tf", type=int, default=sig.BAR_HOURS, help="timeframe เป็นชั่วโมง (1,2,4,...)")
    ap.add_argument("--capital", type=float, default=CAPITAL_DEFAULT)
    ap.add_argument("--risk", type=float, default=sig.RISK_PCT, help="เสี่ยงต่อไม้ (%%)")
    ap.add_argument("--atr", type=float, default=sig.ATR_SL_MULT)
    ap.add_argument("--rr", type=float, default=sig.RR)
    ap.add_argument("--spread", type=float, default=sig.SPREAD_USD, help="สเปรด USD/oz ต่อไม้")
    ap.add_argument("--leverage", type=float, default=500)
    ap.add_argument("--gate", type=float, default=sig.MAX_RISK_PCT,
                    help="ข้ามสัญญาณถ้า risk จริงเกิน %% ของทุน (0 = ไม่จำกัด)")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--rolling", action="store_true",
                    help="เริ่มเทรดทีละหลายจุดเริ่ม (ทุก ~10 วัน) ถือ 6 เดือน แล้วสรุปการกระจายของผล")
    ap.add_argument("--out", default="backtest_out")
    a = ap.parse_args()

    raw = load_csv(a.csv, a.tf) if a.csv else load_td(tf=a.tf)
    df = sig.add_ind(raw.copy())
    print(f"ข้อมูล: {len(df)} แท่ง {a.tf}h  {df.index[0]} → {df.index[-1]}")
    if len(df) < 600:
        print("⚠️ ข้อมูลน้อย ผลไม่น่าเชื่อถือ (แนะนำ 2+ ปี)")

    sigs, _ = signal_series(df)
    bad, n = verify_equivalence(df, sigs)
    print(f"เช็กความตรงกับ signal() ของบอท: {n - bad}/{n} แท่งตรงกัน" + ("  ✅" if bad == 0 else "  ❌ ไม่ตรง!"))
    print(f"จำนวนแท่งที่มีสัญญาณ: {sum(s is not None for s in sigs)}")

    gate = a.gate if a.gate and a.gate > 0 else None
    kw = dict(capital=a.capital, risk_pct=a.risk, spread=a.spread, leverage=a.leverage, max_risk_pct=gate)
    os.makedirs(a.out, exist_ok=True)

    tr, eq, sk, ru = simulate(df, sigs, atr_mult=a.atr, rr=a.rr, **kw)
    s = stats(tr, a.capital, eq, sk, ru, a.risk)
    s["gated_by_risk_cap"] = tr.attrs.get("gated", 0)
    print_stats(f"ผลหลัก  ทุน {a.capital:g}$  risk {a.risk:g}%  SL={a.atr:g}xATR  RR 1:{a.rr:g}  "
                f"spread {a.spread}$  เพดาน risk {gate or 'ไม่มี'}%", s)
    if not tr.empty:
        tr.to_csv(os.path.join(a.out, "trades.csv"), index=False)
        tr["year"] = pd.to_datetime(tr["exit_time"]).dt.year
        yr = tr.groupby("year").agg(trades=("R", "size"), total_R=("R", "sum"), pnl_usd=("pnl_usd", "sum"))
        print("\nแยกรายปี:\n" + yr.round(2).to_string())
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(9, 4))
            plt.plot(pd.to_datetime(tr["exit_time"]), tr["equity"])
            plt.axhline(a.capital, color="grey", lw=0.8)
            plt.title(f"Equity (start {a.capital:g}$)")
            plt.tight_layout()
            plt.savefig(os.path.join(a.out, "equity.png"), dpi=120)
        except Exception:
            pass

    if a.rolling:
        bars_6m = int(6 * 30 * 24 * 5 / 7 / max(a.tf, 1) * 1) if a.tf >= 1 else 0
        step = max(1, int(10 * 24 * 5 / 7 / a.tf))
        res = []
        for st0 in range(250, len(df) - bars_6m, step):
            t3, e3, _, _ = simulate(df, sigs, atr_mult=a.atr, rr=a.rr, start=st0, end=st0 + bars_6m, **kw)
            res.append((df.index[st0].year, e3, len(t3)))
        r = pd.DataFrame(res, columns=["start_year", "end", "n"])
        print(f"\n=== ROLLING 6 เดือน: {len(r)} จุดเริ่ม (ซ้อนทับกัน -> จำนวนอิสระจริงน้อยกว่านี้มาก) ===")
        for name, sub in [("ทั้งหมด", r), ("เริ่มปี 2023-24", r[r.start_year <= 2024]),
                          ("เริ่มปี 2025+", r[r.start_year >= 2025])]:
            if len(sub):
                print(f"  {name:16s} N={len(sub):3d}  จบเหนือทุน {int((sub.end > a.capital).mean()*100):3d}%  "
                      f"ต่ำกว่าครึ่งทุน {int((sub.end < a.capital/2).mean()*100):3d}%  "
                      f"median {sub.end.median():.0f}  p25 {sub.end.quantile(.25):.0f}  p75 {sub.end.quantile(.75):.0f}  "
                      f"ไม่มีไม้เลย {int((sub.n == 0).mean()*100)}%")

    if a.sweep:
        rows = []
        for am in [1.5, 2.0, 2.5, 3.0]:
            for rr in [2.0, 3.0, 5.0]:
                t2, e2, k2, r2 = simulate(df, sigs, atr_mult=am, rr=rr, **kw)
                s2 = stats(t2, a.capital, e2, k2, r2, a.risk)
                rows.append({"SLxATR": am, "RR": rr, "trades": s2.get("trades"),
                             "win%": s2.get("win_rate_%"), "return%": s2.get("return_%"),
                             "maxDD%": s2.get("max_dd_%"), "avg_risk%": s2.get("avg_risk_%_of_equity"),
                             "skipped": s2.get("skipped_cant_afford"), "ruined": s2.get("ruined")})
        print("\n=== SWEEP ===\n" + pd.DataFrame(rows).to_string(index=False))

    print("\nหมายเหตุ: drawdown คิดจากยอดตอนปิดไม้ (ไม่รวมกำไร/ขาดทุนลอยตัวระหว่างถือ) "
          "และผลอดีตไม่การันตีอนาคต")


if __name__ == "__main__":
    main()
