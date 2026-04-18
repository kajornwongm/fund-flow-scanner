"""
Fund Flow Backfill — ดึงข้อมูลย้อนหลังสูงสุด 2 ปี
====================================================
รันครั้งเดียวเพื่อสร้าง historical snapshots สำหรับ backtest

การใช้งาน:
    python backfill.py              # ดึงย้อนหลัง 2 ปี (default)
    python backfill.py --months 6   # ดึงย้อนหลัง 6 เดือน

Output:
    data/snapshot_YYYYMMDD.json     # snapshot รายเดือน
    data/backfill_summary.csv       # สรุป score + flow ทุก market ทุกเดือน
    data/latest.json                # อัปเดต latest ด้วย
"""

import json
import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
from dateutil.relativedelta import relativedelta
import numpy as np
import yfinance as yf

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

UNIVERSE = {
    "country": {
        "South Korea":    ["EWY", "FLKR"],
        "Japan":          ["EWJ", "DXJ"],
        "India":          ["INDA", "EPI"],
        "Taiwan":         ["EWT"],
        "Germany":        ["EWG"],
        "Vietnam":        ["VNM"],
        "Brazil":         ["EWZ"],
        "Saudi Arabia":   ["KSA"],
    },
    "sector": {
        "Technology":     ["XLK", "VGT"],
        "Defense":        ["XAR", "ITA"],
        "Energy":         ["XLE", "VDE"],
        "Nat. Resources": ["XME", "GNR"],
        "Healthcare":     ["XLV", "VHT"],
        "Financials":     ["XLF", "VFH"],
        "Industrials":    ["XLI", "VIS"],
        "Real Estate":    ["XLRE", "VNQ"],
    },
    "asset": {
        "US Equity":      ["SPY", "IVV"],
        "Intl Equity":    ["VEA", "EFA"],
        "EM Equity":      ["EEM", "VWO"],
        "Bond":           ["AGG", "BND"],
        "Gold":           ["GLD", "IAU"],
        "Bitcoin ETF":    ["IBIT", "FBTC"],
        "Money Market":   ["SHV", "BIL"],
    },
    "theme": {
        "AI / Tech":      ["BOTZ", "AIQ"],
        "Robotics":       ["ROBO"],
        "Clean Energy":   ["ICLN", "QCLN"],
        "Cybersecurity":  ["HACK", "CIBR"],
    },
}

# ── Cache ETF data ────────────────────────────────────────────────────────────
_cache = {}

def fetch_etf(ticker: str, start: str, end: str):
    key = (ticker, start, end)
    if key in _cache:
        return _cache[key]
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(start=start, end=end, auto_adjust=True)
        shares = tk.info.get("sharesOutstanding", None)
        if hist.empty or not shares or shares < 1000:
            _cache[key] = None
            return None
        result = {"hist": hist, "shares": shares}
        _cache[key] = result
        return result
    except Exception as e:
        print(f"    warn {ticker}: {e}")
        _cache[key] = None
        return None


def compute_flow_for_window(ticker: str, window_start: str, window_end: str) -> dict | None:
    """คำนวณ net flow ของ ETF ในช่วงเวลาที่กำหนด"""
    data = fetch_etf(ticker, window_start, window_end)
    if data is None:
        return None

    hist = data["hist"]
    shares = data["shares"]

    # กรองเฉพาะช่วง window
    mask = (hist.index >= window_start) & (hist.index <= window_end)
    h = hist[mask]
    if len(h) < 5:
        return None

    close = h["Close"]
    price_ret = close.pct_change().fillna(0)
    aum_b = close * shares / 1e9

    delta_aum = aum_b.diff().fillna(0)
    price_effect = price_ret * aum_b.shift(1).fillna(aum_b.iloc[0])
    flow = delta_aum - price_effect

    total_flow = float(flow.sum())
    price_ret_total = float((close.iloc[-1] / close.iloc[0] - 1) * 100)

    return {
        "total_flow_bn":    round(total_flow, 2),
        "flow_1w_bn":       round(float(flow.iloc[-5:].sum()), 2),
        "flow_4w_bn":       round(float(flow.iloc[-20:].sum()), 2),
        "price_ret_1m_pct": round(price_ret_total, 1),
        "current_aum_bn":   round(float(aum_b.iloc[-1]), 1),
        "latest_price":     round(float(close.iloc[-1]), 2),
        "weekly_series":    [round(float(v), 2) for v in flow.resample("W").sum().tail(12).tolist()],
    }


def compute_surge_score(flows: list[dict]) -> float:
    if not flows:
        return 0.0
    f4w  = np.mean([f["flow_4w_bn"] for f in flows])
    f1w  = np.mean([f["flow_1w_bn"] for f in flows])
    aum  = np.mean([f["current_aum_bn"] for f in flows]) or 1
    pret = np.mean([f["price_ret_1m_pct"] for f in flows])

    intensity    = min(abs(f4w) / (aum * 0.05 + 0.01), 1.0) * np.sign(f4w)
    avg_weekly   = f4w / 4
    momentum     = np.clip((f1w - avg_weekly) / (abs(avg_weekly) + 0.01), -1, 1)
    confirmation = 1.0 if (f4w > 0 and pret > 0) or (f4w < 0 and pret < 0) else -0.5

    raw = intensity * 40 + momentum * 35 + confirmation * 25
    return round(float(np.clip(50 + raw, 0, 100)), 1)


def classify(score: float, prev: float) -> str:
    d = score - prev
    if score >= 65 and d >= 0:  return "surge"
    if score >= 50 and d >= 0:  return "watch"
    if d <= -6:                  return "exit"
    return "neutral"


def build_snapshot(as_of_date: datetime, lookback_days: int = 90) -> dict:
    """สร้าง snapshot ณ วันที่ as_of_date"""
    end_str   = as_of_date.strftime("%Y-%m-%d")
    start_str = (as_of_date - timedelta(days=lookback_days + 15)).strftime("%Y-%m-%d")

    snapshot = {
        "generated_at": as_of_date.isoformat() + "Z",
        "as_of_date":   end_str,
        "universe":     {},
        "alerts":       [],
        "summary":      {"surge": 0, "watch": 0, "exit": 0, "neutral": 0},
    }

    for cat, markets in UNIVERSE.items():
        snapshot["universe"][cat] = {}
        for name, etfs in markets.items():
            flows = []
            used  = []
            for tk in etfs:
                f = compute_flow_for_window(tk, start_str, end_str)
                if f:
                    flows.append(f)
                    used.append(tk)

            if not flows:
                continue

            score    = compute_surge_score(flows)
            cls      = classify(score, score - 1)  # no prev for historical
            total    = round(sum(f["total_flow_bn"] for f in flows), 2)
            f4w      = round(sum(f["flow_4w_bn"] for f in flows), 2)
            f1w      = round(sum(f["flow_1w_bn"] for f in flows), 2)
            pret     = round(np.mean([f["price_ret_1m_pct"] for f in flows]), 1)

            entry = {
                "name":            name,
                "category":        cat,
                "etfs":            used,
                "score":           score,
                "classification":  cls,
                "total_flow_bn":   total,
                "flow_4w_bn":      f4w,
                "flow_1w_bn":      f1w,
                "price_ret_1m":    pret,
            }
            snapshot["universe"][cat][name] = entry
            snapshot["summary"][cls] += 1

    return snapshot


def main():
    parser = argparse.ArgumentParser(description="Fund Flow Backfill")
    parser.add_argument("--months", type=int, default=60,
                        help="จำนวนเดือนย้อนหลัง (default: 60)")
    args = parser.parse_args()

    today     = datetime.today().replace(day=1)  # ต้นเดือนปัจจุบัน
    csv_rows  = []
    snapshots = []

    print(f"Backfilling {args.months} months of fund flow data...")
    print(f"Markets: {sum(len(v) for v in UNIVERSE.values())} total\n")

    # ดึงข้อมูล ETF ทั้งหมดล่วงหน้าครั้งเดียว (เร็วกว่า loop ทีละเดือน)
    global_start = (today - relativedelta(months=args.months + 1)).strftime("%Y-%m-%d")
    global_end   = datetime.today().strftime("%Y-%m-%d")
    all_tickers  = [tk for markets in UNIVERSE.values() for tks in markets.values() for tk in tks]

    print(f"Pre-fetching {len(all_tickers)} ETFs ({global_start} → {global_end})...")
    for tk in all_tickers:
        fetch_etf(tk, global_start, global_end)
        print(f"  {tk}", end=" ", flush=True)
    print("\n")

    # สร้าง snapshot ทีละเดือน
    for i in range(args.months, 0, -1):
        as_of = today - relativedelta(months=i - 1)
        # ใช้วันสุดท้ายของเดือน
        last_day = (as_of + relativedelta(months=1) - timedelta(days=1))
        if last_day > datetime.today():
            last_day = datetime.today()

        date_str = last_day.strftime("%Y-%m-%d")
        print(f"[{args.months - i + 1}/{args.months}] Building snapshot: {date_str}")

        snap = build_snapshot(last_day, lookback_days=90)
        snapshots.append(snap)

        # บันทึก snapshot รายเดือน
        fname = DATA_DIR / f"snapshot_{last_day.strftime('%Y%m%d')}.json"
        fname.write_text(json.dumps(snap, indent=2))

        # เก็บแถว CSV
        for cat, markets in snap["universe"].items():
            for name, m in markets.items():
                csv_rows.append({
                    "date":          date_str,
                    "market":        name,
                    "category":      cat,
                    "score":         m["score"],
                    "classification":m["classification"],
                    "flow_4w_bn":    m["flow_4w_bn"],
                    "flow_1w_bn":    m["flow_1w_bn"],
                    "price_ret_1m":  m["price_ret_1m"],
                    "etfs":          ",".join(m["etfs"]),
                })

    # บันทึก summary CSV (สำหรับ backtest analysis)
    csv_path = DATA_DIR / "backfill_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    # อัปเดต latest.json ด้วย snapshot ล่าสุด
    latest = snapshots[-1]
    (DATA_DIR / "latest.json").write_text(json.dumps(latest, indent=2))

    print(f"\nDone!")
    print(f"  {len(snapshots)} monthly snapshots saved to data/")
    print(f"  {len(csv_rows)} rows in {csv_path}")
    print(f"  latest.json updated")
    print(f"\nNext step: commit & push the data/ folder to GitHub")
    print(f"  git add data/ && git commit -m 'backfill: {args.months}mo historical data' && git push")


if __name__ == "__main__":
    try:
        from dateutil.relativedelta import relativedelta
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dateutil"])
        from dateutil.relativedelta import relativedelta
    main()