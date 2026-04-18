"""
Fund Flow Fetcher — GitHub Actions edition
ดึงข้อมูล ETF AUM จาก yfinance แล้วเขียน data/latest.json
รันทุกวันโดย GitHub Actions หลัง US market close
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
import numpy as np

# ── ETF universe ──────────────────────────────────────────────────────────────
UNIVERSE = {
    "country": {
        "South Korea":   {"etfs": ["EWY", "FLKR"],        "flag": "🇰🇷"},
        "Japan":         {"etfs": ["EWJ", "DXJ"],          "flag": "🇯🇵"},
        "India":         {"etfs": ["INDA", "EPI"],         "flag": "🇮🇳"},
        "Taiwan":        {"etfs": ["EWT"],                 "flag": "🇹🇼"},
        "Germany":       {"etfs": ["EWG"],                 "flag": "🇩🇪"},
        "Vietnam":       {"etfs": ["VNM"],                 "flag": "🇻🇳"},
        "Brazil":        {"etfs": ["EWZ"],                 "flag": "🇧🇷"},
        "Saudi Arabia":  {"etfs": ["KSA"],                 "flag": "🇸🇦"},
    },
    "sector": {
        "Technology":    {"etfs": ["XLK", "VGT"],          "flag": "💻"},
        "Defense":       {"etfs": ["XAR", "ITA"],          "flag": "🛡"},
        "Energy":        {"etfs": ["XLE", "VDE"],          "flag": "⚡"},
        "Nat. Resources":{"etfs": ["XME", "GNR"],          "flag": "⛏"},
        "Healthcare":    {"etfs": ["XLV", "VHT"],          "flag": "🏥"},
        "Financials":    {"etfs": ["XLF", "VFH"],          "flag": "🏦"},
        "Industrials":   {"etfs": ["XLI", "VIS"],          "flag": "🏭"},
        "Real Estate":   {"etfs": ["XLRE", "VNQ"],         "flag": "🏢"},
    },
    "asset": {
        "US Equity":     {"etfs": ["SPY", "IVV", "VTI"],  "flag": "📈"},
        "Intl Equity":   {"etfs": ["VEA", "EFA"],         "flag": "🌍"},
        "EM Equity":     {"etfs": ["EEM", "VWO", "IEMG"], "flag": "🌏"},
        "Bond":          {"etfs": ["AGG", "BND", "IEF"],  "flag": "📄"},
        "Gold":          {"etfs": ["GLD", "IAU"],          "flag": "🥇"},
        "Bitcoin ETF":   {"etfs": ["IBIT", "FBTC"],       "flag": "₿"},
        "Money Market":  {"etfs": ["SHV", "BIL"],         "flag": "💵"},
    },
    "theme": {
        "AI / Tech":     {"etfs": ["BOTZ", "AIQ", "CHAT"],"flag": "🤖"},
        "Robotics":      {"etfs": ["BOTT", "ROBO"],       "flag": "🦾"},
        "Clean Energy":  {"etfs": ["ICLN", "QCLN"],       "flag": "🌱"},
        "Cybersecurity": {"etfs": ["HACK", "CIBR"],       "flag": "🔐"},
        "Fintech":       {"etfs": ["FINX", "ARKF"],       "flag": "💳"},
    },
}

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def estimate_daily_flow(ticker: str, period_days: int = 90) -> dict | None:
    """ประมาณ net flow จาก ETF โดยใช้ ΔAUM − price effect"""
    try:
        tk = yf.Ticker(ticker)
        end = datetime.today()
        start = end - timedelta(days=period_days + 15)
        hist = tk.history(start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"),
                          auto_adjust=True)
        if hist.empty or len(hist) < 5:
            return None

        info = tk.fast_info
        shares = getattr(info, "shares", None)
        if not shares:
            return None

        close = hist["Close"]
        price_ret = close.pct_change().fillna(0)
        aum_b = close * shares / 1e9  # USD billion

        # Flow = ΔAUM − price_return × AUM_prev
        delta_aum = aum_b.diff().fillna(0)
        price_effect = price_ret * aum_b.shift(1).fillna(aum_b.iloc[0])
        flow = delta_aum - price_effect

        recent = flow.iloc[-period_days:]
        weekly = recent.resample("W").sum()

        return {
            "total_flow_bn":    round(float(recent.sum()), 2),
            "flow_1w_bn":       round(float(recent.iloc[-5:].sum()), 2),
            "flow_4w_bn":       round(float(recent.iloc[-20:].sum()), 2),
            "weekly_series":    [round(float(v), 2) for v in weekly.tail(12).tolist()],
            "latest_price":     round(float(close.iloc[-1]), 2),
            "price_ret_1m_pct": round(float((close.iloc[-1]/close.iloc[-21]-1)*100), 1),
            "current_aum_bn":   round(float(aum_b.iloc[-1]), 1),
        }
    except Exception as e:
        print(f"  warn {ticker}: {e}", file=sys.stderr)
        return None


def compute_surge_score(flows: list[dict]) -> float:
    """
    Surge score 0–100 รวม 3 มิติ:
      40% — ขนาด flow สัมพัทธ์ (flow_4w / AUM)
      35% — momentum (flow_1w vs flow_4w ต่อสัปดาห์)
      25% — price confirmation (price return ไปทิศเดียวกับ flow)
    """
    if not flows:
        return 0.0

    f4w  = np.mean([f["flow_4w_bn"] for f in flows])
    f1w  = np.mean([f["flow_1w_bn"] for f in flows])
    aum  = np.mean([f["current_aum_bn"] for f in flows]) or 1
    pret = np.mean([f["price_ret_1m_pct"] for f in flows])

    # Relative flow intensity (normalised, capped at 1)
    intensity = min(abs(f4w) / (aum * 0.05 + 0.01), 1.0) * np.sign(f4w)

    # Momentum: acceleration vs 4-week average weekly pace
    avg_weekly_pace = f4w / 4
    momentum = np.clip((f1w - avg_weekly_pace) / (abs(avg_weekly_pace) + 0.01), -1, 1)

    # Price confirmation: flow and price same direction?
    confirmation = 1.0 if (f4w > 0 and pret > 0) or (f4w < 0 and pret < 0) else -0.5

    raw = (intensity * 40 + momentum * 35 + confirmation * 25)
    return round(float(np.clip(50 + raw, 0, 100)), 1)


def classify(score: float, prev_score: float) -> str:
    delta = score - prev_score
    if score >= 65 and delta >= 0:  return "surge"
    if score >= 50 and delta >= 0:  return "watch"
    if delta <= -6:                 return "exit"
    return "neutral"


def build_output() -> dict:
    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "universe": {},
        "alerts": [],
        "summary": {"surge": 0, "watch": 0, "exit": 0, "neutral": 0},
    }

    # Load previous scores for delta/classify
    prev_path = DATA_DIR / "previous_scores.json"
    prev_scores: dict = {}
    if prev_path.exists():
        prev_scores = json.loads(prev_path.read_text())

    new_scores: dict = {}

    for cat, markets in UNIVERSE.items():
        output["universe"][cat] = {}
        for name, meta in markets.items():
            print(f"  {name} ({', '.join(meta['etfs'])})")
            flows = []
            used_etfs = []
            for ticker in meta["etfs"]:
                result = estimate_daily_flow(ticker)
                if result:
                    flows.append(result)
                    used_etfs.append(ticker)

            if not flows:
                continue

            score = compute_surge_score(flows)
            prev  = prev_scores.get(name, score)
            cls   = classify(score, prev)
            new_scores[name] = score

            # Aggregate across ETFs in group
            total_flow = round(sum(f["total_flow_bn"] for f in flows), 2)
            flow_1w    = round(sum(f["flow_1w_bn"] for f in flows), 2)
            flow_4w    = round(sum(f["flow_4w_bn"] for f in flows), 2)
            price_ret  = round(np.mean([f["price_ret_1m_pct"] for f in flows]), 1)

            # Merge weekly series (sum across ETFs, align length)
            series_len = min(len(f["weekly_series"]) for f in flows)
            weekly = [
                round(sum(f["weekly_series"][i] for f in flows), 2)
                for i in range(series_len)
            ]

            entry = {
                "name":          name,
                "category":      cat,
                "flag":          meta["flag"],
                "etfs":          used_etfs,
                "score":         score,
                "score_prev":    prev,
                "score_delta":   round(score - prev, 1),
                "classification":cls,
                "total_flow_bn": total_flow,
                "flow_1w_bn":    flow_1w,
                "flow_4w_bn":    flow_4w,
                "price_ret_1m":  price_ret,
                "weekly_series": weekly,
            }
            output["universe"][cat][name] = entry
            output["summary"][cls] += 1

            # Alerts
            if cls == "surge" and prev_scores.get(name, 0) < 65:
                output["alerts"].append({
                    "type":    "new_surge",
                    "market":  name,
                    "score":   score,
                    "flow_1w": flow_1w,
                    "etf":     used_etfs[0] if used_etfs else "",
                })
            elif cls == "exit":
                output["alerts"].append({
                    "type":   "exit_signal",
                    "market": name,
                    "score":  score,
                    "delta":  round(score - prev, 1),
                })

    # Save current scores as next run's "previous"
    prev_path.write_text(json.dumps(new_scores, indent=2))

    return output


def main():
    print("Fund Flow Fetcher — GitHub Actions edition")
    print(f"Started: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n")

    data = build_output()

    # Write latest.json (dashboard reads this)
    out = DATA_DIR / "latest.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"\nWrote {out} — {len(str(data))} chars")

    # Also write dated snapshot
    dated = DATA_DIR / f"snapshot_{datetime.utcnow().strftime('%Y%m%d')}.json"
    dated.write_text(json.dumps(data, indent=2))
    print(f"Wrote {dated}")

    # Summary
    s = data["summary"]
    print(f"\nSummary: {s['surge']} surge | {s['watch']} watch | {s['exit']} exit | {s['neutral']} neutral")
    if data["alerts"]:
        print(f"Alerts:  {len(data['alerts'])} new")
        for a in data["alerts"]:
            print(f"  [{a['type']}] {a['market']} — score {a['score']}")


if __name__ == "__main__":
    main()
