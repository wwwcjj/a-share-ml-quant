# -*- coding: utf-8 -*-
"""全A股10年日线 (新浪源 + 断点续传) -> data/all_prices.parquet"""
import os, json, time, random, warnings
warnings.filterwarnings("ignore")
import akshare as ak
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data"); os.makedirs(OUT, exist_ok=True)
CKPT = os.path.join(OUT, "_ckpt_all.json")
BATCH = os.path.join(OUT, "_batch.parquet")
START, END = "20150101", "20251001"

print("[1/4] 获取全A股清单...", flush=True)
lst = ak.stock_info_a_code_name()
lst.columns = [c.strip() for c in lst.columns]
code_col = "code" if "code" in lst.columns else lst.columns[0]
codes = [str(c).zfill(6) for c in lst[code_col].tolist()]
print("  总数:", len(codes), flush=True)

def sina_symbol(c):
    return ("sh" if c.startswith(("6","5","9")) else "sz") + c

start_i, frames, ok, fail = 0, [], 0, 0
if os.path.exists(CKPT):
    try:
        ck = json.load(open(CKPT))
        start_i, ok, fail = ck["i"], ck["ok"], ck["fail"]
        print("  断点续传 from", start_i, flush=True)
    except Exception: pass

for i in range(start_i, len(codes)):
    c = codes[i]
    for attempt in range(2):
        try:
            df = ak.stock_zh_a_daily(symbol=sina_symbol(c), start_date=START,
                                     end_date=END, adjust="qfq")
            if df is not None and len(df) >= 100:
                df = df.reset_index()
                df.columns = [str(x).lower() for x in df.columns]
                dcol = "date" if "date" in df.columns else df.columns[0]
                df = df.rename(columns={dcol: "date"})
                df["code"] = c
                keep = ["date","open","high","low","close","volume","amount","code"]
                keep = [k for k in keep if k in df.columns]
                frames.append(df[keep])
                ok += 1
                break
        except Exception:
            time.sleep(1.2)
    else:
        fail += 1
    if (i+1) % 100 == 0:
        print("  %d/%d ok=%d fail=%d" % (i+1, len(codes), ok, fail), flush=True)
        if frames:
            try:
                pd.concat(frames, ignore_index=True).to_parquet(BATCH, index=False)
            except Exception: pass
        json.dump({"i": i+1, "ok": ok, "fail": fail}, open(CKPT,"w"))
    time.sleep(0.12 + random.random()*0.1)

print("[2/4] 合并...", flush=True)
if not frames:
    print("ERROR: no data"); raise SystemExit(1)
all_df = pd.concat(frames, ignore_index=True)
all_df["date"] = pd.to_datetime(all_df["date"])
all_df = all_df.drop_duplicates(["date","code"]).sort_values(["date","code"]).reset_index(drop=True)
p = os.path.join(OUT, "all_prices.parquet")
all_df.to_parquet(p, index=False)
print("[3/4] DONE shape=%s %s~%s 股票数=%d" % (all_df.shape, all_df.date.min().date(), all_df.date.max().date(), all_df.code.nunique()), flush=True)
print("[4/4] cleanup", flush=True)
for f in (CKPT, BATCH):
    if os.path.exists(f): os.remove(f)
