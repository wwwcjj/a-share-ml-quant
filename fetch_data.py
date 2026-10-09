# -*- coding: utf-8 -*-
"""拉取沪深300成分股日线 -> data/prices.parquet（无缓冲+超时+重试）"""
import os, time, sys
import pandas as pd
import akshare as ak

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)

print("[1/3] 获取沪深300成分股...", flush=True)
hs300 = ak.index_stock_cons_csindex(symbol="000300")
codes = hs300["成分券代码"].astype(str).str.zfill(6).tolist()
print(f"  成分股 {len(codes)} 只", flush=True)

START, END = "20220101", "20251001"
rows, ok, fail = [], 0, 0
for i, code in enumerate(codes):
    got = None
    for attempt in range(2):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                    start_date=START, end_date=END, adjust="qfq")
            if df is not None and len(df) >= 100:
                got = df; break
        except Exception as e:
            time.sleep(0.3)
    if got is not None:
        got["code"] = code
        rows.append(got); ok += 1
    else:
        fail += 1
    if (i+1) % 20 == 0:
        print(f"  {i+1}/{len(codes)} ok={ok} fail={fail}", flush=True)
    time.sleep(0.1)

print(f"[2/3] 合并 {ok} 只股票...", flush=True)
all_df = pd.concat(rows, ignore_index=True)
all_df = all_df.rename(columns={
    "日期":"date","开盘":"open","收盘":"close","最高":"high",
    "最低":"low","成交量":"volume","成交额":"amount","换手率":"turnover",
    "振幅":"amplitude","涨跌幅":"pct_chg"})
all_df["date"] = pd.to_datetime(all_df["date"])
all_df = all_df.sort_values(["date","code"]).reset_index(drop=True)
p = os.path.join(OUT, "prices.parquet")
all_df.to_parquet(p, index=False)
print(f"[3/3] DONE {p} shape={all_df.shape} {all_df.date.min().date()}~{all_df.date.max().date()}", flush=True)
