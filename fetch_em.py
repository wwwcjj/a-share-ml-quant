# -*- coding: utf-8 -*-
"""沪深300日线抓取 -> data/prices.parquet (东财直连, 断点续传)"""
import os, json, time, random
import requests
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data"); os.makedirs(OUT, exist_ok=True)
CKPT = os.path.join(OUT, "_ckpt.json")

S = requests.Session()
S.trust_env = False
S.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"})

def get_json(url, params, tries=4):
    for k in range(tries):
        try:
            r = S.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(1.5 * (k + 1) + random.random())
    return None

def get_codes():
    out, pn = [], 1
    while True:
        p = {"pn": pn, "pz": 100, "po": 1, "np": 1, "fltt": 2, "invt": 2,
             "fid": "f12", "fs": "b:BK0500", "fields": "f12,f14"}
        j = get_json("https://push2.eastmoney.com/api/qt/clist/get", p)
        if not j or not j.get("data") or not j["data"].get("diff"):
            break
        diff = j["data"]["diff"]
        out += [(str(d["f12"]).zfill(6), d["f14"]) for d in diff]
        if len(diff) < 100:
            break
        pn += 1
        time.sleep(0.3)
    return out

def get_kline(code):
    secid = "1." + code if code.startswith(("6", "5", "9")) else "0." + code
    p = {"secid": secid, "ut": "fa5fd1943c7b386f172d6893dbfba10b",
         "fields1": "f1,f2,f3,f4,f5,f6",
         "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
         "klt": 101, "fqt": 1, "beg": "20220101", "end": "20251001"}
    j = get_json("https://push2his.eastmoney.com/api/qt/stock/kline/get", p)
    if not j or not j.get("data") or not j["data"].get("klines"):
        return None
    rows = []
    for line in j["data"]["klines"]:
        a = line.split(",")
        rows.append({"date": a[0], "open": float(a[1]), "close": float(a[2]),
                     "high": float(a[3]), "low": float(a[4]), "volume": float(a[5]),
                     "amount": float(a[6]), "amplitude": float(a[7]),
                     "pct_chg": float(a[8]), "chg": float(a[9]), "turnover": float(a[10]),
                     "code": code})
    return rows

print("[1/3] fetch HSV300 constituents...", flush=True)
codes = get_codes()
print("  constituents:", len(codes), flush=True)

start_i, rows, ok, fail = 0, [], 0, 0
if os.path.exists(CKPT):
    try:
        ck = json.load(open(CKPT))
        start_i, rows, ok, fail = ck["i"], ck["rows"], ck["ok"], ck["fail"]
        print("  resume from:", start_i, flush=True)
    except Exception:
        pass

for i in range(start_i, len(codes)):
    code, name = codes[i]
    rr = get_kline(code)
    if rr:
        rows += rr; ok += 1
    else:
        fail += 1
    if (i + 1) % 20 == 0:
        print("  %d/%d ok=%d fail=%d" % (i+1, len(codes), ok, fail), flush=True)
        json.dump({"i": i+1, "rows": rows, "ok": ok, "fail": fail}, open(CKPT, "w"))
    time.sleep(0.35 + random.random() * 0.25)

print("[2/3] merge", ok, flush=True)
df = pd.DataFrame(rows)
if len(df) == 0:
    print("ERROR: no data"); raise SystemExit(1)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "code"]).reset_index(drop=True)
p = os.path.join(OUT, "prices.parquet")
df.to_parquet(p, index=False)
print("[3/3] DONE shape=%s %s~%s" % (df.shape, df.date.min().date(), df.date.max().date()), flush=True)
if os.path.exists(CKPT): os.remove(CKPT)
