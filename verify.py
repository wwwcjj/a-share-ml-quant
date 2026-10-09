# -*- coding: utf-8 -*-
"""只读审计全A股行情与最近回测产物；不训练模型、不改数据。"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "all_prices.parquet")
OUT = os.path.join(BASE, "out")

print("=== Dataset audit ===")
cols = ["date", "code", "open", "high", "low", "close", "volume", "amount"]
df = pd.read_parquet(DATA, columns=cols)
df["date"] = pd.to_datetime(df["date"])
print("rows:", len(df), "codes:", df.code.nunique(),
      "dates:", df.date.min().date(), "~", df.date.max().date())
print("duplicate (date, code):", int(df.duplicated(["date", "code"]).sum()))
print("missing values:", int(df[cols].isna().sum().sum()))
invalid = ((df.low > df.high) | (df.low > df.open) | (df.low > df.close)
           | (df.high < df.open) | (df.high < df.close)
           | (df.close <= 0) | (df.open < 0) | (df.high < 0) | (df.low < 0)
           | (df.volume < 0) | (df.amount < 0))
print("invalid OHLC/negative values:", int(invalid.sum()))
if invalid.any():
    print(df.loc[invalid, cols].head(10).to_string(index=False))
print("codes by year:")
df["year"] = df.date.dt.year
print(df.groupby("year").code.nunique().to_string())
del df

print("\n=== Result artifact audit ===")
nav_path = os.path.join(OUT, "nav.csv")
if os.path.exists(nav_path):
    nav = pd.read_csv(nav_path, parse_dates=["date"])
    print("nav rows:", len(nav), "columns:", list(nav.columns))
    if {"date", "equity"}.issubset(nav.columns):
        print("equity min/max:", nav.equity.min(), nav.equity.max())
        print("duplicate dates:", int(nav.date.duplicated().sum()))
        print("nonpositive equity:", int((nav.equity <= 0).sum()))
    elif {"date", "nav"}.issubset(nav.columns):
        print("legacy nav.csv detected (old pipeline output); inspect/replace before interpreting")
else:
    print("nav.csv missing")
for name in ["period_returns.csv", "ic_series.csv", "feature_importance.csv", "nav_full.png"]:
    path = os.path.join(OUT, name)
    print(name + ":", "present" if os.path.exists(path) else "missing")
print("\nNote: this is a data/artifact audit, not proof of point-in-time universe or executable fills.")
