# -*- coding: utf-8 -*-
"""共享的行情载入、因子与标签定义。

pipeline.py（回测）与 strategy.py（实盘信号）都从这里导入，保证两边因子定义完全一致。
任何因子改动只需改这一个文件。
"""
import numpy as np
import pandas as pd

# 因子顺序即模型输入顺序，改动后必须重新训练模型。
FACTORS = ["mom5", "mom10", "mom20", "mom60", "vol20", "vol60", "ma5r", "ma20r",
           "ma60r", "amt_ma20_log", "volr", "hl", "close_pos", "vwap_close_ratio"]

REQUIRED_COLUMNS = {"date", "code", "open", "high", "low", "close", "volume", "amount"}


def load_prices(path):
    """载入行情并规范化 date / code，去重排序。"""
    df = pd.read_parquet(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError("数据缺少列: %s" % sorted(missing))
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df.sort_values(["code", "date"]).drop_duplicates(["code", "date"]).reset_index(drop=True)
    return df


def add_factors(df):
    """只用信号日及之前的数据计算 14 个因子；原地修改并返回。"""
    g = df.groupby("code", sort=False)
    df["ret1"] = g["close"].pct_change()
    df["mom5"] = g["close"].pct_change(5)
    df["mom10"] = g["close"].pct_change(10)
    df["mom20"] = g["close"].pct_change(20)
    df["mom60"] = g["close"].pct_change(60)
    df["vol20"] = g["ret1"].transform(lambda s: s.rolling(20, min_periods=20).std())
    df["vol60"] = g["ret1"].transform(lambda s: s.rolling(60, min_periods=60).std())
    df["ma5r"] = df["close"] / g["close"].transform(lambda s: s.rolling(5, min_periods=5).mean()) - 1
    df["ma20r"] = df["close"] / g["close"].transform(lambda s: s.rolling(20, min_periods=20).mean()) - 1
    df["ma60r"] = df["close"] / g["close"].transform(lambda s: s.rolling(60, min_periods=60).mean()) - 1
    df["amt_ma20_log"] = np.log1p(
        g["amount"].transform(lambda s: s.rolling(20, min_periods=20).mean()).clip(lower=0))
    df["volr"] = df["volume"] / g["volume"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()).replace(0, np.nan)
    df["hl"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["close_pos"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    df["vwap_close_ratio"] = df["amount"] / (df["volume"] * df["close"]).replace(0, np.nan)
    for f in FACTORS:
        df[f] = df[f].replace([np.inf, -np.inf], np.nan)
    return df


def add_labels(df, hold_days=5):
    """signal=t 收盘后；entry=t+1 开盘；exit=t+hold_days 收盘。"""
    g = df.groupby("code", sort=False)
    df["entry_open"] = g["open"].shift(-1)
    df["exit_close"] = g["close"].shift(-hold_days)
    df["fwd_ret"] = df["exit_close"] / df["entry_open"] - 1
    df["entry_date"] = g["date"].shift(-1)
    df["exit_date"] = g["date"].shift(-hold_days)
    return df


def cross_sectionalize(data, factors):
    """每日截面 1%/99% 缩尾后标准化；只用同一信号日截面，不跨日期拟合。"""
    for f in factors:
        lo = data.groupby("date")[f].transform(lambda s: s.quantile(0.01))
        hi = data.groupby("date")[f].transform(lambda s: s.quantile(0.99))
        data[f] = data[f].clip(lower=lo, upper=hi)
    mu = data.groupby("date")[factors].transform("mean")
    sd = data.groupby("date")[factors].transform("std").replace(0, np.nan)
    data[factors] = (data[factors] - mu) / sd
    return data
