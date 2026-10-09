# -*- coding: utf-8 -*-
"""因子 + LightGBM + 5日持仓回测"""
import os, numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(BASE, "data", "prices.parquet"))
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["code", "date"]).reset_index(drop=True)

# ---------- 因子 ----------
g = df.groupby("code", group_keys=False)
df["ret1"]  = g["close"].pct_change(1)
df["mom5"]  = g["close"].pct_change(5)
df["mom10"] = g["close"].pct_change(10)
df["mom20"] = g["close"].pct_change(20)
df["vol20"] = g["ret1"].transform(lambda s: s.rolling(20).std())
df["ma5r"]  = df["close"] / g["close"].transform(lambda s: s.rolling(5).mean()) - 1
df["ma20r"] = df["close"] / g["close"].transform(lambda s: s.rolling(20).mean()) - 1
df["turn_ma5"] = g["turnover"].transform(lambda s: s.rolling(5).mean())
df["amt_ma5"]  = g["amount"].transform(lambda s: s.rolling(5).mean())
df["amp_ma5"]  = g["amplitude"].transform(lambda s: s.rolling(5).mean())
df["hl"] = (df["high"] - df["low"]) / df["close"]
df["close_pos"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-9)

FACTORS = ["mom5","mom10","mom20","vol20","ma5r","ma20r",
           "turn_ma5","amt_ma5","amp_ma5","hl","close_pos"]

# ---------- 标签: 未来5日收益 ----------
df["fwd5"] = g["close"].transform(lambda s: s.shift(-5) / s - 1)

# 截面标准化
def cs_norm(x):
    return (x - x.mean()) / (x.std() + 1e-9)
for f in FACTORS:
    df[f] = df.groupby("date")[f].transform(cs_norm)
df["fwd5_cs"] = df.groupby("date")["fwd5"].transform(cs_norm)

data = df.dropna(subset=FACTORS + ["fwd5"]).copy()
print("样本:", data.shape, "| 日期:", data.date.min().date(), "~", data.date.max().date())

# ---------- 时序切分 (70/30 + 5日embargo) ----------
dates = np.sort(data["date"].unique())
split_i = int(len(dates) * 0.7)
train_dates = dates[:split_i]
test_dates  = dates[split_i + 5:]   # embargo 5 日
tr = data[data.date.isin(train_dates)]
te = data[data.date.isin(test_dates)]
print("训练:", tr.shape, "测试:", te.shape)

# ---------- LightGBM ----------
params = dict(objective="regression", learning_rate=0.03, num_leaves=31,
              min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=5, verbose=-1, seed=42)
model = lgb.train(params, lgb.Dataset(tr[FACTORS], tr["fwd5_cs"]),
                  num_boost_round=300)
te = te.copy()
te["pred"] = model.predict(te[FACTORS])

# ---------- IC / RankIC ----------
ics, rics = [], []
for d, grp in te.groupby("date"):
    if len(grp) < 5: continue
    ics.append(np.corrcoef(grp["pred"], grp["fwd5_cs"])[0,1])
    rics.append(spearmanr(grp["pred"], grp["fwd5_cs"]).correlation)
print("IC  均值 %.4f  标准差 %.4f  ICIR %.2f" % (np.mean(ics), np.std(ics), np.mean(ics)/(np.std(ics)+1e-9)))
print("RankIC 均值 %.4f" % np.mean(rics))

# ---------- 回测: 每5日调仓, 买预测Top20% ----------
FEE = 0.001   # 单边0.1%
te = te.sort_values(["date","pred"])
rebal_days = np.sort(te["date"].unique())[::5]
nav, nav_dates, turnover_list = [1.0], [], []
prev_hold = set()
for i in range(len(rebal_days)-1):
    d0, d1 = rebal_days[i], rebal_days[i+1]
    snap = te[te.date == d0]
    k = max(1, int(len(snap) * 0.2))
    hold = set(snap.nlargest(k, "pred")["code"])
    # 期间收益
    nxt = te[(te.date > d0) & (te.date <= d1)]
    rets = nxt[nxt.code.isin(hold)].groupby("code").apply(
        lambda s: s.sort_values("date")["close"].iloc[-1] / s.sort_values("date")["close"].iloc[0] - 1)
    port_ret = rets.mean() if len(rets) else 0.0
    # 换手成本
    to = len(hold - prev_hold) / max(1, len(hold))
    port_ret -= to * FEE * 2
    nav.append(nav[-1] * (1 + port_ret))
    nav_dates.append(d1); turnover_list.append(to)
    prev_hold = hold

nav = np.array(nav)
total = nav[-1] - 1
years = (pd.Timestamp(nav_dates[-1]) - pd.Timestamp(nav_dates[0])).days / 365.25
ann = (nav[-1]) ** (1/years) - 1 if years > 0 else 0
dd = (nav / np.maximum.accumulate(nav) - 1).min()
sharpe = (np.diff(nav)/nav[:-1]).mean() / (np.diff(nav)/nav[:-1]).std() * np.sqrt(52/ (len(nav_dates)/years) ) if years>0 else 0
print("\n===== 回测结果 (5日调仓, 扣0.1%单边) =====")
print("总收益   %.2f%%" % (total*100))
print("年化收益 %.2f%%" % (ann*100))
print("最大回撤 %.2f%%" % (dd*100))
print("平均换手 %.2f%%" % (np.mean(turnover_list)*100))
print("调仓次数 %d" % len(nav_dates))

# ---------- 画图 ----------
plt.figure(figsize=(10,4))
plt.plot(pd.to_datetime(nav_dates), nav, lw=2, color="#c0392b")
plt.axhline(1, ls="--", c="gray", lw=0.8)
plt.title("LightGBM 5-Day Holding Strategy (fee-adjusted)")
plt.ylabel("NAV"); plt.grid(alpha=0.3)
out = os.path.join(BASE, "out", "nav_curve.png")
plt.tight_layout(); plt.savefig(out, dpi=110); print("图:", out)

# 特征重要性
imp = pd.Series(model.feature_importance("gain"), index=FACTORS).sort_values(ascending=False)
print("\n特征重要性(gain):"); print(imp.to_string())
