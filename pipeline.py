# -*- coding: utf-8 -*-
"""全A股 LightGBM 研究基线：回测与指标。

信号在 t 日收盘后生成，下一交易日开盘买入，第 HOLD_DAYS 个交易日收盘退出。
这是研究用近似回测，不含 point-in-time 成分股、涨跌停/停牌成交限制或精确冲击成本。
想要“今天该买什么”的可执行清单，请运行 strategy.py。
"""
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from factors import FACTORS, load_prices, add_factors, add_labels, cross_sectionalize

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)
DATA_PATH = os.path.join(BASE, "data", "all_prices.parquet")

# 账户与成本假设；按小账户最低佣金计，费率需按本人券商核实。
INITIAL_CAPITAL = 6000.0
MAX_HOLDINGS = 4
BUY_COMMISSION_RATE = 0.00025
SELL_COMMISSION_RATE = 0.00025
MIN_COMMISSION = 5.0
SELL_STAMP_RATE = 0.0005  # 2023-08-28 起现行税率；本脚本对全测试期统一用此值，属近似
TRANSFER_RATE = 0.00001
BOARD_LOT = 100
HOLD_DAYS = 5
SPLIT_RATIO = 0.7

print("[1/6] 载入数据...", flush=True)
df = load_prices(DATA_PATH)
print("  rows:", len(df), "codes:", df.code.nunique(), flush=True)

print("[2/6] 计算仅使用信号日及之前数据的因子...", flush=True)
df = add_factors(df)

print("[3/6] 构造可交易时点对齐的标签...", flush=True)
df = add_labels(df, hold_days=HOLD_DAYS)
valid = (df["entry_open"] > 0) & (df["exit_close"] > 0) & df["entry_date"].notna() & df["exit_date"].notna()
data = df.loc[valid].dropna(subset=FACTORS + ["fwd_ret"]).copy()
# 每日截面缩尾与标准化；只用同一信号日截面，不跨日期拟合变换。
data = cross_sectionalize(data, FACTORS).dropna(subset=FACTORS)
# 截面标签仅用于训练目标；不作为特征。
data["target"] = data.groupby("date")["fwd_ret"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
data = data.dropna(subset=["target"])
print("  样本:", data.shape, "dates:", data.date.min().date(), "~", data.date.max().date(), flush=True)

print("[4/6] 按时间切分并 purge 标签重叠...", flush=True)
all_dates = np.sort(data["date"].unique())
split = int(len(all_dates) * SPLIT_RATIO)
# train 标签最远到 signal+HOLD_DAYS，因此留 HOLD_DAYS 日 purge，再额外 HOLD_DAYS 日 embargo。
train_end = max(0, split - HOLD_DAYS)
test_start = min(len(all_dates), split + HOLD_DAYS)
tr = data[data.date.isin(all_dates[:train_end])].copy()
te = data[data.date.isin(all_dates[test_start:])].copy()
if tr.empty or te.empty:
    raise ValueError("时序切分后 train/test 为空")
print("  train:", tr.shape, "test:", te.shape, flush=True)
print("  train signal end:", tr.date.max().date(), "test signal start:", te.date.min().date(), flush=True)

print("[5/6] LightGBM 训练...", flush=True)
params = dict(objective="regression", learning_rate=0.03, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=5, lambda_l2=1.0, verbose=-1, seed=42, num_threads=8)
model = lgb.train(params, lgb.Dataset(tr[FACTORS], tr["target"]), num_boost_round=400)
te["pred"] = model.predict(te[FACTORS])

print("[6/6] IC 与账户级近似回测...", flush=True)
ic_rows = []
for d, grp in te.groupby("date", sort=True):
    if len(grp) < 20:
        continue
    ic = np.corrcoef(grp["pred"], grp["target"])[0, 1]
    ric = spearmanr(grp["pred"], grp["target"]).correlation
    ic_rows.append((d, ic, ric, len(grp)))
ic_df = pd.DataFrame(ic_rows, columns=["date", "IC", "RankIC", "n"])
if ic_df.empty:
    raise ValueError("测试期没有足够截面样本计算 IC")
ics = ic_df["IC"].to_numpy()
print("IC mean %.4f | std %.4f | ICIR %.3f | positive %.1f%%" %
      (ics.mean(), ics.std(), ics.mean() / (ics.std() + 1e-9), (ics > 0).mean() * 100), flush=True)
print("RankIC mean %.4f" % ic_df["RankIC"].mean(), flush=True)

# 每 HOLD_DAYS 个信号日形成一个区间；次日开盘成交、区间末收盘退出。
# 以 6,000 元账户、至多 MAX_HOLDINGS 只、100 股整手、最低佣金估算费用。
te = te.sort_values(["date", "pred"])
signal_dates = np.sort(te["date"].unique())
rebal = signal_dates[::HOLD_DAYS]
all_market_dates = np.sort(df["date"].unique())
capital = INITIAL_CAPITAL
fee_total = 0.0
period_rows = []
for i in range(len(rebal) - 1):
    signal_date, exit_signal_date = rebal[i], rebal[i + 1]
    signal = te[te.date == signal_date]
    selected = signal.nlargest(min(MAX_HOLDINGS, len(signal)), "pred")["code"].astype(str).tolist()
    future_dates = all_market_dates[all_market_dates > signal_date]
    if len(future_dates) == 0:
        break
    entry_date = future_dates[0]
    entry = df[(df.date == entry_date) & df.code.isin(selected)][["code", "open"]]
    exit_rows = df[(df.date == exit_signal_date) & df.code.isin(selected)][["code", "close"]]
    trades = entry.merge(exit_rows, on="code", how="inner")
    trades = trades[(trades.open > 0) & (trades.close > 0)]
    slot_budget = capital / max(1, len(selected))
    trades["shares"] = np.floor((slot_budget / (1 + BUY_COMMISSION_RATE)) / trades.open / BOARD_LOT) * BOARD_LOT
    trades = trades[trades.shares >= BOARD_LOT].copy()
    if trades.empty:
        period_rows.append((exit_signal_date, capital, 0.0, 0, 0.0))
        continue
    trades["buy_notional"] = trades.shares * trades.open
    trades["sell_notional"] = trades.shares * trades.close
    trades["buy_fee"] = np.maximum(MIN_COMMISSION, trades.buy_notional * BUY_COMMISSION_RATE) + trades.buy_notional * TRANSFER_RATE
    trades["sell_fee"] = (np.maximum(MIN_COMMISSION, trades.sell_notional * SELL_COMMISSION_RATE)
                          + trades.sell_notional * (SELL_STAMP_RATE + TRANSFER_RATE))
    cash_after = capital - trades.buy_notional.sum() - trades.buy_fee.sum()
    proceeds = trades.sell_notional.sum() - trades.sell_fee.sum()
    new_capital = max(0.0, cash_after + proceeds)
    fee = float(trades.buy_fee.sum() + trades.sell_fee.sum())
    fee_total += fee
    period_ret = new_capital / capital - 1 if capital > 0 else 0.0
    capital = new_capital
    period_rows.append((exit_signal_date, capital, period_ret, len(trades), fee))

if not period_rows:
    raise ValueError("测试期未产生任何可执行回测区间")
period_df = pd.DataFrame(period_rows, columns=["date", "equity", "return", "filled_names", "fees"])
nav_df = pd.concat([
    pd.DataFrame({"date": [pd.to_datetime(rebal[0])], "equity": [INITIAL_CAPITAL],
                  "return": [0.0], "filled_names": [0], "fees": [0.0]}),
    period_df
], ignore_index=True)
nav = nav_df["equity"].to_numpy()
rets = nav_df["return"].to_numpy()[1:]
span_years = (pd.Timestamp(nav_df.date.iloc[-1]) - pd.Timestamp(nav_df.date.iloc[0])).days / 365.25
ann = (nav[-1] / INITIAL_CAPITAL) ** (1 / span_years) - 1 if span_years > 0 and nav[-1] > 0 else np.nan
max_dd = (nav / np.maximum.accumulate(nav) - 1).min()
periods_per_year = len(rets) / span_years if span_years > 0 else np.nan
sharpe = (rets.mean() / rets.std(ddof=1) * np.sqrt(periods_per_year)
          if len(rets) > 1 and rets.std(ddof=1) > 0 and periods_per_year > 0 else np.nan)
print("\n=== 账户级近似回测 (6000 RMB, max 10 names, board lots) ===")
print("Ending equity: %.2f | total return: %.2f%%" % (nav[-1], (nav[-1] / INITIAL_CAPITAL - 1) * 100))
print("Annualized: %.2f%% | max drawdown: %.2f%% | Sharpe: %.2f" % (ann * 100, max_dd * 100, sharpe))
print("Intervals: %d | mean filled names: %.2f | cumulative fees: %.2f RMB" %
      (len(period_df), period_df.filled_names.mean(), fee_total))
print("CAUTION: still excludes point-in-time/delisted universe, limit-up/down execution, suspensions, slippage and intraday fills.")

period_df["year"] = pd.to_datetime(period_df.date).dt.year
annual = period_df.groupby("year")["return"].agg(lambda x: (1 + x).prod() - 1).rename("return")
print("\nAnnual returns (%):")
print((annual * 100).round(2).to_string())

nav_df.to_csv(os.path.join(OUT, "nav.csv"), index=False)
ic_df.to_csv(os.path.join(OUT, "ic_series.csv"), index=False)
period_df.drop(columns="year", errors="ignore").to_csv(os.path.join(OUT, "period_returns.csv"), index=False)
plt.figure(figsize=(11, 4.5))
plt.plot(pd.to_datetime(nav_df.date), nav_df.equity / INITIAL_CAPITAL, lw=2, color="#c0392b")
plt.axhline(1, ls="--", c="gray", lw=0.8)
plt.title("A-share LightGBM research backtest (5d, 6,000 RMB assumptions)")
plt.ylabel("NAV"); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUT, "nav_full.png"), dpi=110)
imp = pd.Series(model.feature_importance("gain"), index=FACTORS).sort_values(ascending=False)
imp.to_csv(os.path.join(OUT, "feature_importance.csv"), header=["gain"])
print("Saved nav.csv, period_returns.csv, ic_series.csv, feature_importance.csv, nav_full.png")

# ── 最终输出选股策略 ────────────────────────────────────────────────────────
# 回测阶段使用严格留出的 te；生成策略时则用全部已知标签的历史样本重训，
# 再对数据集中最新可用截面打分。若行情不是近期数据，报告会标为过期而非下单指令。
print("\n[7/7] 全历史重训并生成最新可用选股策略...", flush=True)
live_train = data.dropna(subset=FACTORS + ["target"]).copy()
live_model = lgb.train(params, lgb.Dataset(live_train[FACTORS], live_train["target"]),
                        num_boost_round=400)
latest_counts = df.dropna(subset=FACTORS).groupby("date")["code"].count()
latest_usable = latest_counts[latest_counts >= 1000]
if latest_usable.empty:
    raise ValueError("无满足最小截面要求的最新信号日，无法生成策略")
signal_date = latest_usable.index.max()
signal = df[df.date == signal_date].dropna(subset=FACTORS).copy()
signal = cross_sectionalize(signal, FACTORS).dropna(subset=FACTORS)
signal["pred"] = live_model.predict(signal[FACTORS])

# 账户级整手分配：等分目标槽位，买不起一手的候选跳过并由下一名次补位。
slot_budget = INITIAL_CAPITAL / MAX_HOLDINGS
picks = []
for _, row in signal.nlargest(len(signal), "pred").iterrows():
    price = float(row["close"])
    if not np.isfinite(price) or price <= 0:
        continue
    shares = int(np.floor(slot_budget / (1 + BUY_COMMISSION_RATE) / price / BOARD_LOT) * BOARD_LOT)
    if shares < BOARD_LOT:
        continue
    amount = shares * price
    buy_fee = max(MIN_COMMISSION, amount * BUY_COMMISSION_RATE) + amount * TRANSFER_RATE
    picks.append({"rank": len(picks) + 1, "code": str(row["code"]), "signal_date": signal_date,
                  "reference_close": price, "prediction_score": float(row["pred"]),
                  "shares": shares, "estimated_amount": amount,
                  "estimated_buy_fee": buy_fee, "weight_pct": amount / INITIAL_CAPITAL * 100})
    if len(picks) >= MAX_HOLDINGS:
        break
if not picks:
    raise ValueError("整手资金约束下没有可买候选，无法生成策略")

strategy_df = pd.DataFrame(picks)
total_amount = float(strategy_df["estimated_amount"].sum())
total_buy_fees = float(strategy_df["estimated_buy_fee"].sum())
cash_left = INITIAL_CAPITAL - total_amount - total_buy_fees
live_model.save_model(os.path.join(OUT, "model.txt"))
with open(os.path.join(OUT, "model_meta.json"), "w", encoding="utf-8") as f:
    json.dump({"factors": FACTORS, "signal_date": str(pd.Timestamp(signal_date).date()),
               "hold_days": HOLD_DAYS, "capital": INITIAL_CAPITAL,
               "max_holdings": MAX_HOLDINGS, "training_rows": int(len(live_train)),
               "num_boost_round": 400, "params": params}, f, ensure_ascii=False, indent=2)
strategy_df.to_csv(os.path.join(OUT, "target_portfolio.csv"), index=False)

age_days = (pd.Timestamp.now().normalize() - pd.Timestamp(signal_date).normalize()).days
freshness = "CURRENT/RECENT" if age_days <= 7 else "STALE — DO NOT USE AS A CURRENT ORDER LIST"
lines = [
    "# A-share ML strategy candidate", "",
    "Signal date: %s | data age: %d calendar days | status: %s" %
    (pd.Timestamp(signal_date).date(), age_days, freshness),
    "Capital assumption: %.0f RMB | target names: %d | holding horizon: %d trading days" %
    (INITIAL_CAPITAL, MAX_HOLDINGS, HOLD_DAYS), "",
    "## Rules",
    "- Rank the latest eligible cross-section by the full-history LightGBM prediction.",
    "- Allocate approximately equal slot budgets; round down to 100-share lots; skip unaffordable candidates.",
    "- Research timing assumption: signal after close, entry next trading-day open, exit after %d trading days at close." % HOLD_DAYS,
    "- Reference close is not a guaranteed fill price. Recalculate shares using actual available cash and opening prices.", "",
    "## Candidate target portfolio",
    "| Rank | Code | Signal close (RMB) | Score | Shares | Estimated amount (RMB) | Weight |",
    "|---:|---|---:|---:|---:|---:|---:|"
]
for _, r in strategy_df.iterrows():
    lines.append("| %d | %s | %.2f | %.5f | %d | %.2f | %.2f%% |" %
                 (r["rank"], r["code"], r["reference_close"], r["prediction_score"],
                  r["shares"], r["estimated_amount"], r["weight_pct"]))
lines += ["", "Estimated buy fees: %.2f RMB; estimated remaining cash: %.2f RMB." %
          (total_buy_fees, cash_left), "",
          "## Important limitations",
          "- This is a model-generated research candidate list, not personalized financial advice or an order instruction.",
          "- The dataset uses the currently listed universe and has survivorship bias; delisted stocks are absent.",
          "- Backtest and signal generation do not model limit-up/down fills, suspensions, slippage, or exact broker fees.",
          "- If status is STALE, refresh market data and rerun before considering any trade.", ""]
strategy_path = os.path.join(OUT, "strategy.md")
with open(strategy_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("Signal date:", pd.Timestamp(signal_date).date(), "| age days:", age_days, "|", freshness, flush=True)
print("Target portfolio:")
print(strategy_df[["rank", "code", "reference_close", "prediction_score", "shares", "estimated_amount"]].to_string(index=False))
print("Strategy files: out/strategy.md, out/target_portfolio.csv, out/model.txt, out/model_meta.json")
