# -*- coding: utf-8 -*-
"""输出可执行的投资策略：规则 + 目标持仓 + 下单股数。

用法:
    ../quant-env/bin/python strategy.py
    ../quant-env/bin/python strategy.py --capital 6000 --names 10 --hold 5
"""
import argparse, json, os
import numpy as np
import pandas as pd
import lightgbm as lgb
from factors import FACTORS, load_prices, add_factors, add_labels, cross_sectionalize

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")
DATA_PATH = os.path.join(BASE, "data", "all_prices.parquet")

BUY_COMMISSION_RATE = 0.00025
MIN_COMMISSION = 5.0
TRANSFER_RATE = 0.00001
BOARD_LOT = 100
MIN_UNIVERSE = 1000  # 信号日至少要有这么多只股票有因子，才算可用截面


def parse_args():
    p = argparse.ArgumentParser(description="生成当前 A 股 ML 选股策略")
    p.add_argument("--capital", type=float, default=6000.0, help="可用资金（元）")
    p.add_argument("--names", type=int, default=10, help="目标持仓只数")
    p.add_argument("--hold", type=int, default=5, help="持有交易日数")
    p.add_argument("--rounds", type=int, default=400, help="LightGBM 训练轮数")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("[1/5] 载入行情...", flush=True)
    df = load_prices(DATA_PATH)
    df = add_factors(df)
    df = add_labels(df, hold_days=args.hold)
    print("  rows:", len(df), "codes:", df.code.nunique(), flush=True)

    print("[2/5] 训练模型（用全部已有标签的历史数据）...", flush=True)
    tr = df.dropna(subset=FACTORS + ["fwd_ret"]).copy()
    tr = cross_sectionalize(tr, FACTORS).dropna(subset=FACTORS)
    tr["target"] = tr.groupby("date")["fwd_ret"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    tr = tr.dropna(subset=["target"])
    if tr.empty:
        raise ValueError("没有可训练样本")
    params = dict(objective="regression", learning_rate=0.03, num_leaves=63,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=5, lambda_l2=1.0, verbose=-1, seed=42, num_threads=8)
    model = lgb.train(params, lgb.Dataset(tr[FACTORS], tr["target"]), num_boost_round=args.rounds)
    print("  训练样本:", tr.shape, "最后可用标签日:", tr.date.max().date(), flush=True)

    print("[3/5] 选取最新可用信号日...", flush=True)
    counts = df.dropna(subset=FACTORS).groupby("date")["code"].count()
    usable = counts[counts >= MIN_UNIVERSE]
    if usable.empty:
        raise ValueError("没有任何交易日满足最小截面要求")
    signal_date = usable.index.max()
    snap = df[df.date == signal_date].dropna(subset=FACTORS).copy()
    snap = cross_sectionalize(snap, FACTORS)
    snap["pred"] = model.predict(snap[FACTORS])
    print("  信号日:", signal_date.date(), "可选股票:", len(snap), flush=True)

    print("[4/5] 生成目标持仓（整手约束下按排名补位）...", flush=True)
    slot = args.capital / max(1, args.names)
    picks = []
    for _, row in snap.nlargest(len(snap), "pred").iterrows():
        price = float(row["close"])
        if not np.isfinite(price) or price <= 0:
            continue
        shares = int(np.floor(slot / (1 + BUY_COMMISSION_RATE) / price / BOARD_LOT) * BOARD_LOT)
        if shares < BOARD_LOT:
            continue
        picks.append({"code": row["code"], "close": price, "score": float(row["pred"]),
                      "shares": shares, "cost": shares * price})
        if len(picks) >= args.names:
            break
    if not picks:
        raise ValueError("在整手与资金约束下没有可买标的，考虑增加资金或减少只数")
    port = pd.DataFrame(picks)
    port["weight_pct"] = port.cost / args.capital * 100
    port["est_buy_fee"] = np.maximum(MIN_COMMISSION, port.cost * BUY_COMMISSION_RATE) + port.cost * TRANSFER_RATE
    total_cost = float(port.cost.sum())
    total_fee = float(port.est_buy_fee.sum())
    cash_left = args.capital - total_cost - total_fee

    print("[5/5] 写出策略文件...", flush=True)
    model.save_model(os.path.join(OUT, "model.txt"))
    with open(os.path.join(OUT, "model_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"factors": FACTORS, "hold_days": args.hold, "names": args.names,
                   "capital": args.capital, "signal_date": str(signal_date.date()),
                   "train_rows": int(len(tr)), "params": params, "rounds": args.rounds},
                  fh, ensure_ascii=False, indent=2)
    port.to_csv(os.path.join(OUT, "target_portfolio.csv"), index=False)

    lines = []
    lines.append("# A 股 ML 选股策略")
    lines.append("")
    lines.append("信号日（收盘后生成）: " + str(signal_date.date()))
    lines.append("可用资金: %.0f 元   目标持仓: %d 只   持有: %d 个交易日" % (args.capital, args.names, args.hold))
    lines.append("")
    lines.append("## 策略规则")
    lines.append("1. 每个信号日收盘后，计算 %d 个量价因子并做当日截面缩尾与标准化。" % len(FACTORS))
    lines.append("2. 用 LightGBM 预测全市场未来 %d 个交易日的收益排名。" % args.hold)
    lines.append("3. 取预测分最高的若干只，等权分配资金，按 100 股整手向下取整。")
    lines.append("4. 下一交易日开盘买入；持有满 %d 个交易日后收盘卖出，然后重新出信号。" % args.hold)
    lines.append("5. 买不起一手的标的自动跳过，由下一名次补位。")
    lines.append("")
    lines.append("## 目标持仓")
    lines.append("| 排名 | 代码 | 参考价(收盘) | 预测分 | 股数 | 金额(元) | 权重 |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, row in port.reset_index(drop=True).iterrows():
        lines.append("| %d | %s | %.2f | %.4f | %d | %.0f | %.1f%% |" % (
            i + 1, row["code"], row["close"], row["score"], row["shares"], row["cost"], row["weight_pct"]))
    lines.append("")
    lines.append("合计买入金额 %.0f 元，预估买入费用 %.0f 元，剩余现金 %.0f 元。" % (total_cost, total_fee, cash_left))
    lines.append("")
    lines.append("## 执行步骤")
    lines.append("1. 下一个交易日开盘时，按上表股数逐只买入。")
    lines.append("2. 记录实际成交价；持有满 %d 个交易日后的收盘卖出。" % args.hold)
    lines.append("3. 卖出后重新运行本脚本，获取新一轮信号。")
    lines.append("")
    lines.append("## 风险提示")
    lines.append("- 股票池取自当前仍在交易的股票，含幸存者偏差。")
    lines.append("- 未模拟涨停买不进、跌停卖不出、停牌与滑点。")
    lines.append("- 回测收益为研究估算，不代表未来表现，不构成投资建议。")
    report = "\n".join(lines)
    with open(os.path.join(OUT, "strategy.md"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")

    print("")
    print(report)
    print("")
    print("已写出: out/strategy.md  out/target_portfolio.csv  out/model.txt  out/model_meta.json")


if __name__ == "__main__":
    main()
