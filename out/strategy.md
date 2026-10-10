# A-share ML strategy candidate

Signal date: 2025-09-30 | data age: 374 calendar days | status: STALE — DO NOT USE AS A CURRENT ORDER LIST
Capital assumption: 6000 RMB | target names: 10 | holding horizon: 5 trading days

## Rules
- Rank the latest eligible cross-section by the full-history LightGBM prediction.
- Allocate approximately equal slot budgets; round down to 100-share lots; skip unaffordable candidates.
- Research timing assumption: signal after close, entry next trading-day open, exit after 5 trading days at close.
- Reference close is not a guaranteed fill price. Recalculate shares using actual available cash and opening prices.

## Candidate target portfolio
| Rank | Code | Signal close (RMB) | Score | Shares | Estimated amount (RMB) | Weight |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 300152 | 2.21 | 0.17315 | 200 | 442.00 | 7.37% |
| 2 | 300159 | 2.25 | 0.15600 | 200 | 450.00 | 7.50% |
| 3 | 600165 | 2.70 | 0.12135 | 200 | 540.00 | 9.00% |
| 4 | 002305 | 2.04 | 0.09724 | 200 | 408.00 | 6.80% |
| 5 | 300205 | 4.61 | 0.08463 | 100 | 461.00 | 7.68% |
| 6 | 002168 | 4.18 | 0.08311 | 100 | 418.00 | 6.97% |
| 7 | 002005 | 2.38 | 0.07245 | 200 | 476.00 | 7.93% |
| 8 | 300167 | 5.70 | 0.07151 | 100 | 570.00 | 9.50% |
| 9 | 603377 | 3.29 | 0.07060 | 100 | 329.00 | 5.48% |
| 10 | 000909 | 5.61 | 0.06600 | 100 | 561.00 | 9.35% |

Estimated buy fees: 50.05 RMB; estimated remaining cash: 1294.95 RMB.

## Important limitations
- This is a model-generated research candidate list, not personalized financial advice or an order instruction.
- The dataset uses the currently listed universe and has survivorship bias; delisted stocks are absent.
- Backtest and signal generation do not model limit-up/down fills, suspensions, slippage, or exact broker fees.
- If status is STALE, refresh market data and rerun before considering any trade.
