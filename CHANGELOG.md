# Changelog

## 0.1.0 (unreleased)

- SYNTHETIC local-vs-shared disturbance model with an explicit payoff table.
- Three-information-condition experiment (no / private / shared regime information) on common seeded disturbances, with hand-coded baseline policies.
- Metrics: coordination rate, miscoordination rate, welfare, miscoordination cost and regret, with 95% CIs over 200 seeds plus exact expectations.
- Exact unilateral-deviation audit of each baseline in the ex-ante strategy game, via `finite-game-equilibria` (pinned).
- JSON CLI, Markdown table renderer, committed results reproduced byte-for-byte in CI (Python 3.10–3.13).
