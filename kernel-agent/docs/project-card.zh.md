# 求职项目：本地大模型 Kernel Agent Harness

面向大模型算法／研究工程师，改进的是写 kernel 的 agent。

- 复现 KDA/Humanize，搭建独立评测、分类反馈、源码选择、预算和恢复；模型候选不经人工修补。
- 设计可信数值／坐标提示、SHA 行号补丁与预算内回退。同初稿三组对照，22 块／66 结局／
  140 调用：回归行号策略 13/14、重写 9/14、旧补丁 7/14；新生成三组均 6/8。
- 首补丁应用 13/21 → 18/21，补丁拒绝 17/43 → 7/40；按任务族报告区间，
  保留语义失败与旧负结果。默认仍用普通修复，尚未证明外部泛化收益。
- 三路 Qwen 空闲准入／EWMA 调度，GPU4 共卡串行验收，持久化预算与未知调用不重放。
  205 项检查通过；原始数据全部在远端，Git／Mac 只保留代码、文档、派生汇总和最新 HTML。

入口：`reports/kernel-agent-report.html#span-study`、`docs/project-summary.zh.md`。
本地摘要核验：`python3 scripts/verify_span_summary.py results/span-summary.json`；
完整原始审计与实验经远端 `kernel-agent-storage/bin/run.py` 运行。
