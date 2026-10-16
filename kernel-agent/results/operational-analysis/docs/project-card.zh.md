# 求职项目：本地大模型 Kernel Agent Harness

面向大模型算法／研究工程师。复现 KDA/Humanize，改进写 kernel 的 agent。

- 实现可信搜索探针、错误分流、带 SHA 的事务化源码补丁、拒绝后保留源码与独立验收。
  所有候选变更由本地 Qwen 提出，控制器没有人工 kernel 修补。
- 最新同预算对照：22 配对块、44 结局、96 调用。回归补丁 9/14、重写 7/14；
  新增四类任务分别 6/8、7/8。旧 SiLU／部分转置救回，但行均值与补丁协议成本退步。
- 17/42 补丁拒绝形成下一轮可检验假设。默认仍用普通修复，不夸大泛化收益；
  上轮三组均无收益及更早 13/16 对 15/16 的退步全部保留。
- 三路 Qwen 动态空闲准入与 EWMA 调度，物理 GPU4 共卡串行验收，持久化预算、
  未知调用不重放、真实 SIGKILL 恢复。191 项工程测试及原始证据可复核。

入口：`reports/kernel-agent-report.html#operational-study`；
`docs/project-summary.zh.md`；`python3 scripts/verify_evidence.py`。

这是可写简历的 agent harness 研究工程原型。新任务由项目定义、样本小，尚未
证明外部基准泛化或独占 GPU 性能收益。
