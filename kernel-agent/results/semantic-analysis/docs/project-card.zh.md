# 求职项目：本地大模型 Kernel Agent Harness

面向大模型算法／研究工程师。研究对象是写 kernel 的 agent，不以手写算子充当生成成绩。
复现 KDA/Humanize 路线，再用本地 Qwen 后端改进生成、审查与修复；候选源码不经人工改写。

## 贡献与实证

1. 可计算语义审查：受限 AST 核对参考式、候选式与两组完整数值探针，过滤未核验叙述，
   失败后事实回退；独立最终评测仍决定正确性。
2. 同初稿、同审查三组对照，22 块、66 个结局、115 次真实派发。
   回归：重写 7/14、原始审查 7/14、核验 7/14；
   新生成：7/8、7/8、7/8。分层与任务聚类区间均保留。
3. 三路 Qwen 动态发现、空闲准入、EWMA 调度；块内固定进程、GPU4 评测串行，预算
   持久化、未知调用不重放。176 项测试兼容两版 SDK 和新锁定安装。
4. 保留前一轮 13/16 对 15/16 的退步，定位漏因子、布局和虚构精度解释，形成新的
   研究假设。历史真实 SIGKILL 恢复以 0 次新生成调用完成最终验收。

默认普通生成＋普通修复，审查变体维持实验性。公式核验不保证 API／索引实现正确；
小规模回归和一个新种子不等于外部泛化验证。

## 审阅入口

- `reports/kernel-agent-report.html#semantic-study`：对照、表达式轨迹、数值矩阵和完整回复。
- `src/kernel_agent/semantic.py`、`model_pool.py`、`scripts/run_semantic_study.py`：核心机制。
- `python3 scripts/verify_evidence.py`：不用模型或 CUDA 的证据审阅。
- `docs/project-summary.zh.md`：可直接使用的简历和面试文字。
