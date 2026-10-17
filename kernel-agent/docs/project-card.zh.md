# 本地大模型 Kernel Agent Harness

面向大模型算法／研究工程师，研究 agent 的 kernel 生成、修复与验收。

- 参考 KDA，在4090／本地Qwen上搭建真实 Humanize SDK 多阶段 agent：独立草案／计划、模型选工具、证据审查、独立GPU验收。没有修改 Humanize 上游核心。
- 将完整源码记忆、部分源码事实、SHA行号补丁、真实分词预算、三路Qwen调度和最终数据隔离迁入新路线；236项工程检查通过。
- 多阶段修正轮九个结局：固定3/3、基础2/3、经验增强2/3，调用18／41／40次；无性能晋升，普通修复仍默认。保留首轮接口缺陷，不宣称完整KDA复现或多阶段普遍胜出。
- 历史行号补丁已知失败13/14，对照重写9/14、文本补丁7/14；近期新生成三组均6/8。这些历史成绩不与新增移植实验合并。

[最新移植结果](staged-results.zh.md) · [简历与面试](project-summary.zh.md) ·
[历史流程比较](workflow-comparison.zh.md) · [执行与审计](staged-reproduction.md)。
NCU硬件计数器不可用，实际CUDA活动追踪明确降级。原始数据留远端；Mac／Git只保留代码、文档、派生汇总和最新HTML。
