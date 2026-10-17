# KDA 与本项目：流程、贡献和面试边界

2026-10-17 收口审阅的历史比较。下文“当前实现”指当时的固定控制器路线。
用户随后要求搭建 4090 移植版，新增了真实 Humanize SDK 的多阶段路线，见
[移植结果与经验迁移](staged-results.zh.md)和[执行说明](staged-reproduction.md)。
这次新增工作不改变旧实验、旧比较所依据的事实，也没有完成原比赛流程的等价复现。

**项目定位：参考 KDA 工作流，早期接入 Humanize/OpenCode，构建本地 Qwen 的
Kernel Agent Harness。贡献位于下游生成、修复、独立评测、调度恢复和实验审计。
没有完整复现 KDA 比赛研究流程，没有修改 Humanize 上游核心。**

## 先区分两个 KDA

1. **通用参考仓库**，固定 `ef6ce617`：规定任务合约、调查、草案、可执行计划、
   候选实现、验证、测量和证据迭代。它不是带统一任务集、预算和评测器的单一可执行
   agent；具体工具与停止规则需要下游落实。
2. **MLSys 比赛公开流程**，本次核对 `d61573ff`：在上述思想上使用任务分阶段提示、
   Humanize 规划／RLCR、KernelWiki、Nsight Compute，以及 FlashInfer 官方工作负载。
   这是更接近“完整研究流程”的公开实例。公开的是流程、提示和环境说明；完整中间
   候选、搜索 DAG、benchmark 历史和 profiling 轨迹未发布，不能承诺重放原搜索路径。

通用仓库并未要求所有任务都必须跑三个性能阶段。比赛说明中的阶段也有任务差异，
不能把某一个任务的规则说成全体 KDA 的固定算法。

## KDA 完整公开流程怎样走

```text
独立任务工作区 + 合约 + 官方基线／验证命令
    ↓
Phase 1：调查任务、工作负载、已有实现和文档 → 首个正确 kernel
    ↓
Phase 2：对最佳正确版本 profiling → 瓶颈假设 → 多方向探索与验证
    ↓
Phase 3：继续探索／按工作负载形状分组专门化 → 完整验证与交付
```

**每阶段内部**：给阶段提示 → agent 调查 → 先保存 `docs/draft.md` →
`/humanize:gen-plan` 生成详细计划 → `/humanize:start-rlcr-loop` 实现与审查 →
验证、测量、记录候选父子关系与 NCU 证据 → 保留／修改／拒绝。
技术报告的算法是人、Claude 和 Codex 共同制定带逐项验收条件的计划，Claude 在真实
仓库使用工具实施，Codex 依据代码、测试和 profiler 证据独立审查；未通过则继续改进。
因此最关键的差别是**工具和探索路径由 agent 决定，审查逐项推进**。当前项目把这些
操作的顺序交给确定性控制器，模型负责源码提案。这是有意收窄的工程设计，不能用
“都存在生成和反馈”来称两者方法等价。RLCR 循环也不表示本项目训练了模型。
用户可以明确提高目标并重复后两个阶段。这里描述的是发布流程；没有声称本项目运行过
这两个 Humanize 插件命令，也没有把当前 Humanize Python SDK 等同于当时的插件。

Phase 3 的总说明强调形状分析／专门化；具体 GDN decode 提示则要求在 Phase 2
最佳版本上再做一轮探索。该任务还规定每个方向最多五次迭代、先测八个代表工作负载、
重大改进后测完整 54 个。这个例子说明“完整”包含明确实验方向、阶段交付物和性能证据，
不只是把 draft、plan、code 三个字段都返回。

## 收口审阅时的完整流程

有两条已实现路线，不能混称同一次执行：

- **早期 Humanize 路线**：任务文本 → KDA 适配提示 → Humanize `Agent.spawn/run`
  → OpenCode → 本地 Qwen → 结构化提案 → 下游控制器和独立 GPU 评测器。
  agent 被要求不自行调用工具、执行命令或修改文件。
- **近期对照路线**：实验计划冻结 → 三路本地 Qwen 空闲准入／EWMA 调度 → 原生兼容
  Chat 接口 → 结构化提案 → 控制器和独立 GPU 评测器。近期源码反馈与预算对照走这条
  路线，没有经过 Humanize/OpenCode。后端变化本身没有做成公平因果对照。

近期实验的实际闭环：

```text
固定任务合约、种子、代码与调用预算
  → 分配服务并在派发前记账
  → 首次生成一个候选（draft / plan / code 同一次回复）
  → 源码协议、SHA、基础 AST / 已安装 API 检查
  → 独立 GPU 搜索正确性、有限语义探针和计时
  → 反馈给模型，在原预算内最多两次修复
  → 从搜索候选中按性能门槛优先选择，提交所选 SHA
  → 独立最终形状／随机输入验收，只接受或否决，不反馈重选
  → 固化账本、原始证据与可审阅汇总
```

八族新生成对照确实有首次生成；最新预算／隔离诊断队列则从三个已筛选失败父源码
开始，只做修复，没有新任务生成。源码事实、行号补丁和预算适配是实验策略，普通修复
默认保留，不能说这些实验策略已经统一替换了所有入口。

## 逐步比较

| 环节 | KDA 比赛公开流程 | 本项目当前实现 | 判断 |
|---|---|---|---|
| 输入与基线 | FlashInfer 定义、实际 workload、任务指定 baseline | 自定义小型开发任务，参考实现／torch.compile | 任务与基线不同，成绩不横比 |
| 调查 | agent 主动读仓库、查 KernelWiki 与公开资料 | 控制器提供合约、规则卡和固定官方示例 | 有参考辅助，未实现自主调查 |
| 规划 | 先保存草案，再独立生成详细可执行计划 | 同一回复返回简短 draft、plan 和代码，再顺序落盘 | 字段和落盘顺序不能证明先规划后编码 |
| 执行／审查 | Humanize 插件计划与 RLCR 实现审查循环 | 早期 Humanize SDK；近期直接 Chat；验收由确定性程序裁决 | 没有完整 RLCR 复现；没有上游核心改造 |
| 优化探索 | 分阶段、多方向实验，可重复提高目标 | 固定机会数，完整重写或模型补丁修复 | 有闭环，但探索范围受控 |
| 性能诊断 | NCU 指标／源码归因支持瓶颈假设 | 正确性反例、部分源码事实与延迟测量 | 计时不是 profiler，AST 也不是语义证明 |
| 形状专门化 | 任务相关的 workload 分组与专门化 | 多形状验证通用候选，无独立专门化／分派阶段 | 尚未覆盖此阶段 |
| 选择与验收 | 官方验证、性能记录、候选谱系 | SHA 绑定、门槛优先选择、提交后独立最终否决 | 下游实现的具体机制，不声称原流程没有这些能力 |
| 预算与故障 | 公开阶段规则；中间运行账本未公开 | 持久化派发预算、未知调用不重放、任务／阶段恢复、策略故障隔离 | 是本项目可展示的工程贡献，未与完整 KDA 做优劣对照 |
| 证据与交付 | benchmark、候选关系、NCU、最终发布 | 服务／GPU 身份、模型请求、候选谱系、实验冻结、HTML | 原始证据留自有服务器；公开汇总不等于公开全部原始数据 |

早期 `kda_prompt` 是**单次提示适配基线**，反馈模式最多三轮，调用预算不同。
后来的 rewrite／indexed 等对照是在同一自建 harness 内比较策略。
这些结果都不支持“优于完整 KDA”或“复现了 KDA 比赛成绩”。

## 收口审阅时的决定与后续变化

**收口审阅当时不启动。** 求职主张收口为“可控的 agent harness 与可信实验”，已有代码、
真实模型轨迹、独立评测、对照与故障恢复支持这一主张。新增完整研究 agent 会扩展项目，
不是修正当前表述所必需。这个决定不意味着已经证明当前流程和完整 KDA 等价。

平台审计：服务器 GPU 为 RTX 4090，当前评测环境为 Torch 2.6 / Triton 3.2；
公开比赛重跑说明使用 B200、CUDA 13.2、Torch 2.12、Triton 3.6 和 FlashInfer workload。
`ncu` 可执行文件存在，但没有在本次验证采集权限或实际 profiling。不能据此声称缺少
profiler，也不能把 4090 上改写任务的试跑标成完整比赛复现。

若以后确实需要回答“独立规划／工具探索是否值得”，可另立**移植版工作流对照**：
两组同模型、任务、参考资料、搜索／最终验证边界；一组固定预算修复，一组独立调查、
规划、执行、profiling 与审查；同时公布总 tokens、工具／GPU 时间、候选数、最终正确
和性能晋升。三路服务按配对块平衡，不能只对齐外层轮数；多阶段消耗和工具能力必须
单独报告。重新冻结实验前不把这个建议当作已完成实验。

用户随后明确授权了上述移植方向。新增路线使用三次独立阶段中的草案、计划、工具动作
和审查，并在真实 Humanize SDK 上运行；具体阶段是否通过、成本及最终正确性分别报告。
旧的只读平台审计仍为零模型／GPU调用，新增实验另立账本，没有写回旧审计。

真正比赛流程复现还需要匹配目标硬件、官方任务／依赖、独立起始工作区、实际规划／
审查与 profiler 轨迹，并避免读取公开最终解作为生成输入。单任务跑通只证明连通性。

## 面试和投递怎么讲

**项目名称**：本地大模型 Kernel Agent Harness：预算修复与可信评测。

**一分钟表述**：我参考 KDA 的任务合约和证据迭代方法，早期用 Humanize/OpenCode
打通本地 Qwen，后续为可控实验实现了直接模型接口。自己做的是下游控制器：将模型
源码交给独立 GPU 评测，用可信失败证据在固定预算内修复，按门槛选候选，最终检查只
验收不回传；还实现多服务调度、持久化预算和中断恢复。实验出现过退步与上下文故障，
我保留原结果，再单独验证修复。没有完整复现 KDA 比赛流程，也没有改造 Humanize 核心。

投递时突出三个可讲深的案例：门槛优先选择为何必要；补丁协议改善为什么不等于新任务
泛化；新增反馈如何挤爆上下文并破坏对照、又如何隔离恢复。具体实验数字及限定条件见
[简历与面试总结](project-summary.zh.md)。不再新增策略来追更好看的单个分数。

## 固定版本来源和代码入口

- [KDA 通用循环](https://github.com/NVlabs/kda/blob/ef6ce617693ef0782b3ecb9f37e39bbf10226a90/docs/agent-flow.md)、[基础提示](https://github.com/NVlabs/kda/blob/ef6ce617693ef0782b3ecb9f37e39bbf10226a90/prompts/basic-flow.md)。
- [比赛公共阶段定义与 Humanize 命令](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest/blob/d61573ffb2008b6e68fb32fe8e17703e4b2448c9/prompts/README.md)、[重跑边界](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest/blob/d61573ffb2008b6e68fb32fe8e17703e4b2448c9/docs/reproduction.md)。
- [技术报告 §3、Algorithm 1、Figure 3](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest/blob/d61573ffb2008b6e68fb32fe8e17703e4b2448c9/docs/HAN_Lab_Kernel_Mafia_Technical_Report.pdf)：自主工具执行、独立验收 agent、profiling 迭代与形状路由。
- [GDN decode Phase 1](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest/blob/d61573ffb2008b6e68fb32fe8e17703e4b2448c9/prompts/gdn-decode/phase1.md)、[Phase 2](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest/blob/d61573ffb2008b6e68fb32fe8e17703e4b2448c9/prompts/gdn-decode/phase2.md)、[Phase 3](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest/blob/d61573ffb2008b6e68fb32fe8e17703e4b2448c9/prompts/gdn-decode/phase3.md)。
- [早期真实 SDK 流程](../flows/kernel_loop.py)、[公共 API 兼容层](../src/kernel_agent/humanize_api.py)、[一次返回草案／计划／源码的提示](../src/kernel_agent/repair.py)、[最新诊断控制器](../scripts/run_grounded_budget_study.py)。
- [本次只读来源／平台审计](../results/workflow-audit.json)。固定来源原件与 PDF 提取保留远端归档；本次新模型调用 0、GPU 评测 0，不计入策略实验成绩。
