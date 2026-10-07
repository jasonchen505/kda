# KDA（Kernel Design Agents）深入学习文档

> 研读对象：`jasonchen505/kda`（fork 自 `NVlabs/kda`），研读时间 2026-10-07。
> 这是一份个人学习笔记，放在 fork 的 `notes/learning` 分支上，不进入 upstream PR。

---

## 1. 一句话定位

KDA 不是代码库，而是一个**工作流方法论文档仓库**——它回答的问题是：如何让 coding agent 可重复、可审计地完成"性能敏感型任务"（以 CUDA kernel 优化为代表，也可用于 compiler pass、runtime kernel、infra 改动）。

核心洞察：coding agent 的两大顽疾是**过程不可审计**和**失败经验随 session 丢失**。KDA 用"任务契约 + 证据记录规范"把 agent 的试错过程变成一棵显式的、可续跑的搜索树。

## 2. 核心方法论：九步循环（`docs/agent-flow.md`）

```
1. 定义任务契约（task contract）
2. agent 检查本地 workspace
3. agent 写 docs/draft.md（计划草稿）
4. 把 draft 转成可执行 plan
5. 实现第一个候选（candidate）
6. 验证正确性（validation）
7. 测量目标指标（evaluation）
8. 记录证据，决定 keep / revise / reject
9. 重复，直到达到 promotion 标准或 blocker 明确
```

### 2.1 任务契约（7 要素）

每个任务开始前必须写死：

| 要素 | 说明 |
|---|---|
| Objective | 用户可见的目标 |
| Inputs / outputs | 输入输出约定 |
| Correctness requirements | 正确性要求、容差、不变量 |
| Constraints | 实现语言、依赖、API、部署约束 |
| Validation command | **证明正确性**的命令 |
| Evaluation command | **测量目标指标**的命令（与验证不同时） |
| Promotion criteria | 候选被接受前必须为真的条件 |

最关键的设计是 **validation 和 evaluation 是两个独立命令**：前者回答"对不对"，后者回答"快不快"，agent 全程不许自创评判标准。这是对抗 reward hacking 的低成本高回报约束。

### 2.2 Promotion rule

只有同时满足以下两点，候选才能被 promotion：

1. 满足任务契约；
2. 有证据表明目标指标**提升或至少不退化**。

被 reject 的候选必须**记录原因**，不许静默丢弃。

### 2.3 Evidence 记录规范：为"考古"而设计

`docs/agent-flow.md` 明确说："格式不如一致性重要——未来的读者要能重建出：改了什么、测了什么、为什么选了这个候选。"

| 文件 | 用途 |
|---|---|
| `docs/draft.md` | 第一版计划草稿 |
| `docs/plan.md` | 可执行计划 |
| `benchmark.csv` | 表格化的测量结果 |
| `candidates.jsonl` | 候选名、**父子关系（parent links）**、状态 |
| `profile/` | profiler 输出或报告摘要 |
| `runs/` / `outputs/` | 生成产物 |

`candidates.jsonl` 的 parent links 本质上记录了一棵搜索树——哪个候选从哪个候选分叉、为什么被拒。这套文件规范几乎零成本，可以直接搬到任何 agent coding 任务里。

### 2.4 方法论与任务 workspace 的硬隔离

KDA 仓库只讲"怎么干"，具体任务的 harness、阈值、产物一律不许进仓库（`CLAUDE.md` + `.gitignore` 配合执行）。"流程资产 vs 任务资产"分离，是 agent workflow 能够复用的前提。

## 3. Prompt 模板（`prompts/`）

模板刻意保持 task-agnostic，使用前必须填入目标、约束、验证命令、promotion 标准；任务专属的 prompt 必须留在任务 workspace，不许污染本仓库（`prompts/README.md`）。

唯一的 starter prompt `prompts/basic-flow.md` 结构：

1. **人设**："produce the best correct implementation"；
2. **任务契约模板**：8 个 `<fill in>` 占位（与 agent-flow.md 的 7 要素一一对应，多了 task name）；
3. **Workflow 9 步**：读代码 → 找 baseline → 按需调研 → 写 draft → draft 转可执行 plan → 逐个实现候选 → 每次验证 → 记录证据 → 最终改动保持在契约范围内；
4. **Plan Draft Requirements**：draft.md 必须包含 baseline 及验证方式、主要风险、候选方向按期望价值/风险排序、第一步具体动作、确切的验证/评估命令、promotion 所需证据；
5. **硬性规定**：draft 不存在就不许开始写代码。

## 4. Skills（`skills/`，两个 submodule）

| Skill | 来源（pin） | 作用 |
|---|---|---|
| `skills/KernelWiki` | `mit-han-lab/KernelWiki` @ `76d27b5` | 领域知识库：89 个 artifact bundle，覆盖 CUTLASS（20）、FlashInfer（9）、PyTorch（5）、SGLang（10）、vLLM（20）、DeepGEMM（2）等上游项目的代码/文档摘录 + 12 个 KernelWiki 衍生 artifact + 11 个博客/比赛摘录 |
| `skills/ncu-report-skill` | `mit-han-lab/ncu-report-skill` @ `d1887948` | 教 agent 解读 NVIDIA Nsight Compute 报告（性能证据分析用） |

注意两点：
- README 明确警告：必须用本仓库 pin 的 submodule，**不要直接 checkout 上游**——上游可能含受额外条款约束的 artifact 快照；
- `THIRD_PARTY_NOTICES.md` 花了大篇幅做 license 合规：CUTLASS PR bundle 里 19 个带 NVIDIA 专有 SPDX 标识的文件被剔除不分发（另有 3 个混合 diff.patch 一并去掉，共 22 个），每个 bundle 保留 `PROVENANCE.yaml` 记录来源和剔除原因。

Getting Started 要求把这两个 skill 软链到 `~/.claude/skills/`——它们是**装给 agent 看的**，不是给人读的。

## 5. CLAUDE.md：给 agent 的仓库级指令

核心是"本仓库保持小而通用"：
- 英文写作；
- 任务专属的 prompt / 数据集 / validator / benchmark 日志一律不许进本仓库；
- MLSys 比赛是 KDA 的下游应用，不是本仓库范畴；
- 生成物放 `runs/`、`outputs/`、`profile/`（gitignore）；
- 优先记录可复用的 workflow 机制，而非某个任务的私有 harness。

Optional Skills 点名三个：`humanize`（plan 生成和实现循环）、领域知识 skill（背景调研）、profiling/report 分析 skill（性能证据）。

## 6. Community Kernel Wishlist

"我有个 kernel 需要优化" → **直接提 PR，无需先开 issue**：
- PR 目标分支必须是 **`wishlist`**（不是 main）；
- 请求目录：`requests/<github-username>-<kernel-name>/README.md`，写清项目影响、IO contract、正确性要求、workloads、baseline 来源/版本/license、环境、运行命令和实际验证结果；
- 附 `baseline.py`（当前最优实现），`benchmark.py` 可选；
- **目标硬件目前只支持 NVIDIA B200 和 B300**；
- PR 合并 = 请求被记录；优化进展和结果链接留在原 PR 上跟踪；
- 网站源码在 `pages` 分支，PR 模板和 issue chooser 配置在 `main` 分支维护。

## 7. 与 humanize 的关系

Getting Started 三步：① clone（含 submodules；注意 URL 还是旧的 `mit-han-lab/kernel-design-agents`，301 跳转到 `NVlabs/kda`，能用但已 stale）；② 软链 skills 到 `~/.claude/skills/`；③ 在 Claude Code 插件市场装 `humanize` 插件。

Minimal Flow 第 6 步明确写："把 draft 转成可执行 plan，手动或用 Humanize 这样的 planning 工具"。定位很清楚：

> **humanize 是执行编排层（把计划变成可运行的 agent loop），KDA 是领域方法论层。**

## 8. 给外部贡献者的规则（`CONTRIBUTING.md`）

- 欢迎外部贡献，走 issue + PR；**DCO sign-off 强制**（`git commit -s`，无 sign-off 的 commit 不收），用 DCO 而非 CLA；
- 对 KDA 本体的实质改动**先开 issue 讨论 scope**；改动保持聚焦、英文、附验证证据；
- License 混合：文档/prompts/skills 类内容 CC-BY-4.0，源码 Apache-2.0；第三方材料保留原 license 并在 `THIRD_PARTY_NOTICES.md` 登记。

## 9. 值得借鉴的三点

1. **"契约先行 + 双命令分离"**：validation（正确性）与 evaluation（性能）命令在任务开始前写死，agent 不得自创评判标准——防 reward hacking 的最小有效约束，适用于任何"agent 写代码"场景。
2. **Evidence 即搜索树**：`candidates.jsonl`（parent links）+ `benchmark.csv` + reject 必写原因，把试错过程变成可审计、可续跑的显式记录。
3. **流程资产与任务资产硬隔离**：方法论仓库不碰任何任务私有细节，这是 workflow 可复用的前提。

## 10. 局限与前提（诚实清单）

1. **强依赖可自动化验证**：循环成立的前提是有"一键验证正确性 + 一键测量指标"的 harness。CUDA kernel 有这个条件；UI、模糊需求等任务没有，promotion rule 就无从谈起。
2. **硬件门槛**：wishlist 只认 B200/B300；本地验证需要 GPU + Nsight Compute。没有高端卡只能学方法论，跑不完整循环。
3. **成本与信任**：鼓励"多候选 + 反复测量"，token 和 GPU 机时开销大；且 `benchmark.csv` 的数字是 agent 产出的，"证据"需要人抽查，否则是精致的自证。
4. **早期原型**：README 自称 early research prototype，方法论有效性主要靠 MLSys Kernel Contest 名次背书（`mit-han-lab/mlsys2026-flashinfer-contest`），通用任务上的证据还不多。

---

*文件索引：`README.md`、`docs/agent-flow.md`、`prompts/basic-flow.md`、`prompts/README.md`、`CLAUDE.md`、`CONTRIBUTING.md`、`THIRD_PARTY_NOTICES.md`、`third_party_licenses/`、`.github/PULL_REQUEST_TEMPLATE/wishlist.md`*
