# DreamMem：证据引导的建设性机制搜索与可验证记忆

本项目研究一个非因果问题：**Masked Diffusion Language Model 的去噪轨迹能否提供候选假设，再通过符号邻域构造和真实经历验收，形成可在同一 episode 内从支持经历泛化到新查询、可执行验证的离散概念？** 最终主张是证据引导的建设性机制搜索，而不是学习控制器或时间动力学必然有效。

“睡眠”在这里有严格的操作定义：离线阶段不接收新的外部数据，只对清醒阶段保存的经历反复掩码、去噪和修订，形成可持久保存的离散记忆。候选记忆必须通过未参与对应轨迹生成的真实经历片段验证，最终只在全新查询上评分。已完成消融显示经典贪心/束搜索可以超过学习控制器，删除轨迹特征不一定降低性能；完整结论和限制见 `PAPER.md`。

## 实验任务

每个环境有一个隐藏概念，概念是一到三个属性的合取，例如“红色且方形”。一条经历包含一个物体及它的正负标签。模型观察若干经历，同时预测隐藏概念和新物体标签。

- 训练：13 个单属性概念和 80% 双属性组合。
- 验证：10% 未见双属性组合，仅用于模型选择和记忆超参数调优。
- `seen`：训练概念的新经历。
- `unseen`：完全未见的双属性组合。
- `compositional`：训练中从未出现的三属性组合。

真实概念不作为模型输入。语义真值只用于数据生成和最终评价。

## 模型与轨迹

默认模型是 6.61M 参数的双向 Transformer，使用 absorbing-mask diffusion 训练。推理包含多个经历前缀阶段，每个阶段运行多条可修订去噪轨迹。低置信内容会被重新掩码，因此中间状态不只是单调揭示答案。

每个时间步记录概念槽位、查询预测和预测置信度。概念被解析为规范语义状态，修订分为：

- 保留；
- 概括；
- 特化；
- 替换；
- 完全替换；
- 形成；
- 删除；
- 非法状态。

## 记忆形成对照

所有轨迹方法使用相同经历、采样数、去噪步数和中间状态数量。

1. `final`：只读取最终概念状态。
2. `token_stability`：使用各槽位的 token 生存与频率信息。
3. `semantic_stability`：使用完整概念的驻留和跨时间稳定性。
4. `bag_of_states`：保留全部中间状态，但不使用其时间顺序。
5. `revision_no_merge`：只使用有序语义修订评分，不生成合并候选。
6. `merge_only`：允许候选合并，但不使用有序修订评分。
7. `revision_dynamics`：同时使用有序语义转移、真实经历一致性改善和修订产生的概括/合并候选。
8. `evidence_states`：只对轨迹中已有状态使用纯真实证据评分，不使用频率或置信度。
9. `evidence_guided`：对已有状态和修订产生的新候选使用纯真实证据评分，用于隔离新候选形成的贡献。
10. `symbolic`：直接枚举概念并使用真实经历的符号求解基线。

睡眠阶段最多使用 6 条支持经历，最后 2 条真实支持经历仅用于候选记忆验收。测试查询不参与记忆形成或调参。

## 运行

环境需要 Python 3.10+、PyTorch、NumPy、PyYAML、pandas、scikit-learn、matplotlib、Transformers 和 tiktoken，完整依赖见 `pyproject.toml`。可以直接从项目根目录运行，无需安装命令行入口。

检查配置：

```bash
python -m dreammem inspect --config configs/full.yaml
```

训练单个随机种子：

```bash
python -m dreammem train --config configs/full.yaml --seed 17
```

评估检查点：

```bash
python -m dreammem evaluate \
  --config configs/full.yaml \
  --checkpoint outputs/full/seed_17/best.pt
```

运行去噪步数、采样数和重新掩码概率消融：

```bash
python -m dreammem ablate \
  --config configs/full.yaml \
  --checkpoint outputs/full/seed_17/best.pt
```

运行完整多种子实验：

```bash
python -m dreammem full --config configs/full.yaml
```

已有检查点时：

```bash
python -m dreammem full --config configs/full.yaml --skip-training
```

## 输出

每个 `outputs/full/seed_<seed>/` 包含：

- `best.pt`、`latest.pt`：模型检查点；
- `history.csv`、`training.json`：训练历史与耗时；
- `evaluation/trajectories/`：各划分的完整轨迹；
- `tuning_history.csv`、`best_memory_configs.json`：仅基于验证集的超参数搜索；
- `results_<split>.csv`：逐 episode 结果；
- `summary.csv`：概念准确率、查询准确率、NLL、Brier、记忆错误和风险—覆盖指标；
- `intervals.csv`：episode bootstrap 区间；
- `concept_cluster_intervals.csv`：按真实概念聚类重采样的区间，避免把同一概念的重复 episode 当作完全独立证据；
- `paired_comparisons.csv`：修订动力学相对各基线的配对差值；
- `semantic_revision_probe.csv`：不同轨迹特征对未见概念正确性与未来查询效用的预测能力；
- `metrics.png`、`risk_coverage.png`：汇总图。

多种子结果写入 `aggregate_summary.csv`；探针、配对比较和轨迹消融分别聚合到 `aggregate_probe.csv`、`aggregate_paired_comparisons.csv` 和 `aggregate_ablations.csv`。

## 结果解释

有序修订方法必须在相同轨迹和计算量下同时超过 `final` 和 `bag_of_states`，才能支持“修订动力学提供额外信号”。只超过最终输出，不能排除多次采样；与无序状态袋相同，不能证明时间顺序有作用。`revision_no_merge` 和 `merge_only` 用于进一步分离有序评分与候选合并的作用。

## 组合泛化第二阶段

`configs/composition.yaml` 使用全新的 episode 随机种子，并将 16 个三元概念作为 `val_compositional`，其余 112 个作为独立测试。候选评分同时考虑形成期真实经历、保留期真实经历、复杂度、特异性和新候选准入惩罚；所有方法在调参时获得相同信息。验证目标同时奖励总体表现和最弱划分表现，避免靠放弃某一类组合提高平均分。运行方式：

```bash
python -m dreammem evaluate \
  --config configs/composition.yaml \
  --checkpoint outputs/full/seed_17/best.pt \
  --output outputs/composition/seed_17/evaluation
```

## 噪声稳健性实验

`configs/noisy.yaml` 在保持查询标签干净的同时，对 10% 真实支持经历施加标签翻转，并使用另一组全新 episode。该实验检验候选形成与证据验收在 imperfect experiences 下是否仍有效。噪声场景中的 `validation_error` 表示与含噪观察的不一致，不能直接解释为幻觉。

```bash
python -m dreammem evaluate \
  --config configs/noisy.yaml \
  --checkpoint outputs/full/seed_17/best.pt \
  --output outputs/noisy/seed_17/evaluation
```

噪声实验三种子结果：`evidence_guided` 在未见三元组合上取得 0.7214±0.0113 概念恢复和 0.9046±0.0059 查询准确率，符号基线分别为 0.6875 和 0.6826。`evidence_states` 在噪声未见双属性上取得 0.6380±0.0197 概念恢复和 0.8405±0.0098 查询准确率。噪声会造成明显下降，但新候选形成在组合任务上仍保持优势。

## 第二阶段组合确认结果

在全新 episode、16 个独立三元验证概念和 112 个三元测试概念上，三种子结果为：

| 方法 | 未见双属性概念 | 未见双属性查询 | 未见三元概念 | 未见三元查询 |
|---|---:|---:|---:|---:|
| evidence_states | **0.8398±0.0141** | **0.9173±0.0050** | 0.0677±0.0869 | 0.8714±0.0197 |
| evidence_guided | 0.7695±0.0039 | 0.8737±0.0028 | **0.9023±0.0039** | **0.9482±0.0026** |
| symbolic | 0.8359±0.0000 | 0.5596±0.0000 | **0.9531±0.0000** | 0.7109±0.0000 |

结果显示两种记忆操作具有互补性：已有状态的证据选择更适合未见双属性，修订生成的新候选更适合未见三元组合。候选形成不是此前瓶颈的唯一解释；只有加入真实形成期一致性并校准复杂度/特异性后，高覆盖率候选才能转化为正确记忆。符号方法的 MAP 概念更准确，但其候选后验较分散，因此查询准确率不能由概念准确率直接推出。

## 第一阶段实验结果

三个随机种子、每个划分 256 个 episode 的未见双属性组合结果如下，数值为均值±种子标准差：

| 方法 | 概念精确恢复 | 查询准确率 | 查询 NLL |
|---|---:|---:|---:|
| final | 0.1745±0.0553 | 0.7256±0.0306 | 1.2941 |
| bag_of_states | 0.2669±0.0222 | 0.7744±0.0216 | 0.4794 |
| revision_no_merge | 0.4518±0.0341 | **0.8210±0.0144** | **0.4255** |
| revision_dynamics | 0.4518±0.0341 | 0.8005±0.0147 | 0.4338 |
| symbolic | **0.6211±0.0000** | 0.7822±0.0000 | 0.4435 |

修订动力学探针的三种子平均 ROC AUC 为 0.9564，最终输出特征为 0.6839，无序状态袋为 0.8289。`revision_no_merge` 与完整方法的概念恢复相同且查询更好，`merge_only` 没有稳定优势，因此当前证据支持有序修订评分，不支持把候选合并当作主要贡献。

轨迹消融中，8 步配置以一半计算量取得较好的查询准确率；取消主动重新掩码提高了概念恢复但降低部分校准指标。这些属于测试划分上的探索性消融，不作为测试集选模依据。

符号基线概念恢复更强，三元组合精确恢复也仍然很弱。因此当前结果支持“轨迹包含可用于记忆形成的新增信息”，但不支持“dLLM 已优于符号归纳”或“已经完成自然语言长期记忆”。当前实验是合成组合概念上的完整研究环境，不直接证明自然语言记忆、长期权重巩固或真实梦境机制。

## 外部 0.5B dLLM 实测：Open-dCoder

本项目已下载并校验 [Open-dCoder-0.5B](https://huggingface.co/fredzzp/open-dcoder-0.5B)（[官方代码](https://github.com/pengzhangzhi/Open-dLLM)），权重 SHA-256 为 `59c1a005f4b672bdd3bbdab6258b283dff3b87bab5cc4bbc0d479e8388f77b6e`。加载器位于 `dreammem/dcoder.py`，兼容当前 Transformers 的文档装饰器和 SDPA 调用接口。

运行：

```bash
python scripts/dcoder_trial.py \
  --model models/open-dcoder-0.5B \
  --steps 16 --tokens 64 \
  --revision-probability 0.35 \
  --output outputs/dcoder/custom_remask.json
```

实测结果：

- 参数量 494,032,768；本地加载约 2–3 秒。
- 内置 16 步生成约 0.83–1.09 秒，峰值显存约 1.21–1.54GiB。
- 自定义修订循环约 0.84 秒，峰值约 1.24GiB。
- 内置生成器可返回历史，但当前 Hugging Face 版本的 `p2` 参数没有实现主动重掩码分支。
- 自定义循环能发起已提交 token 修订，但本次测试中模型预测保持一致，实际替换率为 0。这证明接口可接入，不证明原模型已经具备纠错能力。

因此，将本方法迁移到该模型需要针对结构化语义输出和纠正性修订进行后训练。单次代码生成文本不是本项目的性能基准，结果保存在 `outputs/dcoder/`。

## Open-dCoder 完整后训练

`configs/dcoder.yaml` 将相同的跨经历概念任务编码为单 token 属性槽位，在 0.5B 模型上执行真实掩码扩散后训练。训练使用 BF16、梯度检查点、梯度累积、混合支持阶段和验证集选模；原始模型保持不变，输出保存到独立目录。

```bash
python scripts/train_dcoder.py \
  --config configs/dcoder.yaml \
  --model models/open-dcoder-0.5B \
  --seed 17
```

完整轨迹与记忆评估：

```bash
python scripts/evaluate_dcoder.py \
  --config configs/dcoder.yaml \
  --model outputs/dcoder_training/seed_17/best \
  --output outputs/dcoder_evaluation/seed_17/evaluation
```

文本语义编码位于 `dreammem/dcoder_text.py`，训练位于 `dreammem/dcoder_train.py`，评估位于 `dreammem/dcoder_experiment.py`。语义槽位使用单 token 属性词和 `none` 空位；掩码 ID 使用模型生成配置，而非误用 tokenizer 中另一个特殊 token。长时间评估会记录批次进度，只在自由变量位置计算 softmax 和预测，并及时释放全词表张量；随机张量仍按原完整形状生成，CPU 小模型 16 步对照已验证优化前后轨迹逐元素一致且随机数状态一致。增加 `--reuse-trajectories` 可保留并复用已经完整保存的划分轨迹。

## 0.5B 后训练正式结果

种子 17、29、43 均完成 800 步后训练，分别使用验证集选出的第 750、450、300 步 checkpoint。每个种子、每个测试划分评估 256 个 episode，即三种子每划分共 768 个 episode；所有方法复用相同的修订轨迹，并独立执行调参和统计协议。下表为三种子均值±种子标准差。

| 划分 | 方法 | 概念精确恢复 | 查询准确率 | 查询 NLL |
|---|---|---:|---:|---:|
| seen | final | 0.3281±0.0979 | 0.7546±0.0489 | 1.2520±0.3245 |
| seen | bag_of_states | 0.5990±0.0918 | 0.8460±0.0297 | 0.3954±0.0568 |
| seen | evidence_guided | 0.6536±0.0748 | 0.8424±0.0264 | 0.3897±0.0529 |
| seen | symbolic | 0.8164±0.0000 | 0.7539±0.0000 | 0.4652±0.0000 |
| unseen | final | 0.3268±0.1037 | 0.7614±0.0679 | 1.1336±0.5513 |
| unseen | bag_of_states | 0.4883±0.0938 | 0.8434±0.0346 | 0.3847±0.0491 |
| unseen | evidence_guided | 0.5208±0.0748 | 0.8320±0.0291 | 0.4049±0.0461 |
| unseen | symbolic | 0.8438±0.0000 | 0.7666±0.0000 | 0.4590±0.0000 |
| compositional | final | 0.0169±0.0098 | 0.7956±0.0250 | 0.8198±0.1300 |
| compositional | bag_of_states | 0.0729±0.0298 | 0.8197±0.0145 | 0.4087±0.0258 |
| compositional | evidence_guided | 0.4466±0.1361 | 0.8587±0.0321 | 0.3689±0.0379 |
| compositional | symbolic | 0.9375±0.0000 | 0.8643±0.0000 | 0.3565±0.0000 |

`evidence_guided` 在所有划分上超过最终输出；在三元组合上，相对无序状态袋的概念恢复和查询准确率平均配对增量分别为 0.3737 和 0.0391，说明修订产生的新候选对组合泛化有实际贡献。但它在已见和未见双属性查询上没有超过无序状态袋，不能声称时间顺序在所有任务上均有效。符号基线的概念精确恢复在所有划分仍然领先，因此当前结果不支持“轨迹方法已经优于符号求解”。

修订动力学探针的三种子 ROC AUC 为 0.7605±0.0765，最终输出特征为 0.6545±0.0697，无序状态袋为 0.6865±0.0731，支持修订轨迹包含额外预测信号；平均精度等探针指标并非全部优于无序状态袋，不能只选择有利指标。

完整结果位于 `outputs/dcoder_evaluation/seed_<seed>/evaluation/`，三种子聚合位于 `outputs/dcoder_evaluation/aggregate_summary.csv`，逐种子原始汇总、探针和配对比较也已保存。三个种子的逐 episode 文件、五组轨迹、统计区间和图像均已验收。优化后的种子 43 每个测试划分轨迹生成约 29–31 分钟；后续复现可复用已保存轨迹。

## 双对象关系归纳

`configs/relational.yaml` 将任务扩展为需要根据两个原始对象归纳关系规则，而不是仅把属性词改写为另一组概念。模型需要从颜色是否相同、形状是否相同、左侧是否更大、纹理是否相同四类关系中形成一到三项合取；测试仍严格区分已见规则、未见双关系和未见三关系组合。模型输入保留两个对象的完整属性，不使用派生关系标签作为输入。

先从已完成概念后训练的模型初始化，再进行完整的 800 步关系后训练：

```bash
python scripts/train_dcoder.py \
  --config configs/relational.yaml \
  --model outputs/dcoder_training/seed_17/best \
  --seed 17
```

```bash
python scripts/evaluate_dcoder.py \
  --config configs/relational.yaml \
  --model outputs/relational_training/seed_17/best \
  --output outputs/relational_evaluation/seed_17/evaluation
```

关系任务使用独立的关系 token 和独立的训练/验证/测试规则划分，合法候选空间为 64 个关系概念。65% 的负例是只改变一个目标关系的最小差异对象对，其余为随机负例；二元大小关系会直接构造合法边界样本。已有概念 checkpoint 仅作为参数初始化，不继承旧任务的训练步数或验证最优分数。

### 关系任务正式结果

完整执行 800 步后训练，使用验证集第 600 步最佳 checkpoint。每个划分包含 256 个独立 episode；结果目前为单种子，不能当作多种子稳定性结论。

| 划分 | 方法 | 概念精确恢复 | 查询准确率 | 查询 NLL |
|---|---|---:|---:|---:|
| seen | final | 0.2656 | 0.7549 | 0.7296 |
| seen | bag_of_states | 0.4180 | 0.8340 | 0.4201 |
| seen | evidence_guided | 0.4219 | 0.8271 | 0.4289 |
| seen | symbolic | 0.4570 | 0.6572 | 0.5156 |
| unseen | final | 0.1289 | 0.6563 | 1.0927 |
| unseen | bag_of_states | 0.2930 | 0.7393 | 0.6187 |
| unseen | evidence_guided | 0.2813 | 0.7246 | 0.6140 |
| unseen | symbolic | 0.4688 | 0.6895 | 0.5136 |
| compositional | final | 0.0000 | 0.7236 | 0.7309 |
| compositional | bag_of_states | 0.0000 | 0.7578 | 0.5312 |
| compositional | evidence_guided | 0.1719 | 0.7871 | 0.5037 |
| compositional | symbolic | 0.8477 | 0.7861 | 0.4718 |

`semantic_stability` 在未见双关系上的查询准确率为 0.7480，高于 `evidence_guided`。修订动力学探针 ROC AUC 为 0.8806，最终特征为 0.6328，说明轨迹仍含额外信号；但调参后的记忆方法没有稳定超过无序状态袋。新候选形成使三关系精确恢复从 0 提升到 0.1719，但远低于符号基线的 0.8477。因此本实验支持查询效用和候选形成的价值，不支持“轨迹方法已经精确解决关系归纳”。

完整结果位于 `outputs/relational_evaluation/seed_17/evaluation/`，聚合表位于 `outputs/relational_evaluation/aggregate_summary.csv`。


## 效用引导的建设性语义搜索

第一阶段诊断位于 `scripts/diagnose_revision.py`。它逐 episode 区分最终状态、全部轨迹状态、相邻修订候选与无顺序并交闭包的真实概念覆盖率，并测量真实候选出现后的排名、8 次时间打乱以及 oracle 记忆查询。诊断脚本和聚合结果分别为 `candidate_episodes.csv`、`candidate_summary.csv`、`order_episodes.csv`、`order_summary.csv` 和各结果根目录下的 `aggregate_candidate_diagnostics.csv`、`aggregate_order_ablation.csv`。

外部模型三组合的真实候选覆盖率从相邻修订的平均 0.4609 提升到并交闭包的 0.7214；真实候选一旦出现，其排名第一的比例约为 0.97。关系三组合对应覆盖率从 0.1758 提升到 0.7422。时间打乱对概念与查询指标的平均影响接近零。因此第二阶段不再把“时间动力学”作为核心主张，而把瓶颈定位为建设性候选覆盖。

### 搜索算法

`dreammem/constructive.py` 实现三种候选生成方式：

- `closure_search`：不学习，对轨迹状态反复求并集和交集；
- `random_search`：以相同初始候选、分支数和候选预算随机扩展；
- `learned_search`：使用训练 episode 拟合的梯度提升策略，选择删除、添加、替换、合并和交叉操作。

策略使用 37 个特征，包括逐经历错误、形成期与保留期错误、正负例错误、概念大小、属性和家族指示、轨迹频率、置信度及来源。训练标签只来自训练 episode 的真实概念；测试概念不参与拟合。候选生成后仍使用与原 `evidence_guided` 相同的形成期/保留期证据评分，已在真实轨迹上验证同候选集合时决策一致。

`random_search_matched` 强制使用学习策略的候选预算，用于排除预算差异。`symbolic` 继续枚举完整概念空间。策略训练和验证使用原训练/验证划分；由于原测试集已用于诊断，正式确认使用 episode 种子 17100 的全新测试经历，避免测试驱动的算法选择。

### 复现

```bash
python scripts/diagnose_revision.py \
  --config configs/dcoder.yaml \
  --evaluation outputs/dcoder_evaluation/seed_17/evaluation
```

```bash
python scripts/collect_dcoder_split.py \
  --config configs/dcoder.yaml \
  --model outputs/dcoder_training/seed_17/best \
  --split train \
  --output outputs/constructive_trajectories/dcoder/seed_17/train.pt
```

```bash
python scripts/train_constructive.py \
  --config configs/dcoder.yaml \
  --evaluation outputs/dcoder_evaluation/seed_17/evaluation \
  --train-trajectories outputs/constructive_trajectories/dcoder/seed_17/train.pt \
  --output outputs/constructive_policy/dcoder/seed_17 \
  --seed 1700
```

```bash
python scripts/evaluate_constructive.py \
  --config configs/dcoder.yaml \
  --model outputs/dcoder_training/seed_17/best \
  --policy outputs/constructive_policy/dcoder/seed_17 \
  --baseline-evaluation outputs/dcoder_evaluation/seed_17/evaluation \
  --output outputs/constructive_evaluation/dcoder/seed_17/evaluation \
  --episode-seed 17100
```

每个测试 episode 仍使用 288 次轨迹前向；oracle 解码额外执行一次前向，将真实概念固定、仅掩盖查询，报告神经模型的查询准确率和 NLL。所有搜索方法复用同一批已保存轨迹，不增加其 dLLM 前向预算。


### 新测试集确认：种子 17

种子 17 的学习策略在验证集选择 24 个候选；随机基线自身选择 96 个候选，另报告同样 24 候选的 `random_search_matched`。

| 划分 | 方法 | 概念精确恢复 | 查询准确率 | 查询 NLL |
|---|---|---:|---:|---:|
| seen | evidence_guided | 0.7188 | 0.8838 | 0.3337 |
| seen | closure_search | 0.7656 | 0.8428 | 0.3702 |
| seen | random_search_matched | 0.7148 | 0.8535 | 0.3587 |
| seen | random_search | 0.8047 | 0.7520 | 0.4577 |
| seen | learned_search | 0.7773 | 0.8604 | 0.3336 |
| seen | symbolic | 0.7930 | 0.7441 | 0.4665 |
| unseen | evidence_guided | 0.5977 | 0.8691 | 0.3551 |
| unseen | closure_search | 0.6289 | 0.8369 | 0.3975 |
| unseen | random_search_matched | 0.6133 | 0.8457 | 0.3771 |
| unseen | random_search | 0.7930 | 0.7813 | 0.4455 |
| unseen | learned_search | 0.7891 | 0.8760 | 0.3377 |
| unseen | symbolic | 0.7930 | 0.7715 | 0.4567 |
| compositional | evidence_guided | 0.5938 | 0.8975 | 0.3100 |
| compositional | closure_search | 0.7617 | 0.9082 | 0.3157 |
| compositional | random_search_matched | 0.6641 | 0.8916 | 0.2982 |
| compositional | random_search | 0.8789 | 0.8652 | 0.3649 |
| compositional | learned_search | 0.9141 | 0.9463 | 0.2635 |
| compositional | symbolic | 0.9336 | 0.8867 | 0.3544 |

学习搜索在三组合上的候选覆盖率为 0.9805，24 候选随机搜索为 0.7031，无学习闭包为 0.8164。学习搜索的概念恢复接近完整符号枚举，并在查询准确率、NLL 上超过符号后验；它不是全面超过符号精确恢复，已见划分仍略低于符号基线。

真实概念固定的神经 oracle 查询准确率分别为 seen 0.7686、unseen 0.7500、compositional 0.7861，明显低于编译记忆程序的查询表现。这说明原 0.5B 模型的条件解码本身仍是瓶颈，不能把记忆程序的查询准确率称为神经生成准确率。


### 新测试集确认：种子 29

种子 29 的学习预算为 32 个候选。三组合上，`evidence_guided`、闭包、同预算随机搜索、96 候选随机搜索、学习搜索和符号基线的概念恢复分别为 0.4844、0.7266、0.5781、0.8008、0.8750 和 0.9336；学习搜索查询准确率为 0.9180，NLL 为 0.2967。学习搜索候选覆盖率为 0.9531，同预算随机搜索为 0.6250。

未见双属性上，学习搜索概念恢复为 0.8125，高于符号基线的 0.7930；已见划分为 0.7969，也略高于符号基线，但其已见查询准确率 0.8125 低于原 `evidence_guided` 的 0.8604。因此提升同样不是所有指标全面占优。该种子的神经 oracle 三组合查询准确率为 0.7334。


### 外部模型三种子聚合

三个外部种子均完成独立训练轨迹采集、策略拟合、验证预算选择和种子 17100 新测试评估。下表为三种子均值±种子标准差。

| 划分 | 方法 | 概念精确恢复 | 查询准确率 | 查询 NLL |
|---|---|---:|---:|---:|
| seen | evidence_guided | 0.6654±0.0672 | 0.8584±0.0264 | 0.3851±0.0553 |
| seen | closure_search | 0.7461±0.0273 | 0.8167±0.0246 | 0.4128±0.0444 |
| seen | random_search_matched | 0.6888±0.0485 | 0.8265±0.0302 | 0.4076±0.0507 |
| seen | random_search | 0.8086±0.0141 | 0.7507±0.0060 | 0.4533±0.0085 |
| seen | learned_search | 0.7904±0.0113 | 0.8275±0.0285 | 0.3582±0.0215 |
| seen | symbolic | 0.7930±0.0000 | 0.7441±0.0000 | 0.4665±0.0000 |
| unseen | evidence_guided | 0.5234±0.0706 | 0.8398±0.0257 | 0.4015±0.0431 |
| unseen | closure_search | 0.6250±0.0103 | 0.8086±0.0245 | 0.4488±0.0460 |
| unseen | random_search_matched | 0.5833±0.0421 | 0.8197±0.0264 | 0.4227±0.0426 |
| unseen | random_search | 0.7852±0.0078 | 0.7813±0.0020 | 0.4560±0.0098 |
| unseen | learned_search | 0.8086±0.0179 | 0.8532±0.0197 | 0.3579±0.0175 |
| unseen | symbolic | 0.7930±0.0000 | 0.7715±0.0000 | 0.4567±0.0000 |
| compositional | evidence_guided | 0.4648±0.1397 | 0.8678±0.0280 | 0.3533±0.0457 |
| compositional | closure_search | 0.7122±0.0580 | 0.8877±0.0200 | 0.3396±0.0268 |
| compositional | random_search_matched | 0.5638±0.1081 | 0.8587±0.0303 | 0.3497±0.0469 |
| compositional | random_search | 0.7904±0.0942 | 0.8584±0.0110 | 0.3842±0.0193 |
| compositional | learned_search | 0.8802±0.0316 | 0.9238±0.0202 | 0.2859±0.0194 |
| compositional | symbolic | 0.9336±0.0000 | 0.8867±0.0000 | 0.3544±0.0000 |

三组合上，学习搜索相对无学习闭包的概念恢复平均配对增量为 0.1680，相对同预算随机搜索为 0.3164，相对各自调参的 96 候选随机搜索为 0.0898。学习搜索候选覆盖率为 0.9518±0.0293，闭包为 0.7734±0.0526，同预算随机搜索为 0.6016±0.1151。三个种子分别达到 0.9141、0.8750 和 0.8516，均超过预设的 0.70 继续投入门槛。

学习搜索没有全面超过符号精确恢复：三组合仍低 0.0534，已见划分基本相当，未见双属性平均高 0.0156。它的已见查询准确率低于原 `evidence_guided` 0.0309，因此主张应限定为“以更少的候选评估实现接近符号求解的概念恢复，并改善程序查询后验”，而不是所有指标全面领先。

神经 oracle 查询准确率为 seen 0.7165±0.0492、unseen 0.7064±0.0405、compositional 0.7409±0.0420，进一步说明编译程序与神经解码必须分开评价。聚合文件为 `outputs/constructive_evaluation/dcoder/aggregate_summary.csv`、`aggregate_search_diagnostics.csv`、`aggregate_oracle_decoder.csv` 和 `aggregate_paired_comparisons.csv`。


### 关系任务新测试确认

关系策略同样使用 256 个训练 episode 和完整验证预算选择，学习预算为 32 个候选。正式结果来自种子 17100 的 256 个全新 episode/划分，目前为单种子，不能当作多种子稳定性结论。

| 划分 | 方法 | 概念精确恢复 | 查询准确率 | 查询 NLL |
|---|---|---:|---:|---:|
| seen | evidence_guided | 0.4297 | 0.7998 | 0.4559 |
| seen | closure_search | 0.4258 | 0.7920 | 0.4196 |
| seen | random_search_matched | 0.4688 | 0.8066 | 0.4387 |
| seen | random_search | 0.5781 | 0.7930 | 0.4250 |
| seen | learned_search | 0.4961 | 0.8604 | 0.3172 |
| seen | symbolic | 0.4766 | 0.6719 | 0.5187 |
| unseen | evidence_guided | 0.2266 | 0.7295 | 0.5712 |
| unseen | closure_search | 0.3555 | 0.7773 | 0.4667 |
| unseen | random_search_matched | 0.2969 | 0.7627 | 0.5530 |
| unseen | random_search | 0.5430 | 0.8037 | 0.4474 |
| unseen | learned_search | 0.5703 | 0.8643 | 0.3224 |
| unseen | symbolic | 0.4922 | 0.7061 | 0.5047 |
| compositional | evidence_guided | 0.2188 | 0.7734 | 0.4901 |
| compositional | closure_search | 0.6758 | 0.8447 | 0.4035 |
| compositional | random_search_matched | 0.2695 | 0.7871 | 0.4720 |
| compositional | random_search | 0.5859 | 0.8145 | 0.4560 |
| compositional | learned_search | 0.6836 | 0.8662 | 0.3243 |
| compositional | symbolic | 0.8164 | 0.7744 | 0.4879 |

三关系学习搜索超过预设的 0.50 门槛，但无学习闭包也达到 0.6758，二者概念恢复差仅 0.0078，配对 bootstrap 区间跨零；闭包候选覆盖率 0.8008，还略高于学习搜索的 0.7891。因此关系任务不能支持“学习策略在组合候选覆盖上必然优于闭包”。学习搜索相对同预算随机搜索的概念恢复增量为 0.4141，相对 96 候选随机搜索为 0.0977；其查询 NLL 在三个划分均明显低于所有非学习基线。

关系神经 oracle 查询准确率仅为 seen 0.5068、unseen 0.5010、compositional 0.5332，说明即使给定真实关系，原模型的文本条件解码也接近随机，而编译出的离散机制仍能泛化。这强化了“输出可执行机制”的必要性，但不等于模型已经学会可靠的自然语言关系推理。

完整结果位于 `outputs/constructive_evaluation/relational/seed_17/evaluation/`，聚合文件位于 `outputs/constructive_evaluation/relational/aggregate_summary.csv`。


## 全空间排序器与完整成本审计

`scripts/audit_constructive.py` 在相同新测试轨迹上重新执行全部搜索，逐 episode 一致性断言通过；不增加 dLLM 前向。`learned_ranker_full` 直接对完整概念库进行策略排序，`learned_ranker_initial` 只对初始轨迹候选排序，`learned_search_policy_rank` 使用学习搜索候选但完全去掉证据评分。

外部模型三种子结果：

| 划分 | 方法 | 概念精确恢复 | 查询准确率 | 查询 NLL |
|---|---|---:|---:|---:|
| seen | learned_ranker_full | 0.9154±0.0098 | 0.9570±0.0026 | 0.0845±0.0009 |
| seen | learned_search_policy_rank | 0.9023±0.0039 | 0.9554±0.0031 | 0.1014±0.0265 |
| seen | learned_search | 0.7904±0.0113 | 0.8275±0.0285 | 0.3582±0.0215 |
| unseen | learned_ranker_full | 0.8672±0.0391 | 0.9574±0.0071 | 0.0867±0.0020 |
| unseen | learned_search_policy_rank | 0.8607±0.0453 | 0.9577±0.0100 | 0.0911±0.0098 |
| unseen | learned_search | 0.8086±0.0179 | 0.8532±0.0197 | 0.3579±0.0175 |
| compositional | learned_ranker_full | 0.0664±0.0305 | 0.8418±0.0074 | 0.4573±0.0296 |
| compositional | learned_search_policy_rank | 0.0638±0.0251 | 0.8418±0.0059 | 0.4686±0.0359 |
| compositional | learned_search | 0.8802±0.0316 | 0.9238±0.0202 | 0.2859±0.0194 |

直接学习排序在已见和未见双属性上优于证据搜索，但在新三组合上完全失败；即使学习搜索已经覆盖了真实候选，去掉证据评分仍然失败。因此不能把当前贡献简化为学习排序。可辩护的主线是“建设性候选与证据验收共同实现新组合机制归纳”，并需承认直接排序在已见机制上更好。

外部三组合平均成本：

| 方法 | 保留候选 | 实际生成的不同邻居 | 策略评分次数 | 证据评分数 | 搜索秒数/episode | 证据秒数/episode |
|---|---:|---:|---:|---:|---:|---:|
| closure_search | 29.46 | 29.45 | 0 | 29.46 | 0.0051 | 0.0010 |
| random_search | 96.00 | 108.17 | 0 | 96.00 | 0.0037 | 0.0024 |
| learned_search | 29.33 | 54.20 | 56.16 | 29.33 | 0.0076 | 0.0010 |
| learned_ranker_full | 203.00 | 0 | 203.00 | 0 | 0.0170 | 0 |
| symbolic | 203.00 | 0 | 0 | 203.00 | 0.0039 | 0.0039 |

学习搜索的策略与证据评分总量仍低于 96 候选随机搜索的证据评分数，但不同运算不能直接等价为同一成本；学习搜索和全空间排序器的墙钟时间更长，完整符号枚举在这些小型离散空间中最快。因此不能宣称绝对速度优势，主张仅限于候选效率和组合泛化。

完整逐 episode 结果位于各评估目录的 `audit/`，聚合文件为 `aggregate_audit_results.csv` 和 `aggregate_audit_costs.csv`。


## 组件消融与同预算经典搜索

`scripts/run_ablations.py` 对全部新测试 episode 执行 17 种方法，每个模型共 13,056 条结果；`scripts/aggregate_ablations.py` 先计算各种子均值再聚合。所有学习消融使用相同候选预算和最终证据验收。

外部模型三组合关键结果：

| 方法 | 概念精确恢复 | 查询准确率 | 查询 NLL |
|---|---:|---:|---:|
| learned_search | 0.8802±0.0316 | 0.9238±0.0202 | 0.2859±0.0194 |
| greedy_search | 0.9010±0.0296 | 0.9176±0.0140 | 0.3043±0.0205 |
| beam_search | 0.9010±0.0296 | 0.9170±0.0138 | 0.3045±0.0202 |
| astar_search | 0.8737±0.0229 | 0.9167±0.0133 | 0.2998±0.0230 |
| learned_search_logistic | 0.8906±0.0295 | 0.9160±0.0135 | 0.2936±0.0172 |
| learned_search_mlp | 0.8880±0.0363 | 0.9147±0.0180 | 0.2964±0.0205 |
| learned_search_no_merge | 0.8802±0.0316 | 0.9238±0.0199 | 0.2858±0.0193 |
| learned_search_no_trajectory_features | 0.8815±0.0332 | 0.9300±0.0142 | 0.2803±0.0178 |
| learned_search_no_evidence_features | 0.4583±0.1376 | 0.8672±0.0264 | 0.3429±0.0357 |
| learned_search_atoms | 0.0000±0.0000 | 0.8197±0.0108 | 0.3697±0.0157 |
| closure_search | 0.7122±0.0580 | 0.8877±0.0200 | 0.3396±0.0268 |
| random_search_matched | 0.5638±0.1081 | 0.8587±0.0303 | 0.3497±0.0469 |
| symbolic | 0.9336±0.0000 | 0.8867±0.0000 | 0.3544±0.0000 |

这些结果否定“学习控制器是必要组件”：经典贪心和束搜索的概念恢复更高，学习策略仅在 NLL 上略有优势。删除合并算子几乎不影响结果，删除轨迹特征也不下降，因此它们都不能作为核心贡献。删除证据特征显著下降，说明证据是策略选择所必需的。

仅从单属性开始的学习搜索在三组合上为 0，但在 seen/unseen 分别达到 0.8893 和 0.9167，超过使用神经轨迹初始化的版本。因此神经轨迹初始化的作用主要是让新组合能够进入候选空间；在已见机制上，简单原子初始化更好。

关系任务也呈相同趋势：三组合贪心/束搜索分别为 0.7188/0.7148，学习搜索为 0.6836，MLP 策略为 0.7383；原子初始化在三组合上为 0，但在 unseen 为 0.9063。由此项目主线必须收紧为**证据引导的建设性机制搜索**，而不是学习控制器或轨迹动力学。后续多种子和新机制实验均以经典搜索为必要强基线。


### 关系任务三种子最终确认

种子 17、29、43 均完成 800 步后训练、独立验证调参、训练轨迹采集、策略拟合、新测试评估、成本审计和全量消融。下表为三种子均值±种子标准差。

| 划分 | 方法 | 概念精确恢复 | 查询准确率 | 查询 NLL |
|---|---|---:|---:|---:|
| seen | closure_search | 0.4284±0.0081 | 0.7926±0.0020 | 0.4221±0.0033 |
| seen | random_search_matched | 0.4661±0.0158 | 0.7946±0.0127 | 0.4333±0.0057 |
| seen | random_search | 0.5820±0.0039 | 0.7917±0.0031 | 0.4094±0.0136 |
| seen | learned_search | 0.5117±0.0156 | 0.8587±0.0028 | 0.3195±0.0047 |
| seen | symbolic | 0.4766±0.0000 | 0.6719±0.0000 | 0.5187±0.0000 |
| unseen | closure_search | 0.3581±0.0119 | 0.7780±0.0088 | 0.4791±0.0327 |
| unseen | random_search_matched | 0.3112±0.0429 | 0.7552±0.0147 | 0.5417±0.0253 |
| unseen | random_search | 0.5469±0.0039 | 0.8060±0.0057 | 0.4414±0.0182 |
| unseen | learned_search | 0.5690±0.0098 | 0.8597±0.0039 | 0.3230±0.0032 |
| unseen | symbolic | 0.4922±0.0000 | 0.7061±0.0000 | 0.5047±0.0000 |
| compositional | closure_search | 0.6693±0.0060 | 0.8467±0.0026 | 0.4041±0.0014 |
| compositional | random_search_matched | 0.2578±0.0141 | 0.7881±0.0010 | 0.4779±0.0069 |
| compositional | random_search | 0.5742±0.0385 | 0.8122±0.0025 | 0.4563±0.0103 |
| compositional | learned_search | 0.6836±0.0117 | 0.8639±0.0025 | 0.3217±0.0049 |
| compositional | symbolic | 0.8164±0.0000 | 0.7744±0.0000 | 0.4879±0.0000 |

三组合上，学习搜索仅比闭包高 0.0143；贪心和束搜索分别为 0.7253±0.0060 和 0.7240±0.0081，均超过学习搜索。MLP 策略为 0.7214±0.0176。学习搜索的查询 NLL 仍最低，因此不同指标必须分别报告。

全空间学习排序器在三组合仅 0.1758±0.0078，原子初始化搜索为 0，再次说明新组合需要神经候选与证据验收，而不是学习排序器单独完成。三种子神经 oracle 查询准确率为 seen 0.5029、unseen 0.4954、compositional 0.5293，仍接近随机。

关系聚合与消融文件位于 `outputs/constructive_evaluation/relational/`，包括 `aggregate_summary.csv`、`aggregate_ablation_results.csv`、`aggregate_audit_results.csv` 和对应逐种子原始表。


## 扩展任务、解码器与第二后端

以下配置共享相同的候选构造、证据验收和独立测试协议；运行状态不能由配置存在推断为实验已完成，最终数值以对应 `evaluation/summary.csv` 为准。

| 配置 | 研究变量 | 测试划分 |
|---|---|---|
| `configs/advanced.yaml` | 否定、析取、XOR、蕴含、阈值、传递关系 | 9 个划分，各 256 episode |
| `configs/large.yaml` | 32 属性、8 家族、4,064 个合法概念 | seen/unseen/compositional |
| `configs/natural.yaml` | 历史、交通、组织、设备、抽象领域模板 | 3 个划分，模板隔离 |
| `configs/noisy10.yaml`、`noisy25.yaml` | 10%/25% 支持标签翻转，查询标签保持干净 | 3 个划分 |
| `configs/support2.yaml` 等 | 2/4/16/32 条可用支持经历及相应轨迹前缀 | 3 个划分 |
| `configs/nanodiff.yaml` | nanoDiff 第二后端 | 3 个划分 |

大空间概念位编码使用 `int64`，避免第 32 个属性的第 31 位超出 `int32`；已用真实 tokenizer 对属性 31 的编码—解码往返和 `2**31` 位值完成边界验证。

### 九划分新算子正式结果

高级模型已完成 800 步后训练、独立调参和种子 17100 的九划分正式评估，每划分 256 个 episode。下表为概念精确恢复；这些是单种子结果，且审计/消融独立保存在 `evaluation/audit/` 与 `evaluation/ablations/`。

| 划分 | 学习搜索 | 无学习闭包 | 96 候选随机 | 符号枚举 |
|---|---:|---:|---:|---:|
| seen | 0.2344 | 0.3555 | 0.3398 | 0.6602 |
| unseen | 0.4063 | 0.3555 | 0.3477 | 0.9609 |
| compositional | 0.4805 | **0.7109** | 0.5273 | 0.8555 |
| negation | 0.1875 | 0.0000 | 0.1953 | 0.4180 |
| disjunction | 0.1172 | 0.0000 | **0.2656** | 0.3633 |
| xor | 0.1914 | 0.0000 | 0.1328 | 0.4805 |
| implication | 0.1992 | 0.0000 | 0.1719 | 0.7070 |
| threshold | 0.1641 | 0.0000 | 0.1953 | 0.4805 |
| transitive | 0.1367 | 0.0000 | **0.1484** | 0.6055 |

审计中的符号枚举与主评估使用完全相同的划分合法子库，并已逐 episode 对拍一致。全空间学习排序在 seen、unseen、compositional 上分别为 0.7813、1.0000、0.8086；这些划分的合法候选数分别只有 25、3、27，不能与 203 概念的全库直接混用。新算子上的全空间排序分别为 negation 0.3750、disjunction 0.2031、xor 0.3516、implication 0.4727、threshold 0.3516、transitive 0.4414，多数高于学习搜索但仍低于符号枚举。因此 advanced 结果同样不支持“构造搜索必然优于直接排序”。

17 方法消融中，三组合贪心/束搜索分别为 0.4922/0.5000，MLP 为 0.5742，均高于学习搜索的 0.4805，但仍低于闭包的 0.7109、全空间排序的 0.8086 和符号枚举的 0.8555。删除证据特征后三组合降至 0.2031，但 seen 上反而升至 0.5625；删除轨迹特征后在 unseen 上为 0.5508，也高于原学习搜索。原子初始化在三组合为 0，在 seen/unseen 则为 0.5352/0.6641。说明特征和初始化收益依任务而变化，不能从主实验直接推广到新算子。

学习搜索在新算子上的查询准确率为 0.7441–0.8516，但精确恢复明显不足；三组合还低于无学习闭包。所有新算子划分的固定真实概念神经 oracle 查询准确率仅为 0.5078–0.5352，接近随机。因此该实验是明确的泛化失败边界，不支持“已经可靠掌握新算子”。模型验证期各 checkpoint 的直接概念准确率也均为 0，最佳 checkpoint 为第 300 步。

### 多模板自然语义正式结果

自然语义模型完成 800 步后训练，最佳 checkpoint 为第 650 步。正式测试使用种子 17100，每个划分 256 个全新 episode；历史、交通、组织、设备和抽象模板按训练、验证与测试条件隔离。下表是单种子结果。

| 划分 | 方法 | 概念精确恢复 | 查询准确率 | 查询 NLL |
|---|---|---:|---:|---:|
| seen | evidence_guided | 0.4766 | 0.7490 | 0.5957 |
| seen | closure_search | 0.6055 | 0.7305 | 0.6365 |
| seen | random_search | 0.7656 | 0.7588 | 0.4641 |
| seen | learned_search | **0.7930** | 0.7500 | **0.4517** |
| seen | symbolic | **0.7930** | 0.7441 | 0.4665 |
| unseen | evidence_guided | 0.0000 | 0.5166 | 2.2338 |
| unseen | closure_search | 0.0000 | 0.5176 | 2.2355 |
| unseen | random_search | 0.2188 | 0.5898 | 0.6451 |
| unseen | learned_search | 0.2383 | 0.5918 | 0.6117 |
| unseen | symbolic | **0.7930** | **0.7715** | **0.4567** |
| compositional | evidence_guided | 0.1836 | 0.8555 | **0.3437** |
| compositional | closure_search | 0.2852 | 0.8604 | 0.3577 |
| compositional | random_search | 0.3320 | 0.8584 | 0.3768 |
| compositional | learned_search | 0.3945 | 0.8662 | 0.3531 |
| compositional | symbolic | **0.9336** | **0.8867** | 0.3544 |

本实验的学习预算为 96 个候选，因此 `random_search_matched` 与 `random_search` 数值相同。学习搜索在已见模板上与符号枚举并列，在未见模板和组合模板上有一定候选构造能力，但精确恢复仍远低于符号基线；尤其未见模板只有 0.2383，不能据此宣称已经解决自然语言语义泛化。固定真实概念的神经 oracle 查询准确率分别为 seen 0.6738、unseen 0.5205、compositional 0.7725，再次说明神经条件解码与编译程序必须分开评价。

大概念空间模型已完成 800 步后训练，训练耗时 2607.71 秒，峰值显存约 5.56GiB；验证集选择第 700 步 checkpoint，验证查询准确率为 0.7344、直接概念准确率为 0.25。训练指标不构成 4,064 概念空间上的正式构造搜索结果。

### 4,064 概念空间正式结果与审计

种子 17100 的三个划分均完成 256 个 episode，学习搜索预算为 48，较大随机预算为 96。下表比较概念精确恢复；`random48` 是与学习搜索同预算的随机基线，`full ranker` 是全空间学习排序：

| 划分 | random48 | random96 | 闭包 | 学习搜索 | 符号枚举 | full ranker |
|---|---:|---:|---:|---:|---:|---:|
| seen | 0.4805 | 0.5234 | 0.5313 | 0.7695 | **0.7930** | **0.8984** |
| unseen | 0.4414 | 0.5313 | 0.5352 | **0.7852** | 0.7813 | **0.8945** |
| compositional | 0.2305 | 0.3086 | 0.4180 | **0.6563** | **0.8438** | 0.0078 |

学习搜索的查询准确率分别为 0.7100、0.7188、0.7783，NLL 分别为 0.5271、0.5128、0.4397；符号查询准确率为 0.6055、0.6162、0.7227。证据引导基线的概念恢复为 0.4375、0.4102、0.1953，查询准确率为 0.7461、0.7393、0.7441，因此学习搜索并非所有查询指标都优于简单基线。学习搜索候选覆盖率为 0.9219、0.9453、0.7188，三组合的候选覆盖明显低于前两个划分。

全空间学习排序在 seen/unseen 上最强，但三组合概念恢复仅 0.0078；其查询准确率仍达到 0.8359，说明高查询准确率不能替代真实机制精确恢复。固定真实概念的神经 oracle 查询准确率为 0.6357、0.6426、0.6523。三组合上，构造搜索和证据验收仍具有直接排序无法替代的作用，但符号枚举仍明显更强。

学习搜索每 episode 证据评分 48 个候选，实际生成不同邻居约 197.9、199.8、188.3 个，策略评分约 205.2、207.8、194.1 个实例；全空间排序和符号枚举均处理 4,064 个概念。学习搜索墙钟时间约 0.02 秒/episode，全空间排序约 0.277 秒，符号枚举约 0.074 秒，但所有非符号方法还共享每 episode 288 次轨迹前向，不能只比较搜索阶段耗时。三个划分的轨迹采集分别耗时 2696.31、2686.52、2673.65 秒。以上为单种子结果，完整审计位于 `evaluation/audit/`。

17 方法消融中，三组合贪心与束搜索均为 **0.7500**，高于学习搜索的 0.6563；MLP 和 logistic 策略分别为 0.7148 和 0.6719。删除证据特征后学习搜索降至 0.1992，删除轨迹特征后反而为 0.6914，删除合并算子保持 0.6563。原子初始化在三组合为 0，但在 seen/unseen 均达到 0.8906，再次说明轨迹初始化主要贡献于新组合，而非已见机制。经典搜索在大空间上超过学习控制器，因此最终主张仍不能写成学习控制器必要或最优。

完整外部模型流水线按以下顺序运行，不能只执行最后一步而省略独立验证或训练轨迹：

```bash
python scripts/train_dcoder.py --config configs/large.yaml --model outputs/dcoder_training/seed_17/best --seed 17
python scripts/prepare_constructive_baseline.py --config configs/large.yaml --model outputs/large_training/seed_17/best --output outputs/constructive_baseline/large/seed_17
python scripts/collect_dcoder_split.py --config configs/large.yaml --model outputs/large_training/seed_17/best --split train --output outputs/constructive_trajectories/large/seed_17/train.pt --episodes 256 --seed 7100
python scripts/train_constructive.py --config configs/large.yaml --evaluation outputs/constructive_baseline/large/seed_17 --train-trajectories outputs/constructive_trajectories/large/seed_17/train.pt --output outputs/constructive_policy/large/seed_17
python scripts/evaluate_constructive.py --config configs/large.yaml --model outputs/large_training/seed_17/best --policy outputs/constructive_policy/large/seed_17 --baseline-evaluation outputs/constructive_baseline/large/seed_17 --output outputs/constructive_evaluation/large/seed_17/evaluation --episode-seed 17100
python scripts/audit_constructive.py --config configs/large.yaml --evaluation outputs/constructive_evaluation/large/seed_17/evaluation --policy outputs/constructive_policy/large/seed_17 --baseline-evaluation outputs/constructive_baseline/large/seed_17
python scripts/run_ablations.py --config configs/large.yaml --evaluation outputs/constructive_evaluation/large/seed_17/evaluation --policy outputs/constructive_policy/large/seed_17 --train-trajectories outputs/constructive_trajectories/large/seed_17/train.pt --baseline-evaluation outputs/constructive_baseline/large/seed_17 --output outputs/constructive_evaluation/large/seed_17/evaluation/ablations
```

噪声与支持长度实验复用 `outputs/dcoder_training/seed_17/best`，但重新采集对应条件的训练、验证和测试轨迹并重新调参。这样隔离的是数据噪声和支持长度，而不是重新训练权重带来的差异。支持长度成本按实际阶段数计算。

### 10% 与 25% 标签噪声正式结果

两种噪声条件均复用同一个 Open-dCoder 后训练 checkpoint，但重新采集训练、验证和测试轨迹并独立调参；每个测试划分使用 256 个全新 episode。下表为单种子结果，概念指标是精确恢复率。

| 支持标签翻转率 | 划分 | 学习搜索概念 | 学习搜索查询 | 符号枚举概念 | 符号枚举查询 |
|---|---|---:|---:|---:|---:|
| 10% | seen | 0.6211 | 0.8125 | 0.6094 | 0.5508 |
| 10% | unseen | 0.5977 | 0.7930 | 0.6094 | 0.5576 |
| 10% | compositional | 0.6484 | 0.8701 | **0.6719** | 0.6982 |
| 25% | seen | 0.2539 | 0.6816 | 0.2383 | 0.5273 |
| 25% | unseen | **0.2773** | **0.6895** | 0.2461 | 0.5322 |
| 25% | compositional | 0.2773 | 0.7598 | **0.3359** | 0.6123 |

10% 噪声下，学习搜索在已见和未见双属性上接近或略高于符号精确恢复，但三组合仍较低；贪心/束搜索的三组合恢复为 0.6758，也高于学习搜索。25% 噪声下所有方法都明显下降：学习搜索的三组合恢复只有 0.2773，无学习闭包为 0.3164，符号枚举为 0.3359。因此后一个条件是明确的负向稳健性边界，不能将查询准确率高于符号后验解释为已可靠恢复真实机制。

含噪支持标签与干净查询标签之间存在不一致；这里评价的是在噪声经历下恢复干净概念和预测干净查询的能力。完整结果及审计、17 方法消融分别位于 `outputs/constructive_evaluation/noisy10/` 和 `outputs/constructive_evaluation/noisy25/`。

### 支持长度实验：2、4、16 与 32 条支持经历

四种条件均完成 256 episode/划分的独立测试、神经 oracle、成本审计和 17 方法消融。下表为单种子结果：

| 支持数 | 划分 | 学习搜索概念 | 学习搜索查询 | 符号枚举概念 | 符号枚举查询 |
|---|---|---:|---:|---:|---:|
| 2 | seen | 0.0391 | 0.6592 | 0.0000 | 0.5488 |
| 2 | unseen | 0.0859 | 0.6885 | 0.0000 | 0.5645 |
| 2 | compositional | **0.3008** | **0.7822** | 0.2929 | 0.6738 |
| 4 | seen | **0.3789** | **0.7500** | 0.3750 | 0.5928 |
| 4 | unseen | **0.4219** | **0.7578** | 0.4023 | 0.5928 |
| 4 | compositional | 0.6953 | **0.8662** | **0.7188** | 0.7295 |
| 16 | seen | **0.9727** | **0.8418** | 0.9688 | 0.8242 |
| 16 | unseen | **0.9922** | **0.8486** | 0.9570 | 0.8242 |
| 16 | compositional | **0.9688** | **0.9434** | 0.6523 | 0.9404 |
| 32 | seen | 0.9961 | **0.9404** | **1.0000** | 0.8682 |
| 32 | unseen | 0.9883 | **0.9316** | **1.0000** | 0.8818 |
| 32 | compositional | 0.9258 | 0.9717 | **0.9531** | **0.9766** |

两条支持时，seen/unseen 的符号精确恢复为 0，不是实现失败，而是有限经历经常无法区分多个合法概念。增加到四条后各划分明显改善，但三组合仍未达到符号枚举。四条支持的三组合上，学习搜索 0.6953 略高于贪心/束搜索的 0.6797/0.6836；这一结果不支持学习控制器必然较差，也不构成普遍优势。

十六条支持时学习搜索在三个划分均超过符号 MAP，32 条支持时符号精确恢复重新领先。符号结果依赖独立调参的评分规则，不能视为信息论上界；也不能由单个种子推断普遍优势。短支持条件必须把 posterior、校准和弃权行为与 MAP 概念精确恢复分开解释，不能把查询准确率提升写成已恢复唯一真实机制。完整结果位于 `outputs/constructive_evaluation/support2/`、`support4/`、`support16/` 和 `support32/`。

### 真正的 AR 与 MLM 解码对照

```bash
python scripts/evaluate_decoders.py --config configs/dcoder.yaml --model outputs/dcoder_training/seed_17/best --baseline-evaluation outputs/dcoder_evaluation/seed_17/evaluation --output outputs/decoder_evaluation/dcoder/seed_17/ar --mode ar --episode-seed 17100
python scripts/evaluate_decoders.py --config configs/dcoder.yaml --model outputs/dcoder_training/seed_17/best --baseline-evaluation outputs/dcoder_evaluation/seed_17/evaluation --output outputs/decoder_evaluation/dcoder/seed_17/mlm --mode mlm --episode-seed 17100
```

AR 路径使用因果注意力、从左到右提交七个输出位置，并以前一位置 logits 预测当前 token；每 episode 为 294 次前向。MLM 每阶段一次并行预测，共 18 次前向。主 dLLM 为 288 次前向。三者采样数和迭代结构不同，必须同时报告准确率、NLL、前向次数和墙钟时间。

两种对照均完成全部 256 episode/划分。下表为种子 17 的结果；`evidence_guided` 是相应轨迹的直接记忆基线，概念列为精确恢复率。

| 解码 | 划分 | 方法 | 概念 | 查询 | 前向/episode | 轨迹秒数 |
|---|---|---|---|---:|---:|---:|
| AR | seen | 非符号方法 | 0.0000 | 0.5000 | 294 | 949.38 |
| AR | unseen | 非符号方法 | 0.0000 | 0.5000 | 294 | 1262.96 |
| AR | compositional | 非符号方法 | 0.0000 | 0.5000 | 294 | 960.21 |
| MLM | seen | evidence_guided | 0.6016 | 0.8525 | 18 | 393.96 |
| MLM | unseen | evidence_guided | 0.4961 | 0.8545 | 18 | 73.77 |
| MLM | compositional | evidence_guided | 0.3672 | 0.8662 | 18 | 159.54 |
| 主 dLLM | seen | learned_search | 0.7773 | 0.8604 | 288 | — |
| 主 dLLM | unseen | learned_search | 0.7891 | 0.8760 | 288 | — |
| 主 dLLM | compositional | learned_search | 0.9141 | 0.9463 | 288 | — |

AR 的非符号轨迹没有形成合法概念，统一退化为 0 概念恢复和随机查询，不能据此宣称 AR 架构普遍无法归纳；这只说明当前权重、因果注意力和从左到右提交协议失败。MLM 在 18 次前向下有一定恢复能力，但三组合仍明显低于迭代去噪和构造搜索。主 dLLM 与 MLM 使用 6 条采样，AR 使用 14 条，因此前向预算、采样数和修订结构并不完全相同，不能把差异全部归因于架构。符号枚举与主实验相同，不依赖这些解码轨迹。

完整结果位于 `outputs/decoder_evaluation/dcoder/seed_17/ar/` 和 `mlm/`。

### nanoDiff 后端

[nanoDiff](https://github.com/BY571/nanoDiff) 适配器位于 `dreammem/nanodiff_adapter.py`。已下载的 `nanodiff-150m-base.pt` 实际包含 203,318,272 个总参数，不能将文件名当作精确参数量。真实权重已通过加载、有限 logits、保存恢复一致性检查；逐块梯度检查点通过输出、损失和全部参数梯度等价检查。运行完整实验时将模型路径替换为 `models/nanodiff-150m-base/nanodiff-150m-base.pt`，配置使用 `configs/nanodiff.yaml`。

种子 17 已完成完整 800 步同任务后训练，训练耗时 558.44 秒，峰值显存约 1.85GiB。验证集选择第 500 步 checkpoint，其验证查询准确率为 0.5625、直接概念准确率为 0。直接输出指标与后续构造搜索结果必须分别报告，不能用直接概念失败预先推断构造搜索也失败。

### nanoDiff 正式测试与后端边界

种子 17100 的三个测试划分均完成 256 个 episode，学习与随机搜索的验证集选择预算都为 96，因此 `random_search_matched` 与 `random_search` 相同。下表同时给出概念精确恢复和程序查询准确率：

| 划分 | 随机搜索概念 | 学习搜索概念 | 学习搜索查询 | 全空间学习排序概念 | 全空间学习排序查询 | 符号概念 | 符号查询 |
|---|---:|---:|---:|---:|---:|---:|---:|
| seen | 0.5781 | 0.6719 | 0.5918 | **0.9102** | **0.9600** | 0.7930 | 0.7441 |
| unseen | 0.5156 | 0.5664 | 0.6152 | 0.7734 | **0.9414** | **0.7930** | 0.7715 |
| compositional | 0.0352 | **0.1250** | 0.7637 | 0.1016 | 0.8418 | **0.9336** | **0.8867** |

学习搜索的查询 NLL 分别为 0.6033、0.5882、0.5083；全空间学习排序为 0.0855、0.0886、0.4269。`final` 在三个划分的概念恢复均为 0，`evidence_guided` 分别为 0.0195、0.0078、0，闭包搜索分别为 0.0195、0.0273、0。构造搜索显著优于直接轨迹选择，但没有超过符号枚举。

学习搜索候选覆盖率达到 seen 0.9531、unseen 0.9648、compositional 0.9844；相同候选下，纯策略排序在 seen/unseen 的概念恢复为 0.8711/0.7617，明显高于证据验收的 0.6719/0.5664。全空间学习排序也强于证据搜索，与 Open-dCoder 三组合上直接排序失败的现象不同。因此不能把“证据验收必然优于学习排序”或某一种后端上的排序结论推广为普遍规律。

组合测试的初始候选真实概念覆盖率为 0，学习构造把它提高到 0.9844，但最终精确恢复仅 0.1250，清楚地区分了候选覆盖和正确验收。固定真实概念的神经 oracle 查询准确率仅为 0.5693、0.5371、0.5566，编译程序和神经条件解码仍需分开评价。三个划分的轨迹采集分别耗时 676.63、696.04、684.75 秒，每 episode 均为 288 次轨迹前向。以上为单种子、单后端结果，完整成本审计位于 `evaluation/audit/`。

17 方法消融进一步限制了就因果机制作出的解释：仅从原子初始化的学习搜索在三个划分达到 0.6875、0.5898、0.1289，均略高于神经轨迹初始化的 0.6719、0.5664、0.1250，因此该后端不能证明神经初始候选是必要的。三组合上学习搜索高于贪心的 0.1055 和束搜索的 0.1094，但 seen 上二者均为 0.6836；控制器没有一致胜负。删除证据特征后三组合降至 0.0469，删除轨迹特征后为 0.1211，删除合并算子后为 0.1211，说明证据特征重要，而后两类组件在该实验中没有独立收益。

## 最终统计与机器验收

```bash
python scripts/final_statistics.py --samples 1000 --final
python scripts/final_plots.py
python scripts/render_paper_tables.py
python scripts/validate_project.py --final
python -m compileall -q dreammem scripts
```

`final_statistics.py` 输出跨任务和跨种子指标、配对 bootstrap 效应、完整成本及全部逐 episode 结果。配对区间在每个种子内部重采样 episode，再等权平均种子估计；它不是对无限种子总体的推断。

`validate_project.py --final` 要求全部计划和训练步数、测试划分、轨迹、oracle、审计、消融、统计文件及 AR/MLM 成本完整存在，并逐一验证“划分 × episode × 方法”的固定组合、键唯一性和结果种子。任何关键指标缺失、关键列 NaN、重复结果或组合不完整都会失败；非闭包搜索的 `closure_iterations` 和单种子标准差允许为空。论文文稿位于 `PAPER.md`，结论以已验收结果为限。

最终验收已通过全部 **18 个 run**，包括 16 个构造实验和 AR/MLM 两个解码对照；最终统计包含 **208,896 条主方法逐 episode 结果**。验收清单位于 `outputs/final_validation/validation.json` 和 `inventory.csv`，自动论文表格位于 `outputs/final_statistics/paper_tables.md`，完整代码编译检查也已通过。
