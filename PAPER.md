# From Denoising Traces to Verifiable Concepts: Evidence-Guided Constructive Memory in Masked Diffusion Language Models

## Abstract

Masked diffusion language models produce a sequence of semantic hypotheses rather than a single final answer. We study whether these hypotheses can support memory formation from support experiences: constructing a compact, executable concept and using it to predict new queries within a held-out episode. We introduce evidence-guided constructive memory, which separates candidate generation from candidate acceptance. Denoising traces supply observed and revised hypotheses; symbolic neighborhood operators construct additional candidates; and held-out experiences determine which candidates survive. The learned search controller is optional and is evaluated separately from the underlying constructive mechanism.

On a post-trained Open-dCoder-0.5B model, three seeds, and fresh test episodes, learned constructive search recovers 0.8802 ± 0.0316 of unseen three-attribute concepts, compared with 0.5638 ± 0.1081 for a candidate-budget-matched random search and 0.7904 ± 0.0942 for a 96-candidate random search. A symbolic enumerator remains stronger at 0.9336. On relational tasks, the corresponding learned-search result is 0.6836 ± 0.0117, compared with 0.2578 ± 0.0141 for matched random search and 0.5742 ± 0.0385 for 96-candidate random search. However, greedy and beam search outperform the learned controller on both task families, and removing trajectory features does not reduce performance. The evidence therefore supports constructive candidate formation followed by evidence-based acceptance, not the necessity of a learned controller or a unique advantage of temporal trajectory features. On the nanoDiff backend, atom-only initialization slightly outperforms trace initialization and full-space learned ranking is stronger on seen and unseen pairs, while compositional exact recovery remains only 0.1250; these results reveal substantial backend dependence. All tasks are synthetic; we do not claim natural-language long-term memory consolidation or biological dreaming.

## 1. Introduction

Iterative masked denoising exposes intermediate predictions that are normally discarded. Prior work has used these predictions for voting, decoding control, reinforcement, and hallucination detection. A distinct question is whether they can contribute to **experience-to-query memory**: given a few labeled experiences, can a model form a reusable rule that remains valid on new objects?

This question requires more than selecting the most frequent final token. A correct rule may never be the final output, and a frequently sampled rule may be confidently wrong. Conversely, retaining every intermediate state is insufficient when the target is a new composition that must be constructed from partial hypotheses. We therefore separate two operations:

1. **Construction:** generate candidate executable concepts from final states, intermediate states, revisions, and symbolic neighborhoods.
2. **Acceptance:** score candidates using formation experiences and separate held-out acceptance experiences, with explicit complexity and specificity terms.

This separation makes negative results informative. A full-space learned ranker can fail even when constructive search succeeds, and a learned controller can underperform classical evidence-guided search without invalidating the value of candidate construction.

## 2. Problem Definition

An episode contains a hidden concept \(c\), a support set \(S=\{(x_i,y_i)\}_{i=1}^{n}\), and queries \(Q=\{(x_j,y_j)\}_{j=1}^{m}\). In the base task, \(c\) is a conjunction of one to three attributes from mutually exclusive attribute families. The executable semantics are

\[
f_c(x)=\bigwedge_{a\in c}\mathbf 1[a\in x].
\]

The model receives support experiences but not \(c\). During offline memory formation, we collect \(R\) denoising trajectories at several support prefixes. Each trajectory contains \(T\) predictions for three concept slots and the query labels. The resulting memory is a discrete concept \(\hat c\), or an abstention-compatible posterior over concepts. Only after memory formation do we evaluate predictions on the queries.

The support examples are divided by the largest trajectory prefix into formation and acceptance portions. In the standard eight-support setting, trajectory generation uses prefixes of two, four, and six examples; the remaining two examples provide additional acceptance evidence. Test queries never participate in model selection, candidate scoring, or hyperparameter tuning.

## 3. Method

### 3.1 Revisable masked denoising

At each denoising step, the model predicts all currently free concept and query positions. Confidence-based commitment fixes a subset of positions, while low-confidence committed positions can be remasked. We record token identities and confidence at every step, not merely the final state. The default external-model configuration uses six samples, 16 denoising steps, and three support prefixes, for 288 backbone forward passes per episode.

Concept tokens are parsed into canonical semantic codes. Invalid sequences are marked invalid rather than silently mapped to a valid concept. This representation enables us to distinguish state frequency from semantic revision and to apply executable operators to candidate concepts.

### 3.2 Candidate construction

The initial candidate pool contains concepts observed anywhere in the trajectories and concepts produced by revision operators. Constructive search then applies four neighborhood operations:

- remove an attribute;
- add a valid attribute;
- substitute an attribute while preserving family constraints;
- merge or intersect a concept with concepts from a small pair pool.

The search is budgeted by the number of retained candidates. We additionally report the number of unique generated neighbors, generated neighbor instances, controller evaluations, and evidence-scored candidates. Retained-candidate budget alone is not a complete account of computation.

### 3.3 Evidence-based acceptance

For candidate \(c\), let \(E_f(c)\) be its error on formation examples and \(E_v(c)\) its error on held-out acceptance examples. Its score is

\[
s(c)=-\lambda_f E_f(c)-\lambda_v E_v(c)
-\lambda_c |c|+\lambda_s |c|-\lambda_g\mathbf 1[c\notin \mathcal C_{\mathrm{obs}}],
\]

where \(\mathcal C_{\mathrm{obs}}\) contains observed candidates. The complexity and specificity coefficients are separate because simpler concepts are not always desirable when the target is known to be compositional. All coefficients are selected on validation episodes only.

The score is converted into a truncated posterior with softmax normalization. Query predictions use the executable semantics of each retained concept. This allows us to evaluate exact concept recovery, query accuracy, negative log-likelihood, Brier score, and risk-coverage behavior.

### 3.4 Search policies and controls

We compare closure construction, random neighborhood search, learned neighborhood ranking, greedy evidence search, beam search, and an A*-style baseline. The learned controller uses histogram gradient boosting over candidate features, including evidence errors, concept size, attribute identity, trajectory frequency, confidence, and whether a candidate was generated. Logistic-regression and MLP controllers provide estimator controls. Feature ablations remove either evidence or trajectory features.

A separate full-space ranker scores every legal concept without search. This control tests whether iterative construction is necessary. Conversely, direct evidence selection over the initial pool tests whether neighborhood expansion contributes beyond the states already present in the traces.

### 3.5 Symbolic reference

The symbolic baseline enumerates all legal concepts for the applicable split and scores them directly from the support labels. It has no denoising model and no candidate-recall bottleneck. It is a reference for the available evidence, not a neural method with identical computation. On compositional splits, its candidate space includes triples even though the denoising model was trained only on atoms and pairs.

## 4. Experimental Protocol

### 4.1 Data separation

The base library contains 13 attributes and 203 legal concepts of size at most three. Training uses atoms and 80% of pairs. Validation uses a disjoint 10% of pairs and 16 triples. Testing uses held-out pairs and the remaining 112 triples. Model selection and memory tuning use validation episodes only. Main test episodes use fresh random seeds.

The relational suite uses predicates over pairs of objects, including same-color, same-shape, size comparison, and texture relations. The advanced suite adds negation, disjunction, XOR, implication, threshold, and transitive-relation tasks. The large suite expands to 32 attributes, eight families, and 4,064 legal conjunctions. The natural suite maps the same underlying concepts to history, traffic, organization, device, and abstract textual templates, with disjoint template assignments across training, validation, and test conditions.

### 4.2 External models and optimization

The primary external backbone is Open-dCoder-0.5B. We apply masked-denoising post-training for 800 optimizer steps with gradient accumulation, mixed precision, and gradient checkpointing. Validation query and concept accuracy select the best checkpoint. Each task family has independently tuned memory hyperparameters and an independently fitted search policy. Test-episode generation is separated from both procedures.

A second backbone, nanoDiff, uses a time-free bidirectional Transformer. Its downloaded checkpoint contains 203,318,272 total parameters despite its 150M package label. The adapter implements actual block-level gradient checkpointing; output, loss, and all parameter gradients were checked against the non-checkpointed path on a small model.

### 4.3 Decoder controls

The autoregressive control uses causal attention and left-to-right commitment over the seven output positions. Logits from position \(i-1\) predict the token at position \(i\). With 14 samples and three support prefixes, it uses 294 forward passes per episode, close to the 288 used by the main dLLM collector. The one-step masked-LM control samples all seven positions independently in parallel and uses 18 forward passes per episode. We report both accuracy and wall-clock cost because these controls differ in samples per episode as well as iterative structure.

### 4.4 Statistical analysis

We report per-episode bootstrap intervals and bootstrap intervals clustered by the true concept, avoiding the assumption that repeated episodes for the same concept are fully independent. Multi-seed summaries report the mean and standard deviation over seed-level means. Paired comparisons use the same test episodes for all methods. A retained-candidate-matched random baseline and a larger random budget are both included.

## 5. Completed Empirical Findings

### 5.1 Base and compositional concepts

On three external-model seeds, learned constructive search achieves unseen-triple concept accuracy of **0.8802 ± 0.0316**, query accuracy of **0.9238 ± 0.0202**, and query NLL of **0.2859 ± 0.0194**. Closure construction obtains 0.7122 concept accuracy; matched random search obtains 0.5638; 96-candidate random search obtains 0.7904; and symbolic enumeration obtains 0.9336.

The results show that neighborhood construction and evidence acceptance recover many triples absent from training. They do not show superiority to symbolic induction. The remaining gap to symbolic enumeration and the variation across seeds are substantive limitations.

### 5.2 Relational concepts

Across seeds 17, 29, and 43, learned constructive search obtains unseen-relational composition accuracy of **0.6836 ± 0.0117**, query accuracy of **0.8639 ± 0.0025**, and query NLL of **0.3217 ± 0.0049**. Closure construction, matched random search, 96-candidate random search, and symbolic enumeration obtain 0.6693, 0.2578, 0.5742, and 0.8164 concept accuracy, respectively.

The neural one-shot oracle decoder has compositional query accuracy near 0.5293, close to chance. Thus, accurate executable memory should not be inferred from direct neural query predictions.

### 5.3 Construction is not equivalent to learned ranking

The full-space learned ranker obtains only 0.0664 concept accuracy on external compositional episodes, despite seeing every legal candidate. Search-based evidence selection is therefore not replaceable by the tested learned full-space ranker. However, direct ranking is stronger than search on some seen and unseen pair splits: direct full-space ranking reaches 0.9154 on seen and 0.8672 on unseen pairs, compared with 0.7904 and 0.8086 for learned search. We therefore do not claim that search is uniformly superior.

### 5.4 Classical search and feature ablations

Greedy and beam search both reach approximately 0.9010 unseen-triple concept accuracy, exceeding the learned controller's 0.8802. On relational composition they reach approximately 0.725, again exceeding learned search. The learned controller's advantage is principally lower query NLL, not higher exact recovery.

Removing evidence features causes a large degradation, whereas removing trajectory features does not. Removing merge operations has almost no effect. These results reject a simple narrative in which learned temporal dynamics are necessary. The supported mechanism is evidence-guided construction; trajectory features may still provide useful candidate hypotheses, but their independent contribution to controller decisions is not established.

### 5.5 Cost accounting

For external compositional episodes, learned search retains an average of 29.33 candidates, generates approximately 54.20 distinct neighbors, scores approximately 56.16 candidate instances with the policy, and evidence-scores 29.33 retained candidates. Reporting only the retained budget would materially understate search computation. All subsequent experiments retain these separate diagnostics.

### 5.6 Difficulty by concept size

Concept-size analyses are computed within each split rather than averaging across different split distributions. On seen base episodes, learned search recovers 0.6933 of atoms and 0.8139 of pairs, while 96-candidate random search recovers 0.7400 and 0.8252. On seen relational episodes, learned search recovers 0.4157 of atoms and 0.5595 of pairs, compared with 0.4784 and 0.6335 for random search. Thus, learned search is not uniformly better as concepts become larger; its clearest advantage is restricted to the more difficult compositional settings. Figures and concept-level values are generated directly from the per-episode outputs.

### 5.7 New operators expose a clear failure boundary

The corrected nine-split advanced evaluation uses 256 fresh episodes per split. Learned-search concept recovery is 0.2344, 0.4063, and 0.4805 on seen, unseen, and compositional concepts, respectively. On compositional episodes, closure search is substantially stronger at 0.7109, and symbolic enumeration reaches 0.8555. For negation, disjunction, XOR, implication, threshold, and transitive relations, learned search recovers only 0.1875, 0.1172, 0.1914, 0.1992, 0.1641, and 0.1367, respectively. The corresponding symbolic results are 0.4180, 0.3633, 0.4805, 0.7070, 0.4805, and 0.6055. Learned search improves query accuracy over independent final-state predictions, but the one-shot neural oracle remains near chance (0.5078–0.5352 across splits). These results do not support a claim of robust new-operator induction.

This evaluation also explains why candidate coverage alone is insufficient: the learned search's novel-candidate rate is high on the new operator splits, yet exact recovery remains low. Training validation shows zero direct concept accuracy at every checkpoint, with the selected checkpoint at step 300. The advanced suite is therefore a negative generalization result, not evidence that additional symbolic operators have been mastered.

The audit uses exactly the same split-specific legal libraries as the main evaluation, with 25, 3, and 27 candidates on seen, unseen, and compositional splits. Full-space learned ranking reaches 0.7813, 1.0000, and 0.8086 on these splits, respectively. Across negation, disjunction, XOR, implication, threshold, and transitive splits, it reaches 0.3750, 0.2031, 0.3516, 0.4727, 0.3516, and 0.4414. Direct ranking is therefore stronger than learned search in several advanced conditions, although it generally remains below symbolic enumeration; constructive search is not universally superior to direct ranking.

In the 17-method ablation, compositional greedy and beam search reach 0.4922 and 0.5000, and the MLP controller reaches 0.5742, all above learned search at 0.4805 but below closure, full-space ranking, and symbolic enumeration. Removing evidence features reduces compositional recovery to 0.2031 but increases seen recovery to 0.5625; removing trajectory features yields 0.5508 on unseen episodes, also above the original learned search. Atom-only initialization scores zero on compositional episodes but 0.5352 and 0.6641 on seen and unseen episodes. Feature and initialization effects are therefore task-dependent rather than universal.

### 5.8 Multi-template semantic evaluation

The natural-semantic suite evaluates history, traffic, organization, device, and abstract phrasings with disjoint template assignments. On 256 fresh episodes per split, learned search obtains concept accuracies of 0.7930, 0.2383, and 0.3945 on seen, unseen-template, and compositional splits, respectively; symbolic enumeration obtains 0.7930, 0.7930, and 0.9336. The corresponding learned-search query accuracies are 0.7500, 0.5918, and 0.8662. The selected learned budget is 96 candidates, so the matched and 96-candidate random controls coincide in this experiment. These results demonstrate some transfer of constructive memory across textual templates, but the large unseen-template gap and single-seed protocol preclude a claim of robust natural-language semantic induction.

### 5.9 Label noise sharply limits exact recovery

We re-collected training, validation, and test trajectories under 10% and 25% support-label flips while keeping query labels clean, independently tuning memory formation in each condition. With 10% noise, learned-search concept accuracies on seen, unseen, and compositional splits are 0.6211, 0.5977, and 0.6484; the corresponding query accuracies are 0.8125, 0.7930, and 0.8701. Symbolic enumeration obtains 0.6094, 0.6094, and 0.6719 concept accuracy. Greedy and beam search both reach 0.6758 on the compositional split, above learned search.

At 25% noise, learned-search concept accuracy falls to 0.2539, 0.2773, and 0.2773 on the three splits. On compositional episodes, closure and symbolic enumeration reach 0.3164 and 0.3359, respectively. Learned-search query accuracy remains higher than the symbolic posterior in this condition, but this does not imply reliable recovery of the clean latent mechanism. These single-seed results identify a substantial robustness limitation rather than evidence of noise immunity. Noisy support observations and clean query labels measure different quantities, so acceptance disagreement with noisy examples must not itself be interpreted as a memory error.

### 5.10 Limited support and underdetermination

With only two support experiences, learned-search concept accuracies are 0.0391, 0.0859, and 0.3008 on seen, unseen, and compositional splits, respectively; query accuracies are 0.6592, 0.6885, and 0.7822. Symbolic enumeration obtains zero exact recovery on seen and unseen pairs and 0.2929 on compositional episodes. The zero pair recovery reflects that two observations generally do not identify a unique legal concept, rather than a symbolic implementation failure.

With four support experiences, learned-search concept recovery improves to 0.3789, 0.4219, and 0.6953, with query accuracies of 0.7500, 0.7578, and 0.8662. Symbolic concept accuracies are 0.3750, 0.4023, and 0.7188. On four-support compositional episodes, learned search is slightly above greedy and beam search (0.6797 and 0.6836), but remains below symbolic enumeration.

With sixteen support experiences, learned search reaches 0.9727, 0.9922, and 0.9687 concept accuracy, with query accuracies of 0.8418, 0.8486, and 0.9434. The corresponding symbolic concept accuracies are 0.9687, 0.9570, and 0.6523. This is the strongest single-seed support condition, but the symbolic MAP result depends on its independently tuned scoring rule and must not be interpreted as an information-theoretic upper bound.

With 32 support experiences, learned-search concept accuracies are 0.9961, 0.9883, and 0.9258, with query accuracies of 0.9404, 0.9316, and 0.9717. Symbolic enumeration reaches 1.0000, 1.0000, and 0.9531 concept accuracy. Thus, more support improves the main method but does not guarantee superiority to symbolic induction. Overall, these results show a strong support-size effect, not universal superiority of learned search. Under limited support, posterior predictions, calibration, and abstention must be interpreted separately from unique-mechanism recovery; query gains do not establish that the true concept has been identified.

### 5.11 Decoder controls

The autoregressive control completed all 256 episodes per split but produced no legal non-symbolic concept: its concept accuracy is zero and query accuracy is 0.5 throughout, using 294 forward passes per episode. This is a failure of the current weights and left-to-right commitment protocol, not evidence that autoregressive models cannot perform induction in general.

The one-step masked-LM control uses 18 forward passes per episode. Its evidence-guided concept accuracies are 0.6016, 0.4961, and 0.3672 on seen, unseen, and compositional splits, with query accuracies of 0.8525, 0.8545, and 0.8662. The corresponding main dLLM learned-search results are 0.7773, 0.7891, and 0.9141 concept accuracy, using 288 forward passes. The main dLLM and MLM use six samples, whereas AR uses 14, so architecture, sampling, and iterative revision are not fully matched and their effects cannot be identified from this comparison alone.

### 5.12 A second backend reveals a different ranking bottleneck

The nanoDiff checkpoint contains 203,318,272 parameters and completed the same 800-step task-specific post-training protocol. Validation selected step 500, with direct concept accuracy of zero and query accuracy of 0.5625. Training took 558.44 seconds and peaked at approximately 1.85 GiB. All three fresh test splits contain 256 episodes, with 288 trajectory forward passes per episode.

| Split | Random concept | Learned-search concept | Learned-search query | Full-ranker concept | Full-ranker query | Symbolic concept | Symbolic query |
|---|---:|---:|---:|---:|---:|---:|---:|
| seen | 0.5781 | 0.6719 | 0.5918 | **0.9102** | **0.9600** | 0.7930 | 0.7441 |
| unseen | 0.5156 | 0.5664 | 0.6152 | 0.7734 | **0.9414** | **0.7930** | 0.7715 |
| compositional | 0.0352 | **0.1250** | 0.7637 | 0.1016 | 0.8418 | **0.9336** | **0.8867** |

The learned and random budgets are both 96, so the matched-random and random controls coincide. Final-state and evidence-guided recovery are near zero on every split. Learned search raises candidate recall to 0.9531, 0.9648, and 0.9844, respectively. In particular, the compositional initial pool contains no true candidate, whereas construction covers 98.44% of episodes but accepts the exact concept in only 12.50%. Candidate coverage and acceptance quality must therefore be reported separately.

Unlike the primary Open-dCoder experiment, full-space learned ranking is much stronger than evidence-guided constructive search on seen and unseen nanoDiff episodes, and policy-only ranking also outperforms evidence acceptance on identical constructed candidates. This backend-dependent reversal rejects a universal claim that evidence scoring is superior to learned ranking. Symbolic enumeration remains far stronger on compositional exact recovery. The neural oracle query accuracies are only 0.5693, 0.5371, and 0.5566, again separating executable-program predictions from direct neural decoding. These are single-seed results and do not establish a general advantage of either backbone or controller.

The full 17-method ablation adds another boundary. Atom-only learned search obtains 0.6875, 0.5898, and 0.1289 on the three splits, slightly exceeding trace-initialized learned search on each. Thus, this backend does not establish that neural initial candidates are necessary, despite their low initial recall. On compositional episodes, learned search exceeds greedy and beam search (0.1055 and 0.1094), but on seen episodes both classical methods reach 0.6836, above learned search. Removing evidence features reduces compositional recovery to 0.0469; removing trajectory features and merge operations yields 0.1211 and 0.1211. Evidence features matter in this run, whereas the latter two components show no independent benefit.

### 5.13 Scaling to 4,064 legal concepts

The large suite completed 800 post-training steps, selecting step 700 with validation direct-concept accuracy 0.25 and query accuracy 0.7344. Training took 2,607.71 seconds and peaked at approximately 5.56 GiB. The learned search budget is 48 candidates and the larger random budget is 96; every fresh test split contains 256 episodes.

| Split | Random 48 | Random 96 | Closure | Learned search | Symbolic | Full ranker |
|---|---:|---:|---:|---:|---:|---:|
| seen | 0.4805 | 0.5234 | 0.5313 | 0.7695 | **0.7930** | **0.8984** |
| unseen | 0.4414 | 0.5313 | 0.5352 | **0.7852** | 0.7813 | **0.8945** |
| compositional | 0.2305 | 0.3086 | 0.4180 | **0.6563** | **0.8438** | 0.0078 |

Learned-search query accuracies are 0.7100, 0.7188, and 0.7783, with NLLs 0.5271, 0.5128, and 0.4397. Candidate recall is 0.9219, 0.9453, and 0.7188. Thus, learned search approaches symbolic exact recovery on seen and unseen pairs and substantially outperforms both random controls, but a sizable compositional gap remains. Evidence-guided selection has higher query accuracy than learned search on seen and unseen pairs, so the method is not uniformly better on every metric.

Full-space learned ranking is strongest on seen and unseen pairs but collapses to 0.0078 exact recovery on compositional episodes. Its 0.8359 compositional query accuracy despite near-zero exact recovery further demonstrates why query prediction and true-mechanism recovery must be distinguished. Neural oracle query accuracies are 0.6357, 0.6426, and 0.6523. Constructive search with evidence acceptance remains necessary for the tested compositional setting, although symbolic enumeration is substantially stronger and all results are single-seed.

The 17-method ablation again favors classical search over the learned controller on compositional episodes: greedy and beam search both reach 0.7500, compared with 0.6563 for learned search, 0.7148 for the MLP controller, and 0.6719 for logistic regression. Removing evidence features reduces learned search to 0.1992, whereas removing trajectory features yields 0.6914 and removing merge operations leaves 0.6563. Atom-only initialization fails on compositional episodes but reaches 0.8906 on both seen and unseen pairs. Neural trace initialization therefore contributes primarily to novel composition, not to already seen mechanisms, and the learned controller is not necessary for the best measured result.

## 6. Related Work

### 6.1 Denoising-time signals

[Time Is a Feature](https://proceedings.iclr.cc/paper_files/paper/2026/file/76931eaba1fcb55b70cde7d0de0161ef-Paper-Conference.pdf) identifies temporal oscillation and uses temporal self-consistency voting and semantic-entropy reinforcement. [TraceLock](https://arxiv.org/abs/2605.24697) learns token commitment from future trace stability. [Trajectory Self-Feedback in Diffusion Language Models](https://aclanthology.org/2026.findings-emnlp.1739/) reviews 24 studies and emphasizes unresolved faithfulness and knowledge-intensive evaluation. [DeMTS](https://arxiv.org/abs/2608.14632) models token-step trajectories for hallucination detection. These works concern final generation, commitment, training rewards, or hallucination classification rather than executable experience-to-query concept formation.

### 6.2 Memory and abstraction

[MetaState](https://arxiv.org/abs/2603.01331) maintains continuous working memory across denoising steps within a generation. Our discrete memories instead persist beyond individual denoising trajectories within an episode and are checked against held-out experiences. [Language Models Need Sleep](https://arxiv.org/abs/2606.03979) studies parameter-level consolidation and self-generated training data. We do not update weights during the offline memory-forming stage.

[Notes to Self](https://arxiv.org/abs/2607.20372) extracts natural-language experiential abstractions from reasoning traces, and [TheoryCoder-2](https://arxiv.org/abs/2602.00929) learns reusable abstractions for hierarchical program-synthesis agents. Consequently, experiential abstraction and program reuse are not new by themselves. Our narrower contribution is the controlled separation of denoising-derived candidate construction, executable symbolic neighborhoods, and held-out evidence acceptance, including direct comparisons against random, learned, classical, and fully symbolic alternatives.

## 7. Limitations and Scope

All current environments are synthetic and have known executable semantics. Template diversity is not equivalent to unrestricted natural-language understanding. Exact concept recovery depends on the candidate language and legal-family constraints. Symbolic references can exploit the same semantics without any diffusion model. Noise can make the observed labels inconsistent with the true concept, so acceptance error under noise is not itself evidence of hallucination.

The experiments do not establish biological dreaming, human-like sleep, unrestricted causal discovery, or permanent weight-level memory consolidation. They also do not show that learned controllers are necessary, that trajectory order is uniquely useful, or that masked diffusion universally outperforms autoregressive modeling. Claims are restricted to measured candidate recovery, query prediction, calibration, and compute under the disclosed synthetic protocols.

## Reproducibility

The repository contains configuration files, model adapters, complete per-episode predictions, search diagnostics, validation histories, bootstrap intervals, concept-clustered intervals, paired comparisons, and ablation outputs. Commands and links to result artifacts are documented in `README.md`. The final machine audit passes all 18 runs (16 constructive experiments plus AR and MLM controls), covering 208,896 primary per-episode result rows; see `outputs/final_validation/validation.json` and `outputs/final_statistics/paper_tables.md`.
