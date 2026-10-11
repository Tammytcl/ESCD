# Objectives and baseline adaptations

Source: the authors' `2026-09-26` branch, commit
`695cebfed68f5ffea5485fff4e8fa766229e9dc9`. Later agentic changes are excluded.

## Actual loss dispatch

All templates enable cross-tokenizer training. The backend calls
`compute_cross_tokenizer_opd_token_losses`, then the chosen plugin. The shared
`loss.opd_loss_type: sampled` field does **not** mean ESCD/BPM use sampled-token KL:
the active plugin supplies the dense objective. `candidate_top_k: 0` means full
teacher-vocabulary support. Rollout top-k 20 and X-Token projection top-k 4 serve
different purposes. Frozen teacher hidden states and its LM head provide the
actor-side teacher distributions.

| Method | Released setting | Reference / qualification |
| --- | --- | --- |
| ESCD | Full teacher support; root FKL + completion-set loss; child weight 1; whitespace mask | [Paper](https://arxiv.org/abs/2609.34738). Historical source calls `escd_full_vocab` an ablation; it is the public ESCD template here. |
| BPM | `bpm_paper`, beta 0 (FKL), full support, residual mass, spanning correction | [Paper](https://arxiv.org/abs/2607.22334). One divergence setting, not all paper variants. |
| SimCT | Minimal common-byte spans, shared candidates plus realized-span supervision, reverse KL | [Paper](https://arxiv.org/abs/2605.07711), [source revision](https://github.com/sunjie279/SimCT-/tree/4dae6fcd374290f718ac0a950e57947e292be927). Canonicalization and semantic EOS are explicit in the adapter. |
| GOLD | Exact vocabulary-string match; observed merge; matched FKL + sorted unmatched L1; fixed weights 1/1, CE off, EOS retained | [TRL source](https://github.com/huggingface/trl/blob/6fae07fb651abfc519d8d0f09514e53b87cd95f1/trl/experimental/gold/gold_trainer.py). Configured OPD adaptation, not every upstream default. |
| X-Token | H-KL; fixed top-4 projection; expanded common set; fixed loss weights, CE off | [Paper](https://arxiv.org/abs/2605.21699). GOLD-style byte alignment replaces DP; dynamic KD/CE rescaling is absent. |

Fixed projection is consistent with X-Token **H-KL**. The paper learns its
projection in the main **P-KL** setting; treating a fixed H-KL matrix as an omitted
trainable parameter would be incorrect. An experimental frozen-projection P-KL
path exists but is not selected by these recipes and is not the paper's main
learned P-KL configuration. The current recipe does not strictly reproduce the
entire original X-Token training algorithm.

GOLD retains unnormalized observed-merge probabilities. Reduction averages aligned
groups within examples, then examples; the plugin compensates for the backend's
token denominator. EOS retention, beta 0 and fixed weights are intentional recipe
choices and must be reported when comparing methods.

## Initial numerical audit

Twelve CPU checks cover BPM mass and FKL gradients (including an extreme low-probability
cell), ESCD groups and actual child-scorer gradients, SimCT masking/teacher
detachment, GOLD observed merge/reduction, and X-Token projection on a toy vocabulary. Additional checks cover BPM terminal/truncated responses and the distinction between SimCT score spaces.
They do not prove CP/TP correctness, real-tokenizer coverage, all EOS/truncation
cases or the historical provenance of Table 1 checkpoints.

## SimCT paper versus released-code score space

The pinned upstream `span_ctkd.py` averages **raw logits** for the realized span.
The paper defines averages of **log probabilities**. These are not generally
equivalent when the native normalization constants vary across positions.
This repository exposes both: `simct` selects the paper's log-probability scoring;
`simct_official` selects raw-logit scoring. The Table 1 templates select `simct`.
Do not describe that setting as numerically identical to the pinned upstream code.
