# Reproduction scope

This release contains Table 1 cross-tokenizer training and the
[portable PHYRD-40 evaluation workflow](phyrd40-evaluation.md). Large-scale
SFT/MoE training recipes, agentic experiments, Open-MOPD, cluster submitters,
the internal evaluation platform, model weights and historical logs are not included.

The Qwen templates originate from commit
`695cebfed68f5ffea5485fff4e8fa766229e9dc9`. Its BPM and SimCT save interval was 100,
while the report specifies 50; public templates use 50. This changes the future
recipe, not historical execution. GLM templates change teacher/tokenizer and
run/output names while preserving the small-scale optimizer and resource settings;
they are newly prepared templates, not recovered execution manifests.

The reference recipes specify 2,500 rollout batches. The paper's configuration
table does not give the final checkpoint iteration. Exact endpoint, seed and
checkpoint-selection provenance for every reported result is not established by
this export. Defaults are training seed 1234 and rollout seed 42; set them explicitly
for a new comparison.

Runtime requirements:

- Share model/data/code/hidden-state paths across nodes and allow required ports.
- Use a unique run ID and a fresh coordination directory.
- Teacher context defaults to 65,536 because tokenizer lengths can differ; the
  student recipe remains 5,120 prompt / 27,648 response / 33,280 sequence tokens.
- Evaluation is disabled in portable training; export and evaluate checkpoints
  independently. PHYRD-40 generation and rubric grading are available as a
  separate workflow; no best-checkpoint selector is bundled.
- Install the adapted Slime and registered SGLang plugin on relevant nodes.
  Dependency files do not lock the full CUDA, kernel, Megatron or driver stack.

Configuration compilation and dry-runs check command construction. Numerical
checks cover selected objective calculations. Neither is a multi-GPU smoke test or
independent benchmark reproduction. Consult [method adaptations](methods.md)
before describing a baseline as an exact reproduction.
