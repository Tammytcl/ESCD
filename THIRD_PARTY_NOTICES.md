# Third-party components

The `slime/` directory contains a locally adapted copy of slime. Its upstream
Apache-2.0 license is retained in [slime/LICENSE](slime/LICENSE). The copy includes
cross-tokenizer integration, model adapters and training-runtime changes; replacing
it with an arbitrary upstream checkout is not equivalent.

The baseline plugins in `opd/cross_tokenizer/` implement BPM, SimCT, GOLD and X-Token
in this shared training runtime. They are not separate upstream trainers.
The local reference snapshots under `third_party/reference/` are audit materials,
not runtime dependencies, and are not included in the training release.

Reference revisions recorded during implementation:

| Method | Reference | Recorded revision |
| --- | --- | --- |
| SimCT | https://github.com/sunjie279/SimCT- | `4dae6fcd374290f718ac0a950e57947e292be927` |
| GOLD | https://github.com/huggingface/trl/tree/main/trl/experimental/gold | `6fae07fb651abfc519d8d0f09514e53b87cd95f1` |

The slime license applies to that component; this notice does not assign a new
license to the project authors' code. Model weights and datasets are not bundled.
See `docs/methods.md` for baseline adaptations and the distinction between SimCT's
paper score space and its pinned released code. Figures in `assets/` and `ESCD.pdf`
are the ESCD authors' report material; they are not upstream Slime assets.
