"""Scope checks for Student routing replay with cross-tokenizer supervision."""


def validate_cross_tokenizer_r3(
    *, algorithm: str, teacher_backend: str, teacher_deployment: str, post_process_path: str
) -> None:
    """Admit external BPM/ESCD paths which preserve Student token trajectories.

    Teacher re-tokenization happens in an independent HTTP service. Its routing
    is never replayed. Do not extend this allowlist to trajectory-replacing
    objectives (e.g. ULD teacher-answer CE) without a separate routing contract.
    """
    if not (
        algorithm in {"bpm_paper", "escd_full_vocab"}
        and teacher_backend == "sglang"
        and teacher_deployment == "external_http"
        and post_process_path in {
            "opd.cross_tokenizer.sglang_teacher.install_external_bpm_paper_hidden_payload",
            "opd.cross_tokenizer.sglang_teacher.install_external_causal_hidden_payload",
        }
        and (algorithm == "bpm_paper" or post_process_path.endswith("install_external_causal_hidden_payload"))
    ):
        raise ValueError(
            "Cross-tokenizer R3 currently requires bpm_paper or escd_full_vocab with an external_http "
            "SGLang Teacher and the built-in causal-hidden payload installer; "
            "Student token trajectories must remain unchanged."
        )
