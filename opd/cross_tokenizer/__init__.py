"""Extensible cross-tokenizer OPD algorithms."""

from .base import (
    AlignedSegment,
    AlignmentResult,
    CandidateScores,
    CrossTokenizerAlgorithm,
    CrossTokenizerCapabilities,
    CrossTokenizerAlignmentError,
    FusedScoringContext,
    JSONObject,
    SharedVocabulary,
    StudentScoringContext,
    TeacherScoringContext,
    TensorPayload,
    VocabularyDistribution,
    VocabularyMapping,
    localize_payload_by_student_positions,
    normalize_json_object,
)
from .registry import (
    available_cross_tokenizers,
    build_cross_tokenizer,
    cross_tokenizer_capabilities,
    discover_cross_tokenizers,
    register_cross_tokenizer,
)
from .bpm import BPMCrossTokenizer, BPMPaperCrossTokenizer
from .escd import ESCDCrossTokenizer
from .escd_full_vocab import ESCDFullVocabularyCrossTokenizer
from .gold import GOLDCrossTokenizer, build_gold_shared_vocabulary
from .simct import (
    SimCTCrossTokenizer,
    SimCTOfficialCrossTokenizer,
    SimCTTopKCrossTokenizer,
    build_shared_vocabulary,
)
from .xtoken import XTokenCrossTokenizer

__all__ = [
    "AlignedSegment",
    "AlignmentResult",
    "BPMCrossTokenizer",
    "BPMPaperCrossTokenizer",
    "CandidateScores",
    "CrossTokenizerAlgorithm",
    "CrossTokenizerCapabilities",
    "CrossTokenizerAlignmentError",
    "ESCDCrossTokenizer",
    "ESCDFullVocabularyCrossTokenizer",
    "FusedScoringContext",
    "GOLDCrossTokenizer",
    "JSONObject",
    "SharedVocabulary",
    "SimCTCrossTokenizer",
    "SimCTOfficialCrossTokenizer",
    "SimCTTopKCrossTokenizer",
    "StudentScoringContext",
    "TeacherScoringContext",
    "TensorPayload",
    "VocabularyDistribution",
    "VocabularyMapping",
    "XTokenCrossTokenizer",
    "available_cross_tokenizers",
    "build_cross_tokenizer",
    "cross_tokenizer_capabilities",
    "build_gold_shared_vocabulary",
    "build_shared_vocabulary",
    "discover_cross_tokenizers",
    "normalize_json_object",
    "localize_payload_by_student_positions",
    "register_cross_tokenizer",
]
