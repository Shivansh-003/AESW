"""
Memory Module
=============
Evidence representation, node caching, and adaptive exponential decay mechanisms for AESW.
"""

from aesw.memory.types import EvidencePolarity, EvidenceSource
from aesw.memory.decay import (
    exponential_decay,
    compute_half_life,
    decay_rate_from_half_life,
)
from aesw.memory.models import NodeEvidence, WeightedEvidence
from aesw.memory.cache import EvidenceCache

__all__ = [
    "EvidencePolarity",
    "EvidenceSource",
    "exponential_decay",
    "compute_half_life",
    "decay_rate_from_half_life",
    "NodeEvidence",
    "WeightedEvidence",
    "EvidenceCache",
]
