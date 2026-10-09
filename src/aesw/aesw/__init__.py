"""
AESW Module
===========

Proposed search algorithm: Adaptive Evidence-Sharing Walkers (AESW).

Three Central Novelty Pillars:
1. Adaptive memory decay dynamically tuned by estimated graph churn.
2. Adaptive local exploitation versus global exploration based on information gain.
3. Asymmetric sharing of positive (target clues) and negative (cleared nodes) evidence.

Key Components:
- Churn Estimator (tracking edge turnover rates)
- Evidence Memory Integration (decaying node cache)
- Node-Mediated Communication Exchange (push, pull, push_pull)
- Mode Controller (local exploitation vs. global exploration)
- Next-Hop Candidate Scorer (incorporating edge availability, evidence, degree)
- Softmax Stochastic Action Selector
"""

from aesw.aesw.churn import ChurnEstimator, DEFAULT_ETA, DEFAULT_EPSILON
from aesw.aesw.communication import (
    CommunicationMode,
    CommunicationEvent,
    CommunicationMetrics,
    SharedNodeCache,
    NodeMediatedExchange,
    validate_communication_mode,
)

__all__: list[str] = [
    "ChurnEstimator",
    "DEFAULT_ETA",
    "DEFAULT_EPSILON",
    "CommunicationMode",
    "CommunicationEvent",
    "CommunicationMetrics",
    "SharedNodeCache",
    "NodeMediatedExchange",
    "validate_communication_mode",
]
