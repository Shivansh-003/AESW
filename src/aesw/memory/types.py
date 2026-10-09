"""
Evidence Memory Types and Enumerations
======================================
Defines strongly typed enumerations for evidence polarity and acquisition source.
"""

from enum import Enum


class EvidencePolarity(str, Enum):
    """Polarity of recorded evidence regarding target presence or absence.

    - POSITIVE: Evidence supporting target presence (sensor detection or physical acquisition).
    - NEGATIVE: Evidence supporting target absence (sensor non-detection or visited empty node).
    """
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class EvidenceSource(str, Enum):
    """Provenance mechanism through which evidence was gathered.

    - DIRECT_VISIT: Physical presence of the walker at the vertex.
    - REMOTE_SENSOR: Sensor observation gathered within signal radius.
    - RECEIVED_EXCHANGE: Inter-walker evidence sharing (prepared for multi-agent exchange).
    """
    DIRECT_VISIT = "DIRECT_VISIT"
    REMOTE_SENSOR = "REMOTE_SENSOR"
    RECEIVED_EXCHANGE = "RECEIVED_EXCHANGE"
