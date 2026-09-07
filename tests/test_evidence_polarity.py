from uuid import uuid4

import pytest

from src.memory.extraction import CandidateRejected, semantic_support, validate_candidate
from src.models import EvidenceSpan, ExtractionCandidate


def _candidate(statement: str, excerpt: str) -> ExtractionCandidate:
    event_id = uuid4()
    return ExtractionCandidate(
        kind="fact",
        subject="Alice",
        statement=statement,
        evidence=[EvidenceSpan(event_id=event_id, start_offset=0, end_offset=len(excerpt), excerpt=excerpt)],
        confidence=1.0,
        durability="session",
    )


def test_semantic_support_does_not_treat_explicit_contradiction_as_support():
    assert semantic_support("Alice likes coffee", "Alice does not like coffee", "Alice")
    candidate = _candidate("Alice likes coffee", "Alice does not like coffee")

    with pytest.raises(CandidateRejected, match="contradictory_evidence_polarity"):
        validate_candidate("n1", candidate, {candidate.evidence[0].event_id: candidate.evidence[0].excerpt}, require_evidence=True)


def test_matching_explicit_negation_remains_valid():
    candidate = _candidate("Alice does not like coffee", "Alice does not like coffee")

    validate_candidate("n1", candidate, {candidate.evidence[0].event_id: candidate.evidence[0].excerpt}, require_evidence=True)
