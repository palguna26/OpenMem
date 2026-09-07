from src.retrieval.retrieval import AtomHit, pack_atoms_token_aware


def test_atom_packing_preserves_retrieval_order():
    hits = [
        AtomHit("best", "answer-session", "The answer is SQLite.", "2020-01-01", "user", 0.99),
        AtomHit("older", "older-session", "Distracting historical detail.", "2019-01-01", "user", 0.50),
    ]

    packed = pack_atoms_token_aware(hits, token_budget=120)

    assert packed["text"].index("answer-session") < packed["text"].index("older-session")
    assert "The answer is SQLite." in packed["text"]
