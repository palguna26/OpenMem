from types import SimpleNamespace

from benchmarks.longmemeval import run_benchmark


def test_judged_abstention_uses_benchmark_literal(monkeypatch):
    calls = []

    def fake_chat(model, messages, budget):
        calls.append(messages)
        return '{"correct": true}'

    monkeypatch.setattr(run_benchmark, "openrouter_chat", fake_chat)
    sample = SimpleNamespace(
        question_id="q_abs",
        question="What is unknown?",
        question_date="",
        answer="",
        unanswerable=False,
    )

    result = run_benchmark.judge_question("answer", "judge", sample, "some context", object())

    assert result["hypothesis"] == "insufficient information"
    assert len(calls) == 1
