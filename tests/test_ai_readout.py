"""
Tests the AI layer with a FAKE Claude response, so no API key or credits are needed.
Checks that the payload is built correctly, ICE scores are computed and
sorted, scores are clamped to 1-10, and the Markdown export works.
"""

import csv
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ai_readout  # noqa: E402
import stats_engine as se  # noqa: E402


def sample_results():
    with open(ROOT / "sample_data" / "sample_tests.csv", newline="") as f:
        tests, _ = se.parse_rows(list(csv.DictReader(f)))
    return [se.analyze_test(t).to_dict() for t in tests]


FAKE_OUTPUT = {
    "executive_summary": "Ship the annual-plan default. Keep the product tour.",
    "test_readouts": [
        {"test_name": "Annual plan pre-selected", "what_happened": "x", "what_it_means": "y", "recommended_action": "Ship"},
    ],
    "key_learnings": ["Visitors respond to visible savings."],
    "backlog": [
        {"idea": "Low", "page_or_channel": "p", "hypothesis": "h", "based_on": "b", "primary_metric": "m",
         "impact": 2, "confidence": 2, "ease": 2, "scoring_rationale": "r"},
        {"idea": "High", "page_or_channel": "p", "hypothesis": "h", "based_on": "b", "primary_metric": "m",
         "impact": 9, "confidence": 8, "ease": 15, "scoring_rationale": "r"},  # 15 should clamp to 10
    ],
}


def install_fake_anthropic(monkeypatch, captured):
    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            block = types.SimpleNamespace(type="tool_use", input=json.loads(json.dumps(FAKE_OUTPUT)))
            return types.SimpleNamespace(content=[block])

    class FakeClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.messages = FakeMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=FakeClient))


def test_generate_readout_sorts_and_clamps(monkeypatch):
    captured = {}
    install_fake_anthropic(monkeypatch, captured)
    out = ai_readout.generate_readout(sample_results(), {"company": "Acme"}, "test-key")

    assert [b["idea"] for b in out["backlog"]] == ["High", "Low"]
    assert out["backlog"][0]["ease"] == 10
    assert out["backlog"][0]["ice"] == 9.0
    assert captured["tool_choice"]["name"] == "submit_experiment_readout"

    payload = captured["messages"][0]["content"]
    assert "Underpowered" in payload and "Invalid" in payload
    assert '"company": "Acme"' in payload


def test_markdown_export():
    results = sample_results()
    md_stats_only = ai_readout.readout_to_markdown(results, None, {"company": "Acme"})
    assert "# Acme - Experiment Readout" in md_stats_only
    assert "Remove phone field" in md_stats_only
    assert "backlog" not in md_stats_only.lower()

    ai = json.loads(json.dumps(FAKE_OUTPUT))
    for b in ai["backlog"]:
        b["ice"] = ai_readout.ice_score(b)
    md_full = ai_readout.readout_to_markdown(results, ai, {"company": "Acme", "goal": "Grow"})
    assert "Executive summary" in md_full and "Next-test backlog" in md_full


def test_payload_is_valid_json():
    payload = ai_readout.build_payload(sample_results(), {"company": "x"})
    data = json.loads(payload)
    assert len(data["tests"]) == 5
