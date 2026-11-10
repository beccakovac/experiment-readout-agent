"""
AI layer for the Experiment Readout agent.

Claude receives the finished statistics (from stats_engine.py) plus the
business context, and returns:
  - an executive summary
  - a plain-English readout for each test
  - key learnings
  - a backlog of next tests, each scored with ICE (Impact, Confidence, Ease)

Claude is told NOT to recalculate anything. It explains and plans;
the stats engine decides the verdicts.
"""

from __future__ import annotations

import json

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 6000

SYSTEM_PROMPT = """You are a senior growth marketing and experimentation lead.
You write test readouts that a CMO can read in two minutes and an analyst would sign off on.

Rules you must follow:
- The statistics and verdicts you receive are final. Never recalculate them and never change a verdict.
- "Underpowered" tests are NOT winners, even if the observed lift is large. Say what it would take to get a real answer.
- "Invalid" tests (sample ratio mismatch) must not be interpreted at all beyond saying the setup is broken; recommend likely causes to check (redirect timing, bot filtering, targeting rules, tracking firing on one variant only, caching).
- "Flat" tests are useful learnings: the change didn't matter enough to move the metric.
- Write in plain English. No jargon without a short explanation. Be specific and concrete, not generic.
- Every backlog idea must connect to a specific result or learning from these tests, or to the business context provided.
- Hypotheses use the format: "If we [change], then [metric] will [improve], because [reason]."
- ICE scores are integers from 1 to 10:
    Impact = how much it could move the primary business goal if it works.
    Confidence = how strong the evidence is that it will work (higher when backed by a winning test or strong learning, lower for pure ideas).
    Ease = how quick and cheap it is to build and run (consider the traffic needed).
- Give 6 to 8 backlog ideas. Include at least one confirmation re-test if any test was underpowered, and a fix-and-rerun if any test was invalid.
"""

READOUT_TOOL = {
    "name": "submit_experiment_readout",
    "description": "Submit the finished experiment readout and prioritized test backlog.",
    "input_schema": {
        "type": "object",
        "properties": {
            "executive_summary": {
                "type": "string",
                "description": "3-5 sentences for a VP/CMO: what we learned overall, what to ship, what to do next.",
            },
            "test_readouts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "test_name": {"type": "string"},
                        "what_happened": {"type": "string", "description": "1-2 sentences, plain English, using the provided numbers."},
                        "what_it_means": {"type": "string", "description": "The customer or business insight - why this probably happened."},
                        "recommended_action": {"type": "string", "description": "One clear next step: ship, roll back, keep running (with how long/how many visitors), or fix and re-run."},
                    },
                    "required": ["test_name", "what_happened", "what_it_means", "recommended_action"],
                },
            },
            "key_learnings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-5 learnings about the customer that carry across tests.",
            },
            "backlog": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "idea": {"type": "string", "description": "Short name for the test."},
                        "page_or_channel": {"type": "string"},
                        "hypothesis": {"type": "string"},
                        "based_on": {"type": "string", "description": "Which test result or learning this builds on."},
                        "primary_metric": {"type": "string"},
                        "impact": {"type": "integer", "minimum": 1, "maximum": 10},
                        "confidence": {"type": "integer", "minimum": 1, "maximum": 10},
                        "ease": {"type": "integer", "minimum": 1, "maximum": 10},
                        "scoring_rationale": {"type": "string", "description": "One sentence explaining the three scores."},
                    },
                    "required": [
                        "idea", "page_or_channel", "hypothesis", "based_on",
                        "primary_metric", "impact", "confidence", "ease", "scoring_rationale",
                    ],
                },
            },
        },
        "required": ["executive_summary", "test_readouts", "key_learnings", "backlog"],
    },
}


def _round(value, digits=4):
    return None if value is None else round(value, digits)


def build_payload(results: list[dict], context: dict) -> str:
    """Compact, rounded version of the stats for the prompt."""
    tests = []
    for r in results:
        tests.append({
            "test_name": r["test_name"],
            "page": r["page"],
            "hypothesis": r["hypothesis"],
            "metric": r["metric"],
            "control": {"visitors": r["control_visitors"], "conversions": r["control_conversions"], "rate": _round(r["control_rate"])},
            "variant": {"visitors": r["variant_visitors"], "conversions": r["variant_conversions"], "rate": _round(r["variant_rate"])},
            "relative_lift": _round(r["relative_lift"]),
            "lift_confidence_interval": [_round(r["lift_ci_low"]), _round(r["lift_ci_high"])],
            "p_value": _round(r["p_value"]),
            "confidence_level": r["confidence_level"],
            "visitors_needed_per_group_for_reliable_result": r["required_visitors_per_arm"],
            "minimum_detectable_lift_used": r["mde_relative"],
            "verdict": r["verdict"],
            "verdict_reason": r["verdict_reason"],
            "warnings": r["warnings"],
        })
    return json.dumps({"business_context": context, "tests": tests}, indent=2)


def ice_score(item: dict) -> float:
    return round((item["impact"] + item["confidence"] + item["ease"]) / 3, 1)


def generate_readout(results: list[dict], context: dict, api_key: str, model: str = DEFAULT_MODEL) -> dict:
    """Call Claude and return the structured readout (backlog sorted by ICE)."""
    import anthropic  # imported here so the stats still work without the package

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=[READOUT_TOOL],
        tool_choice={"type": "tool", "name": READOUT_TOOL["name"]},
        messages=[{
            "role": "user",
            "content": (
                "Write the experiment readout and next-test backlog for these results.\n\n"
                + build_payload(results, context)
            ),
        }],
    )

    data = next(
        (block.input for block in message.content if getattr(block, "type", "") == "tool_use"),
        None,
    )
    if not data:
        raise RuntimeError("The AI didn't return a readout. Please try again.")

    for item in data.get("backlog", []):
        for key in ("impact", "confidence", "ease"):
            item[key] = max(1, min(10, int(item.get(key, 5))))
        item["ice"] = ice_score(item)
    data["backlog"] = sorted(data.get("backlog", []), key=lambda x: x["ice"], reverse=True)
    return data


def readout_to_markdown(results: list[dict], ai: dict | None, context: dict) -> str:
    """Downloadable report a user can paste into a doc, Notion, or Slack."""
    title = context.get("company") or "Experiment"
    lines = [f"# {title} - Experiment Readout", ""]
    if context.get("goal"):
        lines += [f"**Business goal:** {context['goal']}", ""]

    if ai:
        lines += ["## Executive summary", ai["executive_summary"], ""]

    lines += ["## Results", "", "| Test | Control | Variant | Lift | p-value | Verdict |", "|---|---|---|---|---|---|"]
    for r in results:
        lift = f"{r['relative_lift']:+.1%}" if r["relative_lift"] is not None else "n/a"
        lines.append(
            f"| {r['test_name']} | {r['control_rate']:.2%} | {r['variant_rate']:.2%} | {lift} | {r['p_value']:.3f} | {r['verdict']} |"
        )
    lines.append("")

    readouts = {t["test_name"]: t for t in (ai or {}).get("test_readouts", [])}
    for r in results:
        lines += [f"### {r['test_name']} - {r['verdict']}", r["verdict_reason"]]
        t = readouts.get(r["test_name"])
        if t:
            lines += [
                "",
                f"- **What happened:** {t['what_happened']}",
                f"- **What it means:** {t['what_it_means']}",
                f"- **Next step:** {t['recommended_action']}",
            ]
        for w in r["warnings"]:
            lines.append(f"- ⚠️ {w}")
        lines.append("")

    if ai:
        lines += ["## Key learnings"] + [f"- {k}" for k in ai["key_learnings"]] + [""]
        lines += ["## Next-test backlog (ranked by ICE)", "", "| # | Test | Hypothesis | I | C | E | ICE |", "|---|---|---|---|---|---|---|"]
        for i, b in enumerate(ai["backlog"], 1):
            lines.append(
                f"| {i} | {b['idea']} | {b['hypothesis']} | {b['impact']} | {b['confidence']} | {b['ease']} | {b['ice']} |"
            )
        lines.append("")

    lines.append("_Generated by the Experiment Readout Agent - built by Rebecca Kovacevich (beccakovac.com)_")
    return "\n".join(lines)
