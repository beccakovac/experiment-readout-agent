"""
Experiment Readout & Test Backlog Agent
Built by Rebecca Kovacevich - beccakovac.com

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

import ai_readout
import stats_engine as se

APP_DIR = Path(__file__).parent
SAMPLE_CSV = APP_DIR / "sample_data" / "sample_tests.csv"
MAX_AI_RUNS_PER_SESSION = 5

SAMPLE_CONTEXT = {
    "company": "Acme Analytics (demo)",
    "product": "B2B SaaS analytics platform with a free trial and a sales-assisted demo path",
    "goal": "Grow qualified pipeline and paid conversions in Q1",
    "audience": "Marketing and RevOps leaders at mid-size companies",
}

st.set_page_config(
    page_title="Experiment Readout Agent | Becca Kovacevich",
    page_icon="🧪",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:  # no secrets file locally
        value = ""
    return value or os.environ.get(name, default)


API_KEY = get_secret("ANTHROPIC_API_KEY")
MODEL = get_secret("ANTHROPIC_MODEL", ai_readout.DEFAULT_MODEL)


def init_state():
    defaults = {
        "rows": None,
        "source": None,
        "ctx_company": "",
        "ctx_product": "",
        "ctx_goal": "",
        "ctx_audience": "",
        "ai_cache": {},
        "ai_runs": 0,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def load_sample():
    df = pd.read_csv(SAMPLE_CSV)
    st.session_state.rows = df.to_dict("records")
    st.session_state.source = "sample"
    for key, val in SAMPLE_CONTEXT.items():
        st.session_state[f"ctx_{key}"] = val


def template_csv() -> bytes:
    return pd.read_csv(SAMPLE_CSV).head(2).to_csv(index=False).encode()


def fmt_lift(v):
    return "n/a" if v is None else f"{v:+.1%}"


def fmt_ci(r: dict) -> str:
    if r["lift_ci_low"] is None:
        return "n/a"
    return f"{r['lift_ci_low']:+.1%} to {r['lift_ci_high']:+.1%}"


def cache_key(results: list[dict], context: dict) -> str:
    raw = json.dumps({"r": results, "c": context, "m": MODEL}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


init_state()

# ---------------------------------------------------------------------------
# Sidebar: settings + about
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Test settings")
    confidence = st.select_slider(
        "Confidence level", options=[0.90, 0.95, 0.99], value=0.95,
        format_func=lambda x: f"{x:.0%}",
        help="How sure you need to be before calling a winner. 95% is the industry standard.",
    )
    power = st.select_slider(
        "Statistical power", options=[0.80, 0.90], value=0.80,
        format_func=lambda x: f"{x:.0%}",
        help="Chance of detecting a real effect if one exists. 80% is standard.",
    )
    mde = st.slider(
        "Smallest lift worth detecting", min_value=5, max_value=30, value=10, step=1, format="%d%%",
        help="The minimum relative lift that would matter to the business. "
             "Used to judge whether a test had enough traffic.",
    ) / 100

    st.divider()
    st.subheader("How it works")
    st.markdown(
        "1. **Stats engine** (Python) checks the traffic split, significance, "
        "confidence interval and statistical power for each test. The math never "
        "comes from the AI.\n"
        "2. **AI analyst** (Claude) turns the results into a plain-English "
        "readout and a next-test backlog scored by **ICE** (Impact, Confidence, Ease)."
    )
    st.divider()
    st.markdown(
        "Built by **[Rebecca Kovacevich](https://www.beccakovac.com)**  \n"
        "Growth & Integrated Marketing Leader  \n"
        "[LinkedIn](https://www.linkedin.com/in/rebeccakovacevich/) · "
        "[More AI projects](https://www.beccakovac.com)"
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🧪 Experiment Readout Agent")
st.markdown(
    "#### Turn raw A/B test results into a decision-ready readout and a prioritized next-test backlog."
)
st.caption(
    "Catches the mistakes that cost teams money: calling winners too early, "
    "trusting broken traffic splits, and ignoring tests that were too small to tell."
)

# ---------------------------------------------------------------------------
# Step 1: Data
# ---------------------------------------------------------------------------

st.subheader("1 · Add your test results")
tab_sample, tab_upload, tab_manual = st.tabs(["✨ Try sample data", "📄 Upload CSV", "✏️ Enter one test"])

with tab_sample:
    st.write(
        "Five realistic tests from a demo SaaS company. Each one has a known "
        "right answer: a winner, a loser, a flat result, a test too small to "
        "call, and one with a broken traffic split."
    )
    if st.button("Load sample tests", type="primary"):
        load_sample()

with tab_upload:
    st.write(
        "One row per test. Required columns: `test_name`, `control_visitors`, "
        "`control_conversions`, `variant_visitors`, `variant_conversions`. "
        "Optional: `page`, `hypothesis`, `metric`."
    )
    st.download_button("Download CSV template", template_csv(), "experiment_template.csv", "text/csv")
    uploaded = st.file_uploader("Upload your results", type=["csv"])
    upload_id = getattr(uploaded, "file_id", None) or (uploaded and f"{uploaded.name}-{uploaded.size}")
    if uploaded is not None and st.session_state.get("last_upload") != upload_id:
        st.session_state.last_upload = upload_id
        try:
            df = pd.read_csv(uploaded)
            st.session_state.rows = df.to_dict("records")
            st.session_state.source = "upload"
        except Exception:
            st.error("Couldn't read that file. Make sure it's a CSV saved with commas.")

with tab_manual:
    with st.form("manual"):
        name = st.text_input("Test name", "My test")
        c1, c2 = st.columns(2)
        page = c1.text_input("Page or channel", "")
        metric = c2.text_input("Metric", "Conversion rate")
        hypothesis = st.text_area(
            "Hypothesis (optional)", "",
            placeholder="If we..., then... will improve, because...",
        )
        a, b, c, d = st.columns(4)
        cv = a.number_input("Control visitors", min_value=1, value=5000, step=100)
        cc = b.number_input("Control conversions", min_value=0, value=250, step=1)
        vv = c.number_input("Variant visitors", min_value=1, value=5000, step=100)
        vc = d.number_input("Variant conversions", min_value=0, value=290, step=1)
        if st.form_submit_button("Analyze this test", type="primary"):
            st.session_state.rows = [{
                "test_name": name, "page": page, "metric": metric, "hypothesis": hypothesis,
                "control_visitors": cv, "control_conversions": cc,
                "variant_visitors": vv, "variant_conversions": vc,
            }]
            st.session_state.source = "manual"

if not st.session_state.rows:
    st.info("👆 Load the sample tests to see it in action, or add your own results.")
    st.stop()

tests, input_errors = se.parse_rows(st.session_state.rows)
for err in input_errors:
    st.warning(err)
if not tests:
    st.stop()

results = [
    se.analyze_test(t, confidence=confidence, power=power, mde_relative=mde).to_dict()
    for t in tests
]

# ---------------------------------------------------------------------------
# Step 2: Statistical results
# ---------------------------------------------------------------------------

st.divider()
st.subheader("2 · Statistical verdicts")

counts = {v: sum(r["verdict"] == v for r in results) for v in se.VERDICT_EMOJI}
cols = st.columns(len(counts))
labels = {
    se.WINNER: "Winners", se.LOSER: "Losers", se.FLAT: "Flat",
    se.UNDERPOWERED: "Too small to call", se.INVALID: "Invalid setup",
}
for col, (verdict, n) in zip(cols, counts.items()):
    col.metric(f"{se.VERDICT_EMOJI[verdict]} {labels[verdict]}", n)

table = pd.DataFrame([{
    "Test": r["test_name"],
    "Verdict": f"{se.VERDICT_EMOJI[r['verdict']]} {r['verdict']}",
    "Control rate": f"{r['control_rate']:.2%}",
    "Variant rate": f"{r['variant_rate']:.2%}",
    "Lift": fmt_lift(r["relative_lift"]),
    f"{confidence:.0%} range for lift": fmt_ci(r),
    "p-value": f"{r['p_value']:.3f}",
    "Visitors / group": f"{min(r['control_visitors'], r['variant_visitors']):,}",
    "Needed / group": f"{r['required_visitors_per_arm']:,}" if r["required_visitors_per_arm"] else "n/a",
} for r in results])
st.dataframe(table, hide_index=True, width="stretch")

for r in results:
    with st.expander(f"{se.VERDICT_EMOJI[r['verdict']]} {r['test_name']} — {r['verdict']}"):
        if r["hypothesis"]:
            st.markdown(f"**Hypothesis:** {r['hypothesis']}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Control", f"{r['control_rate']:.2%}", help=f"{r['control_conversions']:,} / {r['control_visitors']:,}")
        m2.metric(
            "Variant", f"{r['variant_rate']:.2%}",
            delta=fmt_lift(r["relative_lift"]) if r["relative_lift"] is not None else None,
            help=f"{r['variant_conversions']:,} / {r['variant_visitors']:,}",
        )
        m3.metric("p-value", f"{r['p_value']:.3f}")
        st.write(r["verdict_reason"])
        for w in r["warnings"]:
            st.caption(f"⚠️ {w}")

# ---------------------------------------------------------------------------
# Step 3: AI readout + backlog
# ---------------------------------------------------------------------------

st.divider()
st.subheader("3 · AI readout & next-test backlog")
st.write("A little context helps the AI write a sharper readout and better test ideas.")

c1, c2 = st.columns(2)
c1.text_input("Company or brand", key="ctx_company")
c2.text_input("Primary business goal", key="ctx_goal",
              placeholder="e.g. Grow paid conversions 20% this quarter")
c1.text_input("What you sell", key="ctx_product",
              placeholder="e.g. B2B SaaS with free trial")
c2.text_input("Target audience", key="ctx_audience",
              placeholder="e.g. Marketing leaders at mid-size companies")
ctx = {k: st.session_state[f"ctx_{k}"].strip() for k in ("company", "product", "goal", "audience")}

key = cache_key(results, ctx)
ai_result = st.session_state.ai_cache.get(key)

if not API_KEY:
    st.info(
        "The AI readout needs an Anthropic API key. Add `ANTHROPIC_API_KEY` to "
        "Streamlit secrets (see README). The statistics above work without it."
    )
elif ai_result is None:
    runs_left = MAX_AI_RUNS_PER_SESSION - st.session_state.ai_runs
    if runs_left <= 0:
        st.warning("You've reached the demo limit for AI readouts in this session. Refresh to start over.")
    elif st.button("🤖 Generate readout & backlog", type="primary"):
        with st.spinner("Analyzing results and prioritizing next tests... (about 30 seconds)"):
            try:
                ai_result = ai_readout.generate_readout(results, ctx, API_KEY, MODEL)
                st.session_state.ai_cache[key] = ai_result
                st.session_state.ai_runs += 1
            except Exception as exc:  # show a friendly error, not a stack trace
                st.error("The AI readout couldn't be generated right now. Please try again in a minute.")
                with st.expander("Technical details"):
                    st.code(str(exc))

if ai_result:
    with st.container(border=True):
        st.markdown("#### 📋 Executive summary")
        st.write(ai_result["executive_summary"])

    st.markdown("#### Test-by-test readout")
    by_name = {r["test_name"]: r for r in results}
    for t in ai_result.get("test_readouts", []):
        r = by_name.get(t["test_name"])
        badge = se.VERDICT_EMOJI.get(r["verdict"], "") if r else ""
        with st.container(border=True):
            st.markdown(f"**{badge} {t['test_name']}**")
            st.markdown(f"**What happened:** {t['what_happened']}")
            st.markdown(f"**What it means:** {t['what_it_means']}")
            st.markdown(f"**Next step:** {t['recommended_action']}")

    st.markdown("#### 💡 Key learnings")
    for k in ai_result.get("key_learnings", []):
        st.markdown(f"- {k}")

    st.markdown("#### 🗂️ Next-test backlog, ranked by ICE")
    backlog_df = pd.DataFrame([{
        "Rank": i,
        "Test idea": b["idea"],
        "Where": b["page_or_channel"],
        "Hypothesis": b["hypothesis"],
        "Builds on": b["based_on"],
        "Metric": b["primary_metric"],
        "Impact": b["impact"],
        "Confidence": b["confidence"],
        "Ease": b["ease"],
        "ICE": b["ice"],
        "Why these scores": b["scoring_rationale"],
    } for i, b in enumerate(ai_result.get("backlog", []), 1)])
    st.dataframe(
        backlog_df, hide_index=True, width="stretch",
        column_config={
            "ICE": st.column_config.ProgressColumn("ICE", min_value=0, max_value=10, format="%.1f"),
            "Hypothesis": st.column_config.TextColumn(width="large"),
        },
    )

    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ Download full readout (Markdown)",
        ai_readout.readout_to_markdown(results, ai_result, ctx),
        "experiment_readout.md", "text/markdown",
    )
    d2.download_button(
        "⬇️ Download backlog (CSV)",
        backlog_df.to_csv(index=False).encode(),
        "test_backlog.csv", "text/csv",
    )
else:
    st.download_button(
        "⬇️ Download statistical results (Markdown)",
        ai_readout.readout_to_markdown(results, None, ctx),
        "experiment_results.md", "text/markdown",
    )

st.divider()
st.caption(
    "Method: two-sided two-proportion z-test; relative-lift confidence interval via the "
    "delta method; sample ratio mismatch via chi-square (flagged at p < 0.001); sample "
    "size for the chosen minimum lift, confidence and power. Assumes a planned 50/50 split."
)
