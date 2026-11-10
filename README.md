# 🧪 Experiment Readout Agent

**Turn raw A/B test results into a decision-ready readout and a prioritized next-test backlog.**

Built by [Rebecca Kovacevich](https://www.beccakovac.com), Growth & Integrated Marketing Leader.

Most teams lose money on experimentation in three ways: they call winners too early, they trust tests with broken traffic splits, and they never turn results into the next round of tests. This agent handles all three.

## What it does

1. **Stats engine (Python).** For each test it checks:
   - **Traffic split (sample ratio mismatch).** Did visitors split the way they were supposed to? If not, the result is thrown out.
   - **Significance.** Two-sided, two-proportion z-test.
   - **Lift range.** Confidence interval for the relative lift.
   - **Statistical power.** Was the test big enough to detect a lift that matters?

   Each test then gets one verdict: 🟢 Winner · 🔴 Loser · ⚪ Flat · 🟡 Underpowered · ⛔ Invalid
2. **AI analyst (Claude).** Writes an executive summary and a plain-English readout for each test, pulls out key customer learnings, and builds a **next-test backlog scored with ICE** (Impact, Confidence, Ease).
3. **Exports.** You can download the full readout as Markdown and the backlog as CSV.

**Design choice:** the AI never does the math. Statistics come from tested Python code, and Claude only explains and plans. That keeps the numbers exact and the verdicts consistent.

**Built with:** Python · Streamlit · Claude API (tool use for structured output) · pytest

---

## Sample data answer key

`sample_data/sample_tests.csv` contains five tests, and each one has a planted right answer. The automated tests check every one.

| Test | Planted situation | Correct verdict |
|---|---|---|
| Annual plan pre-selected | 4.0% → 4.8% on 10k/group | 🟢 Winner (+20%, p ≈ 0.006) |
| New hero headline | 5.00% vs 5.03% on 40k/group | ⚪ Flat (had enough traffic, no effect) |
| Remove phone field | +33% lift, but only ~300 visitors/group | 🟡 Underpowered, *not* a winner |
| Trust badges at checkout | 10,000 vs 8,800 visitors | ⛔ Invalid (sample ratio mismatch) |
| Skip product tour | 12.0% → 10.6% on 8k/group | 🔴 Loser (−11.7%, p ≈ 0.005) |

---

## Run it on your computer

You need Python 3.10+ and an Anthropic API key from [console.anthropic.com](https://console.anthropic.com).

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your API key into it
streamlit run app.py
```

Run the tests (no API key needed):

```bash
python -m pytest -q
```

> **Tip for Claude Code:** open this folder and say *"Install the requirements, run the tests, and start the Streamlit app."*

---

## Put it live (Streamlit Community Cloud, free)

1. **Create a GitHub repo.** On GitHub, click **New repository** and name it `experiment-readout-agent`. Make it **Public** so hiring managers can see the code. Upload all the files in this folder, but **never upload `secrets.toml`**. The `.gitignore` blocks it if you push with git.
2. **Deploy.** Go to [share.streamlit.io](https://share.streamlit.io), click **Create app**, and pick your repo. Main file path: `app.py`.
3. **Add your API key.** Under **Advanced settings → Secrets**, paste:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-key"
   ```
4. **Pick a URL**, for example `experiment-readout-becca.streamlit.app`, to match your other apps. Then click **Deploy**.
5. **Protect your budget.** In the Anthropic Console, set a monthly spending limit. The app also allows only 5 AI readouts per visitor session and caches repeat runs.

If the model name ever changes, add `ANTHROPIC_MODEL = "..."` to Secrets. No code change is needed.

---

## Add it to beccakovac.com

Add a third card to the **AI projects** section, next to CRO Auditor and AI Visibility Agent:

> **Experiment Readout Agent**
> Turns raw A/B test results into a decision-ready readout. It catches false winners, broken traffic splits and tests too small to call, then builds a next-test backlog ranked by ICE.
> *Built with Claude, Python, and Streamlit.*
> **[Try the live app]** → `https://experiment-readout-becca.streamlit.app` · **[See the code]** → your GitHub repo URL

**Optional embed.** To show the app inside a page on your site:

```html
<iframe src="https://experiment-readout-becca.streamlit.app/?embed=true"
        width="100%" height="900" style="border:0;"></iframe>
```

**Heads-up:** free Streamlit apps go to sleep after some days without visits, and the first visitor has to click "wake up". Before you send the link to a hiring manager, open it yourself.

---

## CSV format

| Column | Required | Example |
|---|---|---|
| test_name | ✅ | Annual plan pre-selected |
| control_visitors | ✅ | 10000 |
| control_conversions | ✅ | 400 |
| variant_visitors | ✅ | 10000 |
| variant_conversions | ✅ | 480 |
| page | optional | Pricing page |
| hypothesis | optional | If we..., then..., because... |
| metric | optional | Checkout starts |

Up to 15 tests per upload. Column names are forgiving: for example, "Test Name" and "treatment_visitors" both work.

## Files

```
app.py                  Streamlit interface
stats_engine.py         All statistics (no AI)
ai_readout.py           Claude prompt, structured output, ICE ranking, Markdown export
sample_data/            Demo tests with known answers
tests/                  Automated checks (stats + AI layer with a fake Claude response)
.streamlit/             Theme + secrets template
```
