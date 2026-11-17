"""Reproducible local dashboard for the Math Handbook RAG evaluation.
Run from sample_code: streamlit run dashboard_app.py
Uses CSV data in evaluation_results; no Ollama/GPU required.
"""
from pathlib import Path
import csv
import statistics
import streamlit as st

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "evaluation_results"
V1_PATH = DATA_DIR / "results.csv"
V2_PATH = DATA_DIR / "results_v2.csv"
GRADES_PATH = DATA_DIR / "manual_review.csv"

# Initial HUMAN-REVIEWED draft labels from the 20 responses inspected during the project.
# Review or edit evaluation_results/manual_review.csv before citing results academically.
INITIAL_GRADES = {f"Q{i:02d}": (0 if i in (6, 15) else 1) for i in range(1, 21)}
CATEGORIES = {
    **{f"Q{i:02d}": "Major & graduation" for i in range(1, 5)},
    **{f"Q{i:02d}": "Minor" for i in range(5, 9)},
    "Q09": "Courses", "Q10": "Tracks", "Q11": "Tracks", "Q12": "Tracks", "Q13": "Tracks",
    "Q14": "Electives & policy", "Q15": "Electives & policy", "Q16": "Electives & policy", "Q17": "Electives & policy",
    "Q18": "Study plan", "Q19": "Core curriculum", "Q20": "Out of scope",
}
NOTES = {
    "Q06": "Wrong unit: 6 credits instead of 6 MATH courses",
    "Q15": "Wrong requirement: 200-level instead of at least one 300-level elective",
}


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ensure_review_file():
    if GRADES_PATH.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with GRADES_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "score", "review_note"])
        writer.writeheader()
        for qid, grade in INITIAL_GRADES.items():
            writer.writerow({"question_id": qid, "score": grade, "review_note": NOTES.get(qid, "")})


def parse_grade(value):
    value = str(value).strip()
    return int(value) if value in ("0", "1") else None


def format_seconds(value):
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


st.set_page_config(page_title="Math RAG | Evaluation", page_icon="📊", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
h1, h2, h3 {letter-spacing:-0.025em;}
[data-testid='stMetric'] {background: rgba(120,120,120,.07);border: 1px solid rgba(120,120,120,.16);padding: 1rem;border-radius: 13px;}
</style>""", unsafe_allow_html=True)
st.title("📊 Mathematics Handbook — RAG Evaluation")
st.caption("Local, reproducible evaluation • Qwen2.5 1.5B • ChromaDB • PDF Handbook")

v1 = read_csv(V1_PATH)
v2 = read_csv(V2_PATH)
if not v2:
    st.error(f"Could not find V2 data: {V2_PATH}")
    st.stop()
ensure_review_file()
grades = {row.get("question_id", ""): row for row in read_csv(GRADES_PATH)}

rows = []
for r in v2:
    qid = r.get("question_id", "")
    grade = parse_grade(grades.get(qid, {}).get("score", ""))
    rows.append({
        "ID": qid, "Category": CATEGORIES.get(qid, "Other"),
        "Question": r.get("question", ""), "Expected": r.get("expected_answer", ""),
        "Model answer": r.get("actual_answer", ""),
        "Score": grade, "Status": "PASS" if grade == 1 else ("FAIL" if grade == 0 else "NOT REVIEWED"),
        "Latency (s)": format_seconds(r.get("response_time_seconds")),
        "Pages": r.get("source_pages", ""),
        "Review note": grades.get(qid, {}).get("review_note", ""),
    })

reviewed = [r for r in rows if r["Score"] is not None]
correct = sum(r["Score"] for r in reviewed)
latencies = [r["Latency (s)"] for r in rows if r["Latency (s)"] is not None]
accuracy = 100 * correct / len(reviewed) if reviewed else None

# Match the same four question IDs, one fixed baseline configuration.
shared_ids = {"Q01", "Q02", "Q03", "Q04"}
baseline = [r for r in v1 if r.get("question_id") in shared_ids
            and r.get("model") == "qwen2.5:1.5b" and str(r.get("top_k")) == "10"]
baseline_id_set = {r["question_id"] for r in baseline}
# The four baseline labels were reviewed against the handbook, NOT inferred by keyword matching.
baseline_human_grades = {"Q01": 0, "Q02": 1, "Q03": 1, "Q04": 1}
baseline_correct = sum(baseline_human_grades.get(r["question_id"], 0) for r in baseline)
v2_shared = [r for r in rows if r["ID"] in shared_ids and r["Score"] is not None]

overview, results_tab, errors_tab, method_tab = st.tabs(
    ["Overview", "All 20 questions", "Error analysis", "Methodology"]
)
with overview:
    a, b, c, d = st.columns(4)
    a.metric("Reviewed accuracy", f"{accuracy:.1f}%" if accuracy is not None else "N/A")
    b.metric("Correct / reviewed", f"{correct}/{len(reviewed)}")
    c.metric("Mean latency", f"{statistics.mean(latencies):.2f}s" if latencies else "N/A")
    d.metric("Median latency", f"{statistics.median(latencies):.2f}s" if latencies else "N/A")
    st.caption(f"{len(rows)} rows in V2 CSV • {len(reviewed)} manually graded • {len(v1)} baseline runs")

    left, right = st.columns(2)
    with left:
        st.subheader("Shared Q01–Q04")
        if len(baseline_id_set) == 4 and len(v2_shared) == 4:
            comparison = [
                {"Configuration": "V1 · semantic · K10", "Correct": baseline_correct},
                {"Configuration": "V2 · hybrid · K5", "Correct": sum(x["Score"] for x in v2_shared)},
            ]
            st.bar_chart(comparison, x="Configuration", y="Correct")
        else:
            st.info("Missing some shared questions; comparison unavailable.")
        st.caption("Descriptive only: V1 and V2 also differ in prompt and Top-K.")
    with right:
        st.subheader("Response time by question")
        st.bar_chart([{"Question": r["ID"], "Seconds": r["Latency (s)"]}
                      for r in rows if r["Latency (s)"] is not None], x="Question", y="Seconds")
        st.caption("Cold starts were not measured separately.")

    st.subheader("Accuracy by category")
    groups = {}
    for r in reviewed:
        entry = groups.setdefault(r["Category"], [0, 0])
        entry[0] += r["Score"]
        entry[1] += 1
    group_rows = [{"Category": k, "Correct": v[0], "Tested": v[1],
                   "Accuracy (%)": round(100 * v[0] / v[1], 1)} for k, v in groups.items()]
    st.dataframe(group_rows, use_container_width=True, hide_index=True)

with results_tab:
    st.subheader("Question-level results")
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("Scores come from manual_review.csv, not from automatic text similarity.")
    selected = st.selectbox("Inspect answer", [r["ID"] for r in rows])
    record = next(r for r in rows if r["ID"] == selected)
    st.markdown(f"**Question:** {record['Question']}")
    st.markdown(f"**Expected:** {record['Expected']}")
    st.markdown("**Model answer:**")
    st.write(record["Model answer"])
    st.write(f"**Review:** {record['Status']} — {record['Review note'] or 'No note'}")
    st.caption(f"Retrieved pages: {record['Pages']}")

with errors_tab:
    failures = [r for r in rows if r["Score"] == 0]
    st.subheader(f"Manually marked failures ({len(failures)})")
    if failures:
        for r in failures:
            with st.expander(f"{r['ID']} — {r['Category']}", expanded=True):
                st.write("**Question:**", r["Question"])
                st.write("**Expected:**", r["Expected"])
                st.write("**Actual:**", r["Model answer"])
                st.write("**Review note:**", r["Review note"] or "Not provided")
                st.caption(f"Retrieved pages: {r['Pages']}")
    else:
        st.info("No failures currently marked in the manual review file.")

with method_tab:
    st.subheader("Reproducibility")
    st.markdown("""
    - **Source data:** `evaluation_results/results.csv` and `results_v2.csv`.
    - **Human judgments:** `evaluation_results/manual_review.csv` (generated initially with draft grades; **review and edit yourself**).
    - **Correctness:** one point only when both the fact and its unit/condition match the handbook. Q06 and Q15 are failures under this rule.
    - **Matched comparison:** V1 (1.5B, semantic, K=10) versus V2 (1.5B, hybrid, K=5) on Q01–Q04 only.
    - **Limitations:** prompts and Top-K differ between versions; 20 questions, one PDF; a single out-of-scope question; potential warm/cold-start timing effects.
    - **Not used:** LLM-based judging or unverified automated accuracy scoring.
    """)
    st.code("streamlit run dashboard_app.py", language="bash")
    st.info("To change a PASS/FAIL judgment, edit manual_review.csv (score = 1 or 0), save, and refresh the dashboard.")
