
import csv
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------
# Page configuration
# ---------------------------------

st.set_page_config(
    page_title="AI Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI Response Evaluation Dashboard")
st.write(
    "Analyze generated answers, evaluation scores, "
    "and failed test cases."
)

RESULTS_FILE = Path("evaluation_results.csv")
FAILURES_FILE = Path("failed_tests.csv")


# ---------------------------------
# Load evaluation results
# ---------------------------------

if not RESULTS_FILE.exists():
    st.warning(
        "No evaluation_results.csv found. "
        "Run batch_evaluate.py first."
    )
    st.stop()

try:
    with RESULTS_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        results = list(csv.DictReader(file))
except Exception as error:
    st.error(f"Could not read results: {type(error).__name__}")
    st.stop()

if not results:
    st.info("The results file contains no evaluated answers.")
    st.stop()

required_columns = {
    "question",
    "expected_answer",
    "generated_answer",
    "score",
    "reason",
}

if not required_columns.issubset(results[0].keys()):
    st.error("The CSV is missing required evaluation columns.")
    st.stop()

try:
    for result in results:
        result["score"] = int(result["score"])

    if any(
        not 1 <= result["score"] <= 5
        for result in results
    ):
        raise ValueError("Scores must be between 1 and 5.")

except (ValueError, TypeError):
    st.error("The CSV contains invalid evaluation scores.")
    st.stop()


# ---------------------------------
# Calculate metrics
# ---------------------------------

scores = [result["score"] for result in results]
total = len(results)
average_score = sum(scores) / total
passed = sum(score >= 4 for score in scores)
pass_rate = (passed / total) * 100

failed_tests = []

if FAILURES_FILE.exists():
    try:
        with FAILURES_FILE.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            failed_tests = list(csv.DictReader(file))
    except Exception:
        st.warning("Could not read failed_tests.csv.")

score_counts = {
    score: scores.count(score)
    for score in range(1, 6)
}


# ---------------------------------
# Display metric cards
# ---------------------------------

st.subheader("Overall Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Evaluated Answers", total)
col2.metric("Average Score", f"{average_score:.2f} / 5")
col3.metric("Pass Rate", f"{pass_rate:.1f}%")
col4.metric("Failed Tests", len(failed_tests))

st.divider()


# ---------------------------------
# Score distribution chart
# ---------------------------------

st.subheader("Score Distribution")

distribution_df = pd.DataFrame({
    "Score": list(score_counts.keys()),
    "Number of Answers": list(score_counts.values()),
})

st.bar_chart(
    distribution_df.set_index("Score")
)

st.caption(
    "Pass threshold: score of 4 or 5. "
    "This is a project-defined threshold."
)

st.divider()


# ---------------------------------
# Search and filter results
# ---------------------------------

st.subheader("Explore Evaluation Results")

minimum_score = st.selectbox(
    "Minimum score",
    options=[1, 2, 3, 4, 5],
    index=0,
)

search_text = st.text_input(
    "Search questions",
    placeholder="Type a keyword...",
)

filtered_results = [
    result
    for result in results
    if result["score"] >= minimum_score
    and search_text.lower() in result["question"].lower()
]

st.write(f"Showing {len(filtered_results)} of {total} answers.")

if filtered_results:
    display_df = pd.DataFrame(filtered_results)
    st.dataframe(display_df, use_container_width=True)

    st.download_button(
        "Download filtered results",
        data=display_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="filtered_evaluation_results.csv",
        mime="text/csv",
    )
else:
    st.info("No answers match these filters.")

st.divider()


# ---------------------------------
# Failed test details
# ---------------------------------

st.subheader("Failed Test Cases")

if failed_tests:
    st.dataframe(
        pd.DataFrame(failed_tests),
        use_container_width=True,
    )
else:
    st.success("No failed tests were recorded in the failure report.")

st.caption(
    "Evaluation scores are AI-generated judgments, "
    "not independently verified ground truth."
)
