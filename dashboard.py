
import csv
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================
# 1. PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI Response & RAG Evaluation Dashboard")
st.write(
    "Evaluate answer quality, retrieved context relevance, "
    "and potential weaknesses in your RAG pipeline."
)

RESULTS_FILE = Path("evaluation_results.csv")
FAILURES_FILE = Path("failed_tests.csv")

ANSWER_SCORE_COLUMN = "score"
RELEVANCE_SCORE_COLUMN = "context_relevance_score"

ANSWER_PASS_THRESHOLD = 4
RELEVANCE_PASS_THRESHOLD = 4


# =========================================
# 2. LOAD AND VALIDATE RESULTS
# =========================================

if not RESULTS_FILE.exists():
    st.warning(
        "No evaluation_results.csv found. "
        "Run batch_evaluate.py first."
    )
    st.stop()

try:
    with RESULTS_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        results = list(reader)
        csv_columns = set(reader.fieldnames or [])

except (OSError, csv.Error, UnicodeError) as error:
    st.error(
        f"Could not read results: {type(error).__name__}: {error}"
    )
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

missing_columns = required_columns - csv_columns

if missing_columns:
    st.error(
        "The results CSV is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()

has_context = "context" in csv_columns
has_relevance = RELEVANCE_SCORE_COLUMN in csv_columns
has_relevance_reason = "context_relevance_reason" in csv_columns

if has_relevance and not has_context:
    st.warning(
        "A context relevance score exists, but the context column "
        "is missing. Check the evaluation output."
    )


def validate_scores(rows, column):
    """Convert a CSV score column to integers and validate 1–5."""

    for row_number, row in enumerate(rows, start=2):
        raw_score = row.get(column, "")

        try:
            score = int(raw_score)
        except (ValueError, TypeError):
            raise ValueError(
                f"Invalid {column!r} at CSV row {row_number}."
            )

        if not 1 <= score <= 5:
            raise ValueError(
                f"{column!r} must be between 1 and 5 "
                f"(CSV row {row_number})."
            )

        row[column] = score


try:
    validate_scores(results, ANSWER_SCORE_COLUMN)

    if has_relevance:
        validate_scores(results, RELEVANCE_SCORE_COLUMN)

except ValueError as error:
    st.error(f"Invalid evaluation data: {error}")
    st.stop()


# =========================================
# 3. LOAD FAILED TEST REPORT
# =========================================

failed_tests = []

if FAILURES_FILE.exists():
    try:
        with FAILURES_FILE.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            failed_tests = list(csv.DictReader(file))

    except (OSError, csv.Error, UnicodeError) as error:
        st.warning(
            f"Could not read failed_tests.csv: "
            f"{type(error).__name__}"
        )


# =========================================
# 4. CALCULATE METRICS
# =========================================

total = len(results)

answer_scores = [
    row[ANSWER_SCORE_COLUMN]
    for row in results
]

average_answer_score = sum(answer_scores) / total

answer_passed = sum(
    score >= ANSWER_PASS_THRESHOLD
    for score in answer_scores
)

answer_pass_rate = (answer_passed / total) * 100

answer_distribution = {
    score: answer_scores.count(score)
    for score in range(1, 6)
}

answer_percentages = {
    score: (count / total) * 100
    for score, count in answer_distribution.items()
}

if has_relevance:
    relevance_scores = [
        row[RELEVANCE_SCORE_COLUMN]
        for row in results
    ]

    average_relevance_score = (
        sum(relevance_scores) / total
    )

    relevance_passed = sum(
        score >= RELEVANCE_PASS_THRESHOLD
        for score in relevance_scores
    )

    relevance_pass_rate = (
        relevance_passed / total
    ) * 100

    relevance_distribution = {
        score: relevance_scores.count(score)
        for score in range(1, 6)
    }

    relevance_percentages = {
        score: (count / total) * 100
        for score, count in relevance_distribution.items()
    }


# =========================================
# 5. OVERALL PERFORMANCE
# =========================================

st.header("Overall Performance")

if has_relevance:
    metric_columns = st.columns(5)

    metric_columns[0].metric(
        "Evaluated Answers",
        total,
    )

    metric_columns[1].metric(
        "Avg Answer Quality",
        f"{average_answer_score:.2f} / 5",
    )

    metric_columns[2].metric(
        "Answer Pass Rate",
        f"{answer_pass_rate:.1f}%",
    )

    metric_columns[3].metric(
        "Avg Context Relevance",
        f"{average_relevance_score:.2f} / 5",
    )

    metric_columns[4].metric(
        "Failed Tests",
        len(failed_tests),
    )

else:
    metric_columns = st.columns(4)

    metric_columns[0].metric("Evaluated Answers", total)
    metric_columns[1].metric(
        "Avg Answer Quality",
        f"{average_answer_score:.2f} / 5",
    )
    metric_columns[2].metric(
        "Answer Pass Rate",
        f"{answer_pass_rate:.1f}%",
    )
    metric_columns[3].metric(
        "Failed Tests",
        len(failed_tests),
    )

    st.warning(
        "This CSV uses the older format. Run batch_evaluate.py "
        "to generate context relevance metrics."
    )

st.caption(
    "A score of 4 or 5 is considered passing by this project's "
    "configured threshold; it is not a universal RAG benchmark."
)

st.divider()


# =========================================
# 6. ANSWER QUALITY ANALYSIS
# =========================================

st.header("Answer Quality Analysis")

answer_chart_data = pd.DataFrame({
    "Score": list(answer_distribution.keys()),
    "Number of Answers": list(answer_distribution.values()),
})

st.subheader("Answer Score Distribution")
st.bar_chart(answer_chart_data.set_index("Score"))

answer_percentage_data = pd.DataFrame({
    "Score": list(answer_percentages.keys()),
    "Percentage (%)": [
        round(value, 2)
        for value in answer_percentages.values()
    ],
})

with st.expander("View answer score percentages"):
    st.dataframe(
        answer_percentage_data,
        use_container_width=True,
        hide_index=True,
    )

st.divider()


# =========================================
# 7. CONTEXT RELEVANCE ANALYSIS
# =========================================

if has_relevance:
    st.header("Context Relevance Analysis")

    relevance_chart_data = pd.DataFrame({
        "Score": list(relevance_distribution.keys()),
        "Number of Contexts": list(
            relevance_distribution.values()
        ),
    })

    st.subheader("Context Relevance Distribution")

    st.bar_chart(
        relevance_chart_data.set_index("Score")
    )

    relevance_percentage_data = pd.DataFrame({
        "Score": list(relevance_percentages.keys()),
        "Percentage (%)": [
            round(value, 2)
            for value in relevance_percentages.values()
        ],
    })

    with st.expander("View context relevance percentages"):
        st.dataframe(
            relevance_percentage_data,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()


# =========================================
# 8. RAG QUALITY DIAGNOSTICS
# =========================================

st.header("🔎 RAG Quality Diagnostics")

if has_relevance:
    low_answer_results = [
        row for row in results
        if row["score"] <= 3
    ]

    low_relevance_results = [
        row for row in results
        if row[RELEVANCE_SCORE_COLUMN] <= 3
    ]

    answer_low_context_high = [
        row for row in results
        if (
            row["score"] <= 3
            and row[RELEVANCE_SCORE_COLUMN] >= 4
        )
    ]

    both_low_results = [
        row for row in results
        if (
            row["score"] <= 3
            and row[RELEVANCE_SCORE_COLUMN] <= 3
        )
    ]

    diag_columns = st.columns(4)

    diag_columns[0].metric(
        "Low-Quality Answers",
        len(low_answer_results),
    )

    diag_columns[1].metric(
        "Low-Relevance Contexts",
        len(low_relevance_results),
    )

    diag_columns[2].metric(
        "Good Context, Weak Answer",
        len(answer_low_context_high),
    )

    diag_columns[3].metric(
        "Both Scores Low",
        len(both_low_results),
    )

    st.markdown("**How to interpret these patterns**")

    st.markdown(
        "- **Low context relevance:** investigate retrieval and "
        "the quality of the selected documents.\n"
        "- **Relevant context but weak answer:** inspect generation "
        "instructions, answer completeness, and model behaviour.\n"
        "- **Both scores low:** retrieval and generation may both "
        "need investigation."
    )

    if low_answer_results:
        with st.expander(
            f"Review {len(low_answer_results)} low-quality answers"
        ):
            st.dataframe(
                pd.DataFrame(low_answer_results),
                use_container_width=True,
                hide_index=True,
            )

    if low_relevance_results:
        with st.expander(
            f"Review {len(low_relevance_results)} low-relevance contexts"
        ):
            st.dataframe(
                pd.DataFrame(low_relevance_results),
                use_container_width=True,
                hide_index=True,
            )

    if not low_answer_results and not low_relevance_results:
        st.success(
            "No answers or contexts crossed the low-score threshold."
        )

else:
    low_answer_results = [
        row for row in results
        if row["score"] <= 3
    ]

    if low_answer_results:
        st.warning(
            f"{len(low_answer_results)} answers scored 3 or below."
        )
        st.dataframe(
            pd.DataFrame(low_answer_results),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No low-scoring answers found.")

st.divider()


# =========================================
# 9. SEARCH AND FILTER RESULTS
# =========================================

st.header("Explore Evaluation Results")

filter_columns = st.columns(3)

minimum_answer_score = filter_columns[0].selectbox(
    "Minimum answer quality",
    options=[1, 2, 3, 4, 5],
    index=1,
)

if has_relevance:
    minimum_relevance_score = filter_columns[1].selectbox(
        "Minimum context relevance",
        options=[1, 2, 3, 4, 5],
        index=1,
    )
else:
    minimum_relevance_score = 1
    filter_columns[1].info(
        "Context relevance filter requires the new CSV format."
    )

search_text = filter_columns[2].text_input(
    "Search questions",
    placeholder="Type a keyword...",
)

filtered_results = []

for row in results:
    matches_answer = (
        row["score"] >= minimum_answer_score
    )

    matches_relevance = (
        not has_relevance
        or row[RELEVANCE_SCORE_COLUMN] >= minimum_relevance_score
    )

    matches_search = (
        search_text.strip().casefold()
        in row["question"].casefold()
    )

    if matches_answer and matches_relevance and matches_search:
        filtered_results.append(row)

st.write(
    f"Showing {len(filtered_results)} of {total} evaluated answers."
)

display_columns = [
    "question",
    "expected_answer",
]

if has_context:
    display_columns.append("context")

display_columns.extend([
    "generated_answer",
    "score",
    "reason",
])

if has_relevance:
    display_columns.append(RELEVANCE_SCORE_COLUMN)

if has_relevance_reason:
    display_columns.append("context_relevance_reason")

if filtered_results:
    display_df = pd.DataFrame(filtered_results)
    display_df = display_df[display_columns]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download filtered results",
        data=display_df.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="filtered_evaluation_results.csv",
        mime="text/csv",
    )

else:
    st.info("No answers match the selected filters.")

st.divider()


# =========================================
# 10. FAILED TEST DETAILS
# =========================================

st.header("Failed Test Cases")

if failed_tests:
    st.warning(
        f"{len(failed_tests)} test case(s) failed during execution."
    )

    st.dataframe(
        pd.DataFrame(failed_tests),
        use_container_width=True,
        hide_index=True,
    )

else:
    st.success("No failed tests were recorded in the failure report.")


# =========================================
# 11. FOOTER
# =========================================

st.divider()

st.caption(
    "AI-generated evaluation scores are estimates, not verified "
    "ground truth. The diagnostic patterns are clues for debugging, "
    "not proof of the root cause."
)
