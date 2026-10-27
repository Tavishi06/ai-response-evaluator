
import os
import csv
import json
import time
from statistics import mean

from dotenv import load_dotenv
from google import genai

from test_dataset import test_cases


# ----------------------------------------
# 1. Configuration
# ----------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

RESULTS_FILE = "evaluation_results.csv"
FAILURES_FILE = "failed_tests.csv"

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing from your .env file."
    )

client = genai.Client(api_key=api_key)

results = []
failed_tests = []


# ----------------------------------------
# 2. Helper functions
# ----------------------------------------

def get_valid_evaluation(prompt, score_field):
    """Ask Gemini for JSON and validate the evaluation."""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )

    if not response.text or not response.text.strip():
        raise ValueError("The evaluator returned an empty response.")

    data = json.loads(response.text)

    if not isinstance(data, dict):
        raise ValueError("The evaluation response must be a JSON object.")

    if score_field not in data or "reason" not in data:
        raise ValueError(
            f"Evaluation must contain {score_field!r} and 'reason'."
        )

    score = data[score_field]
    reason = data["reason"]

    if isinstance(score, bool) or not isinstance(score, int):
        raise ValueError("The evaluation score must be an integer.")

    if not 1 <= score <= 5:
        raise ValueError("The evaluation score must be between 1 and 5.")

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("The evaluation reason must be a non-empty string.")

    return score, reason.strip()


def evaluate_answer(question, expected_answer, generated_answer):
    """Evaluate the quality of the generated answer."""

    prompt = f"""
You are an independent AI response evaluator.

Evaluate the generated answer against the reference answer.

Question:
{question}

Reference answer:
{expected_answer}

Generated answer:
{generated_answer}

Judge meaning, correctness, relevance, and completeness.
Do not require exact wording.
The reference answer may not contain every valid detail.

Return JSON with exactly these fields:
{{
    "score": 1,
    "reason": "Short explanation"
}}

Scoring:
1 = Very poor, incorrect, or irrelevant.
2 = Major errors or missing important information.
3 = Partially correct with notable gaps.
4 = Mostly correct and sufficiently complete.
5 = Accurate, relevant, and complete.

The score must be an integer from 1 to 5.
"""

    return get_valid_evaluation(prompt, "score")


def evaluate_context_relevance(question, context):
    """Evaluate whether the retrieved context helps answer the question."""

    prompt = f"""
You are an independent RAG context relevance evaluator.

Determine how relevant the retrieved context is to answering
the user's question.

User question:
{question}

Retrieved context:
{context}

Judge the usefulness of the context for answering this question.
Do not judge the quality of a generated answer.
A context can be relevant even if it does not contain every detail
needed for a complete answer.

Return JSON with exactly these fields:
{{
    "context_relevance_score": 1,
    "reason": "Short explanation"
}}

Scoring:
1 = Completely unrelated to the question.
2 = Mostly unrelated; provides very little useful information.
3 = Partially relevant; some useful information is present.
4 = Mostly relevant and useful for answering the question.
5 = Directly relevant and strongly supports answering the question.

The score must be an integer from 1 to 5.
"""

    return get_valid_evaluation(
        prompt,
        "context_relevance_score",
    )


# ----------------------------------------
# 3. Generate and evaluate each test case
# ----------------------------------------

total_tests = len(test_cases)

print("=" * 55)
print("AI RESPONSE + RAG CONTEXT EVALUATOR")
print("=" * 55)
print("Total test cases:", total_tests)
print("Model:", model)
print("API calls per successful test: 3")
print("")

for index, test in enumerate(test_cases, start=1):
    question = test.get("question", "")
    expected_answer = test.get("expected_answer", "")
    context = test.get("context", "")

    print(f"\nTest {index}/{total_tests}")
    print("-" * 45)
    print("Question:", question)

    try:
        # Validate the test data first.
        if not isinstance(question, str) or not question.strip():
            raise ValueError("The test question is missing.")

        if (
            not isinstance(expected_answer, str)
            or not expected_answer.strip()
        ):
            raise ValueError("The expected answer is missing.")

        if not isinstance(context, str) or not context.strip():
            raise ValueError("The retrieved context is missing.")

        # Call 1: Generate an answer using the retrieved context.
        print("1/3 Generating a context-aware answer...")

        generation_prompt = f"""
Answer the user's question using the retrieved context below.

User question:
{question}

Retrieved context:
{context}

Instructions:
- Use the context as your primary source.
- Do not invent facts that the context does not support.
- If the context is insufficient, clearly say so.
- Give a concise, helpful answer.
"""

        response = client.models.generate_content(
            model=model,
            contents=generation_prompt,
        )

        generated_answer = response.text

        if not generated_answer or not generated_answer.strip():
            raise ValueError("The model returned an empty answer.")

        generated_answer = generated_answer.strip()

        # Call 2: Evaluate the generated answer.
        print("2/3 Evaluating answer quality...")

        answer_score, answer_reason = evaluate_answer(
            question,
            expected_answer,
            generated_answer,
        )

        # Call 3: Evaluate retrieved context relevance.
        print("3/3 Evaluating context relevance...")

        relevance_score, relevance_reason = (
            evaluate_context_relevance(question, context)
        )

        # Save the complete successful evaluation.
        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "context": context,
            "generated_answer": generated_answer,
            "score": answer_score,
            "reason": answer_reason,
            "context_relevance_score": relevance_score,
            "context_relevance_reason": relevance_reason,
        })

        print("Answer quality:", answer_score, "/ 5")
        print("Answer reason:", answer_reason)
        print("Context relevance:", relevance_score, "/ 5")
        print("Context reason:", relevance_reason)

    except Exception as error:
        print("Test failed:", type(error).__name__)
        print("Details:", str(error))

        failed_tests.append({
            "question": question,
            "error_type": type(error).__name__,
            "error": str(error),
        })

    # Short delay between test cases.
    if index < total_tests:
        time.sleep(1)


# ----------------------------------------
# 4. Save evaluation results
# ----------------------------------------

result_columns = [
    "question",
    "expected_answer",
    "context",
    "generated_answer",
    "score",
    "reason",
    "context_relevance_score",
    "context_relevance_reason",
]

# Always rewrite the report to avoid showing stale results.
with open(
    RESULTS_FILE,
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:
    writer = csv.DictWriter(file, fieldnames=result_columns)
    writer.writeheader()
    writer.writerows(results)


# ----------------------------------------
# 5. Save failed tests
# ----------------------------------------

failure_columns = ["question", "error_type", "error"]

if failed_tests:
    with open(
        FAILURES_FILE,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=failure_columns)
        writer.writeheader()
        writer.writerows(failed_tests)

    print("\nFailed test details saved to:", FAILURES_FILE)

elif os.path.exists(FAILURES_FILE):
    # Remove an old failure report when this run has no failures.
    os.remove(FAILURES_FILE)


# ----------------------------------------
# 6. Calculate evaluation metrics
# ----------------------------------------

answer_scores = [item["score"] for item in results]

relevance_scores = [
    item["context_relevance_score"]
    for item in results
]

if results:
    average_answer_score = mean(answer_scores)
    average_relevance_score = mean(relevance_scores)

    answer_passed = sum(
        score >= 4 for score in answer_scores
    )

    relevance_passed = sum(
        score >= 4 for score in relevance_scores
    )

    answer_pass_rate = (
        answer_passed / len(results)
    ) * 100

    relevance_pass_rate = (
        relevance_passed / len(results)
    ) * 100

    answer_distribution = {
        score: answer_scores.count(score)
        for score in range(1, 6)
    }

    relevance_distribution = {
        score: relevance_scores.count(score)
        for score in range(1, 6)
    }

else:
    average_answer_score = 0
    average_relevance_score = 0
    answer_passed = 0
    relevance_passed = 0
    answer_pass_rate = 0
    relevance_pass_rate = 0

    answer_distribution = {score: 0 for score in range(1, 6)}
    relevance_distribution = {score: 0 for score in range(1, 6)}


# ----------------------------------------
# 7. Print final summary
# ----------------------------------------

print("\n" + "=" * 55)
print("RAG EVALUATION SUMMARY")
print("=" * 55)

print("Total test cases:", total_tests)
print("Successful evaluations:", len(results))
print("Failed tests:", len(failed_tests))

if results:
    print("\nANSWER QUALITY")
    print("Average:", round(average_answer_score, 2), "/ 5")
    print("Passed (score >= 4):", answer_passed)
    print("Pass rate:", round(answer_pass_rate, 2), "%")

    print("\nCONTEXT RELEVANCE")
    print("Average:", round(average_relevance_score, 2), "/ 5")
    print("Passed (score >= 4):", relevance_passed)
    print("Pass rate:", round(relevance_pass_rate, 2), "%")

    print("\nANSWER SCORE DISTRIBUTION")

    for score, count in answer_distribution.items():
        print(f"Score {score}: {count} answer(s)")

    print("\nCONTEXT RELEVANCE DISTRIBUTION")

    for score, count in relevance_distribution.items():
        print(f"Score {score}: {count} context(s)")

else:
    print("\nNo successful evaluations were recorded.")

print("\nResults saved to:", RESULTS_FILE)
print("Evaluation run complete.")
