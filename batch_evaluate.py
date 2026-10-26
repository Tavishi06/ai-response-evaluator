
import os
import csv
import json
import time
from statistics import mean

from dotenv import load_dotenv
from google import genai

from test_dataset import test_cases


# ----------------------------------------
# 1. Load environment variables
# ----------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing from your .env file."
    )

client = genai.Client(api_key=api_key)

results = []
failed_tests = []


# ----------------------------------------
# 2. Run and evaluate each test case
# ----------------------------------------

total_tests = len(test_cases)

print("=" * 45)
print("AI RESPONSE EVALUATOR")
print("=" * 45)
print("Total test cases:", total_tests)
print("Model:", model)

for index, test in enumerate(test_cases, start=1):
    question = test["question"]
    expected_answer = test["expected_answer"]

    print(f"\nRunning test {index}/{total_tests}")
    print("Question:", question)

    try:
        # Generate an answer
        print("Generating answer...")

        response = client.models.generate_content(
            model=model,
            contents=question,
        )

        generated_answer = response.text

        if not generated_answer or not generated_answer.strip():
            raise ValueError("The model returned an empty answer.")

        # Prepare evaluation instructions
        evaluation_prompt = f"""
You are an AI response evaluator.

Evaluate the generated answer against the reference answer.

Question:
{question}

Reference answer:
{expected_answer}

Generated answer:
{generated_answer}

Judge meaning and important details, not exact wording.
The reference answer may not contain every valid detail.

Return a JSON object with exactly these fields:
{{
    "score": 1,
    "reason": "Short explanation"
}}

Scoring rules:
1 = Very poor answer; mostly incorrect or irrelevant.
2 = Major errors or missing important information.
3 = Partially correct but has notable gaps.
4 = Mostly correct, relevant, and sufficiently complete.
5 = Accurate, relevant, and complete for the question.

The score must be an integer from 1 to 5.
Do not give a high score if important information is wrong.
"""

        # Ask Gemini to evaluate the answer
        print("Evaluating answer...")

        evaluation = client.models.generate_content(
            model=model,
            contents=evaluation_prompt,
            config={"response_mime_type": "application/json"},
        )

        if not evaluation.text:
            raise ValueError("The evaluator returned an empty response.")

        evaluation_data = json.loads(evaluation.text)

        score = evaluation_data["score"]
        reason = evaluation_data["reason"]

        # Validate the returned score
        if isinstance(score, bool) or not isinstance(score, int):
            raise ValueError("The evaluation score must be an integer.")

        if not 1 <= score <= 5:
            raise ValueError("The evaluation score must be between 1 and 5.")

        # Save successful evaluation
        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "score": score,
            "reason": reason,
        })

        print("Score:", score, "/ 5")
        print("Reason:", reason)

    except Exception as error:
        print("Test failed:", type(error).__name__)
        print("Details:", str(error))

        failed_tests.append({
            "question": question,
            "error_type": type(error).__name__,
            "error": str(error),
        })

    # Brief pause between test cases
    if index < total_tests:
        time.sleep(1)


# ----------------------------------------
# 3. Save successful evaluations to CSV
# ----------------------------------------

if results:
    results_filename = "evaluation_results.csv"

    columns = [
        "question",
        "expected_answer",
        "generated_answer",
        "score",
        "reason",
    ]

    with open(
        results_filename,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    scores = [result["score"] for result in results]

    average_score = mean(scores)

    passed = sum(1 for score in scores if score >= 4)
    pass_rate = (passed / len(scores)) * 100

    score_distribution = {
        score: scores.count(score)
        for score in range(1, 6)
    }

else:
    average_score = 0
    passed = 0
    pass_rate = 0

    score_distribution = {
        score: 0 for score in range(1, 6)
    }


# ----------------------------------------
# 4. Save failed tests separately
# ----------------------------------------

if failed_tests:
    failures_filename = "failed_tests.csv"

    with open(
        failures_filename,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        columns = ["question", "error_type", "error"]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(failed_tests)

    print("\nFailed test details saved to:", failures_filename)


# ----------------------------------------
# 5. Print final evaluation summary
# ----------------------------------------

print("\n" + "=" * 45)
print("EVALUATION SUMMARY")
print("=" * 45)

print("Total test cases:", total_tests)
print("Successful evaluations:", len(results))
print("Failed tests:", len(failed_tests))

if results:
    print("Average score:", round(average_score, 2), "/ 5")
    print("Passed tests (score >= 4):", passed)
    print("Pass rate:", round(pass_rate, 2), "%")

    print("\nScore distribution:")

    for score, count in score_distribution.items():
        print(f"Score {score}: {count} answer(s)")

    print("\nResults saved to:", results_filename)

else:
    print("\nNo successful evaluations were recorded.")

print("\nEvaluation run complete.")
