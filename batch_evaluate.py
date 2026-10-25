
import os
import csv
import json
from statistics import mean

from dotenv import load_dotenv
from google import genai

from test_dataset import test_cases

# 1. Load our API settings
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)

# 2. Keep evaluation results in a list
results = []

# 3. Test each question
for index, test in enumerate(test_cases, start=1):
    question = test["question"]
    expected_answer = test["expected_answer"]

    print(f"\nRunning test {index}/{len(test_cases)}")
    print("Question:", question)

    try:
        # Generate the AI answer
        response = client.models.generate_content(
            model=model,
            contents=question,
        )

        generated_answer = response.text

        if not generated_answer:
            print("No answer returned. Skipping this test.")
            continue

        # Ask the evaluator for structured JSON
        prompt = f"""
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

The score must be an integer from 1 to 5.
Use 1 for a very poor match and 5 for an excellent match.
Do not give a high score if important information is wrong.
"""

        evaluation = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )

        # Convert the JSON text into a Python dictionary
        evaluation_data = json.loads(evaluation.text)

        score = int(evaluation_data["score"])
        reason = evaluation_data["reason"]

        if not 1 <= score <= 5:
            raise ValueError("Score must be between 1 and 5.")

        # Save this test's results in our list
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
        print("Check the model settings, response format, or API quota.")

# 4. Save results to a CSV file
if results:
    filename = "evaluation_results.csv"

    with open(filename, "w", newline="", encoding="utf-8-sig") as file:
        columns = [
            "question",
            "expected_answer",
            "generated_answer",
            "score",
            "reason",
        ]

        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    # 5. Calculate the average score
    scores = [result["score"] for result in results]
    average_score = mean(scores)

    print("\n" + "=" * 40)
    print("EVALUATION SUMMARY")
    print("Successful tests:", len(results))
    print("Average score:", round(average_score, 2), "/ 5")
    print("Results saved to:", filename)
else:
    print("\nNo successful evaluations to summarize.")
