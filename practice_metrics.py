
import csv

# Step 1: Open the actual evaluation results file
with open(
    "evaluation_results.csv",
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    # Step 2: Read each row as a dictionary
    reader = csv.DictReader(file)

    # Step 3: Store all rows in a list
    results = list(reader)


# Step 4: Convert the score in each row to an integer
scores = []

for result in results:
    score = int(result["score"])
    scores.append(score)


# Step 5: Calculate the total and average

if scores:
    total_answers = len(scores)
    total_score = sum(scores)
    average_score = total_score / total_answers

    # Count answers with a score of 4 or 5
    passed = sum(1 for score in scores if score >= 4)

    # Calculate pass rate as a percentage
    pass_rate = (passed / total_answers) * 100

    print("Total evaluated answers:", total_answers)
    print("Average score:", round(average_score, 2), "/ 5")
    print("Passed answers:", passed)
    print("Pass rate:", round(pass_rate, 2), "%")

else:
    print("No evaluation results found.")
