
test_cases = [
    {
        "question": "What is an API?",
        "expected_answer": (
            "An API allows different software "
            "systems to communicate with each other."
        ),
    },
    {
        "question": "What is RAG?",
        "expected_answer": (
            "RAG retrieves relevant external information "
            "and uses it to help generate an answer."
        ),
    },
    {
        "question": "What is Python?",
        "expected_answer": (
            "Python is a programming language used "
            "to build software and automate tasks."
        ),
    },
]

print("Total test cases:", len(test_cases))

for test in test_cases:
    print("\nQuestion:", test["question"])
    print("Expected answer:", test["expected_answer"])
