test_cases = [
    {
        "question": "What is RAG?",
        "difficulty": "easy",
        "expected_keywords": ["retrieval", "generation"]
    },
    {
        "question": "What are embeddings?",
        "difficulty": "medium",
        "expected_keywords": ["vectors", "text"]
    },
    {
        "question": "Why evaluate an AI answer?",
        "difficulty": "easy",
        "expected_keywords": ["accuracy"]
    }
]

for case in test_cases:

    if case["difficulty"] == "easy":
        label = "Beginner test"
    else:
        label = "Advanced test"

    print(label, ":", case["question"])
    print("Expected keywords:", ", ".join(case["expected_keywords"]))
    print("-" * 40)