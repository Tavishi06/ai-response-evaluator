
test_cases = [
    {
        "question": "What is an API?",
        "expected_answer": (
            "An API allows different software "
            "systems to communicate with each other."
        ),
        "context": (
            "An Application Programming Interface (API) "
            "defines how different software applications "
            "communicate and exchange data."
        ),
    },
    {
        "question": "What is RAG?",
        "expected_answer": (
            "RAG retrieves relevant external information "
            "and uses it to help generate an answer."
        ),
        "context": (
            "Retrieval-Augmented Generation (RAG) retrieves "
            "relevant information from an external knowledge "
            "source and provides it to a language model "
            "to help generate an answer."
        ),
    },
    {
        "question": "What is Python?",
        "expected_answer": (
            "Python is a programming language used "
            "to build software and automate tasks."
        ),
        "context": (
            "Python is a high-level programming language "
            "used for software development, automation, "
            "data analysis, and machine learning."
        ),
    },
]

print("Total test cases:", len(test_cases))

for test in test_cases:
    print("\nQuestion:", test["question"])
    print("Expected answer:", test["expected_answer"])
    print("Retrieved context:", test["context"])
