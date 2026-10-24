
import os

import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

st.set_page_config(
    page_title="AI Response Evaluator",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 AI Response Evaluator")
st.write(
    "Generate an AI answer and evaluate its quality."
)

# Connect to Gemini
api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "generated_question" not in st.session_state:
    st.session_state.generated_question = ""

if "evaluation" not in st.session_state:
    st.session_state.evaluation = ""

# Take the user's question
question = st.text_area(
    "Enter your question",
    placeholder="What is RAG? Explain with an example.",
)

# Generate an answer
if st.button("Generate Answer"):
    if not question.strip():
        st.warning("Please enter a question.")

    elif not api_key:
        st.error("Gemini API key is missing from .env.")

    else:
        try:
            client = genai.Client(api_key=api_key)

            with st.spinner("Generating answer..."):
                response = client.models.generate_content(
                    model=model,
                    contents=question,
                )

            if response.text:
                st.session_state.answer = response.text
                st.session_state.generated_question = question
                st.session_state.evaluation = ""
            else:
                st.warning("The model returned no text.")

        except Exception:
            st.error(
                "Answer generation failed. Check your model, "
                "API key, internet connection, or quota."
            )

# Display the saved answer
if st.session_state.answer:
    st.subheader("Generated Answer")
    st.write(st.session_state.answer)

    st.divider()
    st.subheader("Evaluate Answer")

    st.write(
        "We'll score the answer from 1 to 5 on four criteria."
    )

    if st.button("Evaluate Response"):
        try:
            client = genai.Client(api_key=api_key)

            evaluation_prompt = f"""
You are an AI response quality evaluator.

Evaluate the answer to the question below.

QUESTION:
{st.session_state.generated_question}

ANSWER:
{st.session_state.answer}

Score each criterion from 1 to 5.

1. Relevance: Does the answer address the question?
2. Completeness: Does it cover the important points?
3. Clarity: Is it understandable and well-organized?
4. Factual reliability: Are claims accurate and appropriately qualified?

For each criterion, provide:
- Score out of 5
- A short reason for the score
- One specific improvement, if needed

Then calculate the average score out of 5.

Use this format:

### Relevance: X/5
Reason:
Improvement:

### Completeness: X/5
Reason:
Improvement:

### Clarity: X/5
Reason:
Improvement:

### Factual Reliability: X/5
Reason:
Improvement:

### Overall Score: X/5
### Final Feedback:
"""

            with st.spinner("Evaluating response..."):
                result = client.models.generate_content(
                    model=model,
                    contents=evaluation_prompt,
                )

            if result.text:
                st.session_state.evaluation = result.text
            else:
                st.warning("No evaluation was returned.")

        except Exception:
            st.error(
                "Evaluation failed. Check your API settings "
                "and available quota."
            )

    if st.session_state.evaluation:
        st.markdown(st.session_state.evaluation)
