# AI Friendly Study Tutor

A student-friendly AI tutor built with the Groq API and Gradio. Select a subject, topic, language, and level. Then generate a clear explanation, curated resources, a structured roadmap, and a short quiz. Finish by evaluating your answers to see your score and feedback.

## Features

- **Generate Explanation**: Step-by-step explanation of the topic.
- **Generate Resources**: At least 3 recommended links (articles, videos, docs) with a short reason.
- **Generate Roadmap**: A structured learning path with stages, estimated effort, and outcomes.
- **Generate Quiz**: 3–5 multiple-choice questions with radio-button options (one correct answer).
- **Display Results**: Evaluates your selected answers and returns a score with brief feedback.

## Tech

- **Frontend/Backend**: Gradio  
- **Model Runtime**: Groq API using `llama-3.1-8b-instant`

## Run Locally

1. (Optional) Create a virtual environment
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate

