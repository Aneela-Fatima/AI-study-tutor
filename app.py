import os
import json
from typing import List, Dict, Any, Tuple

import gradio as gr
from groq import Groq

# -----------------------------
# Configuration
# -----------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
client = Groq(api_key=GROQ_API_KEY)

LANG_OPTIONS = [
    "English",
    "Urdu",
    "Mandarin Chinease",   # kept as requested
    "Hindi",
    "Spanish",
    "Standard Arabic",
    "French",
    "Bengali",
    "Protaguese",          # kept as requested
    "Russian",
    "Indonasion",          # kept as requested
]

LEVEL_OPTIONS = ["Beginner", "Intermediate", "Advanced"]


# -----------------------------
# Helpers
# -----------------------------
def generate_with_groq(prompt: str) -> str:
    """
    Call Groq chat completions with the specified model and return text content.
    Includes basic error handling and a concise error message for the UI.
    """
    if not GROQ_API_KEY:
        return "❌ Missing GROQ_API_KEY. Set it in your environment or Space secrets."

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ API error: {e}"


def build_system_context(subject: str, topic: str, language: str, level: str) -> str:
    return (
        f"Subject: {subject}\n"
        f"Topic: {topic}\n"
        f"Language: {language}\n"
        f"Student Level: {level}\n"
    )


def prompt_explanation(subject: str, topic: str, language: str, level: str) -> str:
    ctx = build_system_context(subject, topic, language, level)
    return (
        f"{ctx}\n"
        "Task: Write a clear, friendly, step-by-step explanation of the topic. "
        "Use short paragraphs, numbered steps where helpful, and examples. "
        "Keep it concise but thorough. Reply in the specified language only."
    )


def prompt_resources(subject: str, topic: str, language: str, level: str) -> str:
    ctx = build_system_context(subject, topic, language, level)
    return (
        f"{ctx}\n"
        "Task: Recommend at least 3 quality learning resources (mix of articles, videos, documentation). "
        "Return as a markdown bulleted list. Each item must include a title, the type (Article/Video/Docs), "
        "a one-line why it's useful, and a URL. Reply in the specified language only."
    )


def prompt_roadmap(subject: str, topic: str, language: str, level: str) -> str:
    ctx = build_system_context(subject, topic, language, level)
    return (
        f"{ctx}\n"
        "Task: Produce a structured learning roadmap for this topic and level. "
        "Organize into stages with bullet points, estimated effort, and key outcomes. "
        "Add a short list of common mistakes to avoid. Reply in the specified language only."
    )


def prompt_quiz(subject: str, topic: str, language: str, level: str) -> str:
    ctx = build_system_context(subject, topic, language, level)
    return (
        f"{ctx}\n"
        "Task: Create a short multiple-choice quiz with 3 to 5 questions. "
        "Return STRICT JSON only with this schema:\n"
        "{\n"
        '  "questions": [\n'
        '    {\n'
        '      "question": "string",\n'
        '      "options": ["A", "B", "C", "D"],\n'
        '      "answer_index": 0\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Requirements:\n"
        "- options length 3-5\n"
        "- answer_index is an integer index into the options array\n"
        "- No additional commentary or code fences\n"
        f"- Write the question text and options in {language}."
    )


def _extract_first_json_object(text: str) -> str | None:
    """
    Safely extract the first top-level JSON object from text by balancing braces.
    Avoids regex recursion (not supported in Python re).
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def parse_quiz_json(text: str) -> Dict[str, Any]:
    """
    Extract and parse the JSON quiz from model output.
    Tries direct JSON, then a balanced-brace extraction.
    """
    # Try direct JSON first
    try:
        parsed = json.loads(text)
        if "questions" in parsed:
            return parsed
    except Exception:
        pass

    # Fallback: balanced extraction
    block = _extract_first_json_object(text)
    if block:
        try:
            parsed = json.loads(block)
            if "questions" in parsed:
                return parsed
        except Exception:
            pass

    return {"questions": []}


def normalize_quiz(quiz: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Ensure each question has required fields. Drop invalid ones.
    """
    cleaned = []
    for q in quiz.get("questions", []):
        question = q.get("question")
        options = q.get("options", [])
        answer_index = q.get("answer_index")
        if (
            isinstance(question, str)
            and isinstance(options, list)
            and 3 <= len(options) <= 5
            and isinstance(answer_index, int)
            and 0 <= answer_index < len(options)
        ):
            cleaned.append(
                {
                    "question": question.strip(),
                    "options": [str(o).strip() for o in options],
                    "answer_index": answer_index,
                }
            )
    return cleaned[:5]  # at most 5


def evaluate_answers(
    user_choices: List[int | None], quiz_data: List[Dict[str, Any]]
) -> Tuple[str, str]:
    """
    Compute score and short feedback summary.
    """
    correct = 0
    details = []
    for i, q in enumerate(quiz_data):
        user_idx = user_choices[i] if i < len(user_choices) else None
        ans_idx = q["answer_index"]
        is_correct = (user_idx == ans_idx)
        if is_correct:
            correct += 1
        chosen = (
            f"{q['options'][user_idx]}"
            if isinstance(user_idx, int) and 0 <= user_idx < len(q["options"])
            else "No answer"
        )
        details.append(
            f"Q{i+1}: {'✅ Correct' if is_correct else '❌ Incorrect'} | "
            f"Your answer: {chosen} | Correct: {q['options'][ans_idx]}"
        )

    total = len(quiz_data)
    if total == 0:
        return "No quiz generated yet.", ""
    score_text = f"Score: {correct} / {total}"
    if correct == total:
        feedback = "Great job. You’ve mastered this set."
    elif correct >= (total * 0.6):
        feedback = "Good work. Review the missed questions and try again."
    else:
        feedback = "Keep practicing. Revisit the explanation and roadmap."
    return score_text, feedback + "\n\n" + "\n".join(details)


# -----------------------------
# Gradio Callbacks
# -----------------------------
def on_generate_explanation(subject, topic, language, level):
    prompt = prompt_explanation(subject, topic, language, level)
    return generate_with_groq(prompt)


def on_generate_resources(subject, topic, language, level):
    prompt = prompt_resources(subject, topic, language, level)
    return generate_with_groq(prompt)


def on_generate_roadmap(subject, topic, language, level):
    prompt = prompt_roadmap(subject, topic, language, level)
    return generate_with_groq(prompt)


def on_generate_quiz(subject, topic, language, level):
    raw = generate_with_groq(prompt_quiz(subject, topic, language, level))
    quiz = normalize_quiz(parse_quiz_json(raw))

    # Build updates for up to 5 radios and their labels
    vis = [False] * 5
    labels = [("Question", ["Option 1", "Option 2", "Option 3"])] * 5

    for i, q in enumerate(quiz):
        vis[i] = True
        labels[i] = (f"Q{i+1}. {q['question']}", q["options"])

    return (
        quiz,  # gr.State
        gr.update(visible=vis[0], label=labels[0][0], choices=labels[0][1], value=None),
        gr.update(visible=vis[1], label=labels[1][0], choices=labels[1][1], value=None),
        gr.update(visible=vis[2], label=labels[2][0], choices=labels[2][1], value=None),
        gr.update(visible=vis[3], label=labels[3][0], choices=labels[3][1], value=None),
        gr.update(visible=vis[4], label=labels[4][0], choices=labels[4][1], value=None),
        raw if not quiz else "Quiz generated. Select your answers below."
    )


def on_display_results(quiz_state, a1, a2, a3, a4, a5):
    quiz = quiz_state or []
    # Map selected option text back to index
    selected_values = [a1, a2, a3, a4, a5]
    selections: List[int | None] = []
    for i, q in enumerate(quiz):
        chosen = selected_values[i]
        if chosen is None:
            selections.append(None)
            continue
        try:
            idx = q["options"].index(chosen)
        except ValueError:
            idx = None
        selections.append(idx)

    score_text, feedback = evaluate_answers(selections, quiz)
    return score_text, feedback


# -----------------------------
# UI
# -----------------------------
CSS = """
:root {
  --brand-blue: #1e40af;
  --brand-blue-600: #2563eb;
  --card-bg: #f8fafc;
  --border: #cbd5e1;
}
.gradio-container {max-width: 1200px !important}
#title h1 {color: var(--brand-blue); margin-bottom: 6px}
#subtitle {color:#334155; margin-top:0}
.card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px;
  box-shadow: 0 2px 8px rgb(2 6 23 / 6%);
}
.btn-primary button {
  background: var(--brand-blue-600) !important;
  border-color: var(--brand-blue-600) !important;
  color: white !important;
}
.section-title {
  font-weight: 700; color: var(--brand-blue);
  margin-bottom: 6px; font-size: 16px;
}
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("<div id='title'><h1>AI Study Tutor</h1><p id='subtitle'>Powered by Groq + Gradio</p></div>")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                gr.Markdown("### Inputs")
                subject = gr.Textbox(label="Subject", placeholder="e.g., Mathematics")
                topic = gr.Textbox(label="Topic", placeholder="e.g., Derivatives of Trigonometric Functions")
                language = gr.Dropdown(LANG_OPTIONS, value="English", label="Language")
                level = gr.Radio(LEVEL_OPTIONS, value="Beginner", label="Level")

        with gr.Column(scale=2):
            with gr.Group(elem_classes="card"):
                gr.Markdown("<div class='section-title'>Generate Explanation</div>")
                btn_explain = gr.Button("Generate Explanation", elem_classes="btn-primary")
                explanation = gr.Markdown(label="Explanation", value="")

            with gr.Group(elem_classes="card"):
                gr.Markdown("<div class='section-title'>Generate Resources</div>")
                btn_resources = gr.Button("Generate Resources", elem_classes="btn-primary")
                resources = gr.Markdown(label="Resources", value="")

    with gr.Row():
        with gr.Column():
            with gr.Group(elem_classes="card"):
                gr.Markdown("<div class='section-title'>Generate Roadmap</div>")
                btn_roadmap = gr.Button("Generate Roadmap", elem_classes="btn-primary")
                roadmap = gr.Markdown(label="Roadmap", value="")

    with gr.Row():
        with gr.Column():
            with gr.Group(elem_classes="card"):
                gr.Markdown("<div class='section-title'>Generate Quiz</div>")
                btn_quiz = gr.Button("Generate Quiz", elem_classes="btn-primary")
                quiz_info = gr.Markdown("Click the button to create a quiz.")
                quiz_state = gr.State([])
                q1 = gr.Radio(label="Question 1", choices=[], visible=False, interactive=True)
                q2 = gr.Radio(label="Question 2", choices=[], visible=False, interactive=True)
                q3 = gr.Radio(label="Question 3", choices=[], visible=False, interactive=True)
                q4 = gr.Radio(label="Question 4", choices=[], visible=False, interactive=True)
                q5 = gr.Radio(label="Question 5", choices=[], visible=False, interactive=True)

    with gr.Row():
        with gr.Column():
            with gr.Group(elem_classes="card"):
                gr.Markdown("<div class='section-title'>Display Results</div>")
                btn_results = gr.Button("Evaluate Answers", elem_classes="btn-primary")
                score = gr.Markdown("Score will appear here.")
                feedback = gr.Markdown("Feedback will appear here.")

    # Events
    btn_explain.click(
        fn=on_generate_explanation,
        inputs=[subject, topic, language, level],
        outputs=[explanation],
    )

    btn_resources.click(
        fn=on_generate_resources,
        inputs=[subject, topic, language, level],
        outputs=[resources],
    )

    btn_roadmap.click(
        fn=on_generate_roadmap,
        inputs=[subject, topic, language, level],
        outputs=[roadmap],
    )

    btn_quiz.click(
        fn=on_generate_quiz,
        inputs=[subject, topic, language, level],
        outputs=[quiz_state, q1, q2, q3, q4, q5, quiz_info],
    )

    btn_results.click(
        fn=on_display_results,
        inputs=[quiz_state, q1, q2, q3, q4, q5],
        outputs=[score, feedback],
    )

if __name__ == "__main__":
    demo.launch()
