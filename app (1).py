import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import tensorflow as tf
import gradio as gr
from fpdf import FPDF
import pandas as pd

# Load Features from CSV (curate a subset for demo clarity)
features_df = pd.read_csv("Feature-Description.csv")
key_features = [
    "Automatic Code Analysis",
    "Context-Aware Documentation",
    "Real-Time Updates",
    "Dependency Mapping",
    "API Documentation",
    "Test Suite Generation",
    "UML Diagram Generation",
    "Bug/Issue Identification",
    "Natural Language Explanations",
    "Customizable Output Formats",
    "Language Agnostic",
    "Automated Refreshes",
    "Analytics and Insights",
    "Automated Code Summaries"
]
features_list = [row for row in features_df.to_dict(orient="records") if row["Feature"] in key_features]

def features_html():
    html = "<ul style='margin:0; padding-left:1.2em; font-size:16px; color:#f4f6fa;'>"
    for f in features_list:
        html += f"<li><b>{f['Feature']}</b>: {f['Description']}</li>"
    html += "</ul>"
    return html

model_name = "Salesforce/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

class CodeComplexityScorer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')
    def call(self, inputs):
        x = self.dense1(inputs)
        score = self.dense2(x)
        return score

complexity_model = CodeComplexityScorer()

def extract_code_features(code_text):
    length = len(code_text)
    lines = code_text.count('\n') + 1
    words = code_text.split()
    avg_word_len = sum(len(w) for w in words) / (len(words) + 1)
    features = tf.constant([[length/1000, lines/50, avg_word_len/20]], dtype=tf.float32)
    return features

LANG_PROMPTS = {
    "Python": "summarize Python code:",
    "JavaScript": "summarize JavaScript code:",
    "Java": "summarize Java code:",
    "Other": "summarize code:",
}

def automatic_code_analysis(code_text):
    return f"Code contains {code_text.count(chr(10))+1} lines and {len(code_text)} characters."

def context_aware_documentation(code_text):
    return "Generates context-aware, readable documentation (demo placeholder)."

def bug_issue_identification(code_text):
    return "No obvious issues detected (demo placeholder)."

def automated_code_summaries(code_text):
    return "Provides concise summaries of code modules (demo placeholder)."

feature_functions = {
    "Automatic Code Analysis": automatic_code_analysis,
    "Context-Aware Documentation": context_aware_documentation,
    "Bug/Issue Identification": bug_issue_identification,
    "Automated Code Summaries": automated_code_summaries,
}

def generate_documentation(code_text, language, export_format, selected_features):
    features = extract_code_features(code_text)
    complexity_score = complexity_model(features).numpy()[0][0]
    prompt = LANG_PROMPTS.get(language, LANG_PROMPTS["Other"])
    input_text = f"{prompt} {code_text.strip()}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=128, num_beams=5, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    extra_sections = ""
    for feature in selected_features:
        if feature in feature_functions:
            extra_sections += f"\n**{feature}:**\n{feature_functions[feature](code_text)}"
    doc_output = f"""### AI-Generated Documentation

{summary}

**Code Complexity Score:** {complexity_score:.2f} (0=low,1=high)
{extra_sections}
"""
    if export_format == "Markdown":
        return doc_output
    elif export_format == "PDF":
        pdf_filename = "/tmp/generated_doc.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in doc_output.split('\n'):
            pdf.cell(0, 10, txt=line, ln=True)
        pdf.output(pdf_filename)
        return pdf_filename
    else:
        return doc_output

def process_uploaded_file(uploaded_file, language, export_format, selected_features):
    code_bytes = uploaded_file.read()
    code_text = code_bytes.decode("utf-8", errors="ignore")
    return generate_documentation(code_text, language, export_format, selected_features)

# --- CSS: Use .gradio-container for full-page background image ---
custom_css = """
.gradio-container {
    background-image: url('https://media.istockphoto.com/photos/programming-code-abstract-technology-background-of-software-developer-picture-id1201405775?b=1&k=20&m=1201405775&s=170667a&w=0&h=XZ-tUfHvW5IRT30nMm7bAbbWrqkGQ-WT8XSS8Pab-eA=');
    background-repeat: no-repeat;
    background-position: center center;
    background-attachment: fixed;
    background-size: cover;
    min-height: 100vh;
}
#container {
    background: rgba(16, 24, 40, 0.85);
    border-radius: 22px;
    padding: 2.5rem 3.5rem;
    max-width: 900px;
    margin: 2rem auto 3rem auto;
    box-shadow: 0 12px 48px 0 rgba(60,120,220,0.28), 0 1.5px 12px 0 rgba(0,0,0,0.15);
    color: #f4f6fa !important;
    backdrop-filter: blur(7px);
    border: 2.5px solid rgba(0,255,255,0.10);
}
#animated-header {
    font-size: 2.6em !important;
    font-weight: 900;
    text-align: center;
    margin-bottom: 1em;
    background: linear-gradient(270deg, #00f2fe, #4facfe, #43e97b, #fa709a, #fee140, #00f2fe);
    background-size: 800% 800%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 12s ease-in-out infinite;
    letter-spacing: 2px;
    text-shadow: 0 2px 8px rgba(0,255,255,0.18);
}
@keyframes gradientShift {
    0%{background-position:0% 50%;}
    50%{background-position:100% 50%;}
    100%{background-position:0% 50%;}
}
#feature-panel {
    background: rgba(34, 49, 63, 0.95);
    border-radius: 14px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 18px rgba(0,255,255,0.10);
    max-height: 200px;
    overflow-y: auto;
    font-size: 1.13em;
    line-height: 1.5em;
    color: #f4f6fa !important;
    border: 2px solid #00f2fe;
    animation: fadeInUp 1.2s ease forwards, neon-glow 2.5s infinite alternate;
}
@keyframes fadeInUp {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}
@keyframes neon-glow {
    0% { box-shadow: 0 0 8px #00f2fe, 0 0 16px #00f2fe70; border-color: #00f2fe;}
    100% { box-shadow: 0 0 16px #43e97b, 0 0 32px #43e97b70; border-color: #43e97b;}
}
#generate-btn {
    background: linear-gradient(90deg, #43e97b, #38f9d7, #00f2fe);
    color: #192a56 !important;
    font-weight: 800;
    border-radius: 14px;
    padding: 0.9em 2.2em;
    font-size: 1.25em;
    border: none;
    box-shadow: 0 6px 24px 0 rgba(0,255,255,0.22);
    transition: all 0.3s cubic-bezier(.4,2,.6,1);
    letter-spacing: 1px;
    outline: none;
}
#generate-btn:hover {
    background: linear-gradient(90deg, #fa709a, #fee140);
    color: #192a56 !important;
    box-shadow: 0 8px 32px rgba(250,112,154,0.22);
    transform: scale(1.06);
    cursor: pointer;
}
#credits {
    text-align: center;
    margin-top: 2.5rem;
    font-size: 1.15em;
    color: #fee140;
    font-weight: 800;
    letter-spacing: 0.08em;
    animation: fadeIn 2s ease forwards;
    text-shadow: 0 2px 8px #fa709a50;
}
@media (max-width: 600px) {
    #container {
        padding: 1rem 0.5rem;
        margin: 1rem;
    }
    #animated-header {
        font-size: 1.4em !important;
    }
    #feature-panel {
        padding: 0.7rem 0.7rem;
        font-size: 1em;
    }
}
"""

with gr.Blocks(css=custom_css, elem_id="container") as demo:
    gr.HTML("<div id='animated-header'>AI-Powered Code Documentation Generator</div>")
    with gr.Row():
        gr.HTML(f"<div id='feature-panel'><b>Supported Features (scroll if needed):</b>{features_html()}</div>")
    file_input = gr.File(label="Upload Code File (.py, .js, .java)", file_types=[".py", ".js", ".java"])
    code_input = gr.Textbox(label="Or Paste Code Here", lines=8, max_lines=15, placeholder="Paste your code snippet here...")
    language_dropdown = gr.Dropdown(label="Select Language", choices=["Python", "JavaScript", "Java", "Other"], value="Python")
    export_dropdown = gr.Dropdown(label="Export Format", choices=["Markdown", "PDF"], value="Markdown")
    feature_options = gr.CheckboxGroup(
        label="Select Features to Include",
        choices=[f["Feature"] for f in features_list],
        value=["Automatic Code Analysis", "Context-Aware Documentation", "Bug/Issue Identification"],
        interactive=True,
        container=False,
        show_label=True,
    )
    generate_btn = gr.Button("Generate Documentation", elem_id="generate-btn")
    output_box = gr.Textbox(label="Generated Documentation", lines=10, max_lines=20, interactive=False, show_copy_button=True)
    pdf_output = gr.File(label="Download PDF", visible=False)
    gr.HTML("<div id='credits'>Credits: Sreelekha Putta</div>")

    def on_generate(file_obj, code_str, language, export_format, selected_features):
        if file_obj is not None:
            result = process_uploaded_file(file_obj, language, export_format, selected_features)
        elif code_str.strip() != "":
            result = generate_documentation(code_str, language, export_format, selected_features)
        else:
            return "Please upload a file or paste code to generate documentation.", None
        if export_format == "PDF":
            return None, gr.update(value=result, visible=True)
        else:
            return result, gr.update(visible=False)

    generate_btn.click(
        on_generate,
        inputs=[file_input, code_input, language_dropdown, export_dropdown, feature_options],
        outputs=[output_box, pdf_output]
    )

demo.launch(share=True)

