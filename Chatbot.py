from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import gradio as gr

# Load model (lightweight)
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cpu")  # Runs safely on low-end systems

# Inference function with graceful exit
def logical_reasoning_bot(question):
    if question.strip().lower() in ["exit", "quit", "bye", "stop"]:
        return "👋 Thank you for using the Logical Reasoning Chatbot! You can now close this tab."
    
    prompt = f"Answer the question logically: {question}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio interface
interface = gr.Interface(
    fn=logical_reasoning_bot,
    inputs=gr.Textbox(label="🧠 Ask a Logical Question"),
    outputs=gr.Textbox(label="🤖 Reasoned Answer"),
    title="Logical Reasoning Chatbot",
    description="💡 Ask logic-based questions.\n\nType 'exit', 'quit', or 'bye' to end the session politely."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()

