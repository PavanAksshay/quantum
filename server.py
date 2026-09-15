import gradio as gr
from app.api import app as fastapi_app

# Interactive Status Interface
with gr.Blocks(title="Quantum Text Security API") as demo:
    gr.Markdown("# ⚛️ Quantum Text Security API Backend")
    gr.Markdown("✅ **FastAPI Backend is Active & Healthy**")
    gr.Markdown("All REST API endpoints (`/api/predict`, `/api/health`, `/api/research/*`, `/docs`) are live.")
    
    with gr.Row():
        test_input = gr.Textbox(label="Test Input", value="Subject: Urgent Verification Required")
        test_status = gr.Textbox(label="Backend State", value="API Active")

# Mount FastAPI onto Gradio
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
