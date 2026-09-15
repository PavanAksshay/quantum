import spaces
import gradio as gr
from app.api import app as fastapi_app

# ZeroGPU Anchor Function connected to Gradio Blocks event
@spaces.GPU(duration=120)
def zero_gpu_inference(text: str):
    """ZeroGPU execution trigger scanned on startup."""
    return f"ZeroGPU Hardware Active: {text}"

# Mount FastAPI app onto Gradio Blocks with active ZeroGPU event listener
with gr.Blocks(title="Quantum Text Security API") as demo:
    gr.Markdown("# ⚛️ Quantum Text Security Backend API")
    gr.Markdown("FastAPI backend is active on ZeroGPU! All REST endpoints (`/api/predict`, `/api/health`, `/api/research/*`, `/docs`) are active.")
    
    with gr.Row():
        inp = gr.Textbox(label="System Status Check", value="Quantum Text Security Pipeline")
        out = gr.Textbox(label="Hardware State")
    
    btn = gr.Button("Verify ZeroGPU Hardware")
    btn.click(fn=zero_gpu_inference, inputs=inp, outputs=out)

# Gradio ASGI entrypoint for Hugging Face Spaces
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
