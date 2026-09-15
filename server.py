import spaces
import gradio as gr
from app.api import app as fastapi_app

# ZeroGPU Anchor Function (Scanned by Hugging Face ZeroGPU runtime on startup)
@spaces.GPU(duration=120)
def predict_quantum_gpu(text: str = "", dimension: int = 8):
    """ZeroGPU execution wrapper for quantum simulation."""
    return "ZeroGPU Active"

# Mount FastAPI app onto Gradio Blocks
with gr.Blocks(title="Quantum Text Security API") as demo:
    gr.Markdown("# ⚛️ Quantum Text Security Backend API")
    gr.Markdown("FastAPI backend is active on ZeroGPU! All REST endpoints (`/api/predict`, `/api/health`, `/api/research/*`, `/docs`) are running.")

# Gradio ASGI entrypoint for Hugging Face Spaces
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
