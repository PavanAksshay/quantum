import gradio as gr
import uvicorn
from app.api import app as fastapi_app

# ZeroGPU Compatibility Check
try:
    import spaces

    @spaces.GPU
    def zero_gpu_anchor():
        """Satisfies ZeroGPU startup scanner requirement."""
        return "ZeroGPU Initialized"
except Exception:
    spaces = None

# Mount the FastAPI backend onto Gradio
with gr.Blocks(title="Quantum Text Security API") as demo:
    gr.Markdown("# ⚛️ Quantum Text Security Backend API")
    gr.Markdown("FastAPI backend is active! All REST endpoints (`/api/predict`, `/api/health`, `/api/research/*`, `/docs`) are running.")

app_with_gradio = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app_with_gradio, host="0.0.0.0", port=7860)
