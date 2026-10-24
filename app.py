"""
Small Gradio demo around predict.py.   python app.py   -> open http://127.0.0.1:7860
Runs fully offline once the model files exist.
"""
import gradio as gr
from predict import predict_image


def run(img):
    if img is None:
        return "no image", {}
    r = predict_image(img)
    return r["verdict"], r


demo = gr.Interface(
    fn=run,
    inputs=gr.Image(type="pil", label="chest X-ray (AP, child)"),
    outputs=[gr.Textbox(label="decision"), gr.JSON(label="details")],
    title="Arogya pneumonia screen (research prototype)",
    description="Screening aid only. Anything in the LOW CONFIDENCE band or REJECTED goes to a clinician. "
                "Not validated on adults, lateral views, or other hospitals' scanners.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
