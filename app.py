import gradio as gr
from fastai.vision.all import *

model = load_learner(".models/export.pkl")

def classify_image(img):
    pred, idx, probs = model.predict(img)
    return dict(zip(model.dls.vocab, map(float, probs)))

image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()

interface = gr.Interface(fn=classify_image, inputs=image, outputs=label)
interface.launch(inline=False)