import gradio as gr
import numpy as np

# Prototip analiz fonksiyonu

def drone_detection(image,audio):
    if image is None and audio is None:
        return "Veri Yok!"

    # ...

    image_score=0.6 if not None else 0
    audio_score=0.4 if not None else 0
    final_score=(image_score+audio_score) /2

    if final_score>0.5:
        return "DRONE TESPİT EDİLDİ!"
    else:
        return "Drone tespit edilemedi"

# Arayüz

with gr.Blocks() as ui:
    gr.Markdown("Drone Tespit Sistemi")
    gr.Markdown("Kamera ve ses verisi ile drone analizi (taslak versiyon)")

    with gr.Row():
        image_input=gr.Image(sources=["webcam"],type="numpy", label="Görüntü" )
        audio_input=gr.Audio(sources=["microphone"],type="numpy",label="Ses")

    button=gr.Button("Analize Başla")

    output=gr.Textbox(label="Sonuç:",lines=2)

    button.click(fn=drone_detection, inputs=[image_input,audio_input],outputs=[output])

    ui.launch()