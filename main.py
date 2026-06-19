"""
Siber Drone Tespit Sistemi - Kesintisiz Füzyon Kontrol Merkezi
===========================================================
"""

import os
import gradio as gr
import numpy as np
from collections import deque

from camera_module import process_camera
from audio_module import process_audio
from yolo_module import yolo_detect
from mfcc_module import mfcc_analyze

current_dir = os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────────────────────────────────────────
# Küresel Durum Yönetimi ve Zaman Penceresi (Buffer)
# ──────────────────────────────────────────────────────────────
class AdvancedSystemState:
    audio_buffer = deque(maxlen=5)
    visual_probability = 0.0
    audio_probability = 0.0
    camera_output = "Optik spektrum bekleniyor..."
    audio_output = "Akustik spektrum bekleniyor..."


def calculate_fusion_score():
    """
    Siber Karar Mekanizması (Ağırlıklı Füzyon)
    Görsel Kanal Ağırlığı: %60 | Akustik Kanal Ağırlığı: %40
    """
    W_VISUAL = 0.60
    W_AUDIO = 0.40
    THRESHOLD = 0.45  # %45 güven eşiği

    total_score = (AdvancedSystemState.visual_probability * W_VISUAL) + \
                  (AdvancedSystemState.audio_probability * W_AUDIO)

    if total_score >= THRESHOLD:
        final_output = f"Drone Algılandı! (Sistem Füzyon Güven Oranı: %{total_score * 100:.1f})"
    else:
        final_output = f"Drone Unsurlarına Rastlanmadı. (Sistem Güven Oranı: %{total_score * 100:.1f})"

    status_text = f"Optik Güven: %{AdvancedSystemState.visual_probability * 100:.1f} | Akustik Güven: %{AdvancedSystemState.audio_probability * 100:.1f}"

    return (
        AdvancedSystemState.camera_output,
        AdvancedSystemState.audio_output,
        final_output,
        status_text
    )


def process_only_camera(image):
    if image is None:
        return calculate_fusion_score()

    camera_result = process_camera(image)
    yolo_is_drone, yolo_result = yolo_detect(image)

    AdvancedSystemState.camera_output = f"{camera_result}\n{yolo_result}"

    if yolo_is_drone:
        try:
            # Metinden güven skorunu güvenli bir şekilde ayıkla
            if "%" in yolo_result:
                conf_str = yolo_result.split("%")[-1].replace(")", "").strip()
                AdvancedSystemState.visual_probability = float(conf_str) / 100.0
            else:
                AdvancedSystemState.visual_probability = 0.85
        except:
            AdvancedSystemState.visual_probability = 0.85
    else:
        AdvancedSystemState.visual_probability = 0.0

    return calculate_fusion_score()


def process_only_audio(audio):
    if audio is None:
        return calculate_fusion_score()

    sample_rate, audio_data = audio
    if audio_data.size == 0:
        return calculate_fusion_score()

    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    AdvancedSystemState.audio_buffer.append(audio_data)
    combined_audio_data = np.concatenate(list(AdvancedSystemState.audio_buffer))

    audio_result = process_audio((sample_rate, combined_audio_data))
    mfcc_is_drone, mfcc_result, confidence_score = mfcc_analyze((sample_rate, combined_audio_data))

    AdvancedSystemState.audio_output = f"{audio_result}\n{mfcc_result}"
    AdvancedSystemState.audio_probability = confidence_score

    return calculate_fusion_score()


# ──────────────────────────────────────────────────────────────
# Arayüz Yapılandırması
# ──────────────────────────────────────────────────────────────
dark_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="#050B1F",
    background_fill_primary="#070E22",
    background_fill_secondary="#070E22",
    block_background_fill="#070E22",
    block_border_width="0px",
    block_label_text_color="#d1d5db",
    block_title_text_color="#ffffff",
)

css = """
html, body, .gradio-container { background: #050B1F !important; }
h1 { color: #fff !important; font-size: 36px !important; font-weight: 700 !important; }
h3 { color: #8B5CF6 !important; margin-bottom: 20px;}
textarea { background: #070E22 !important; color: #e2e8f0 !important; }
#kutu-kamera { border: 2px solid #3B82F6 !important; border-radius: 16px !important; }
#kutu-mikrofon { border: 2px solid #8B5CF6 !important; border-radius: 16px !important; }
#kutu-sonuc { border: 2px solid #EF4444 !important; border-radius: 16px !important; box-shadow: 0 0 20px rgba(239,68,68,.4) !important; }
#kutu-sonuc textarea { color: #FCA5A5 !important; font-weight: bold !important; font-size: 20px !important; text-align: center !important; }
#kutu-sistem { border: 2px solid #06B6D4 !important; border-radius: 14px !important; }
#kutu-sistem textarea { color: #67E8F9 !important; font-weight: 600 !important; text-align: center !important; }
"""

with gr.Blocks(title="Real-Time Drone Detection") as ui:
    with gr.Row():
        gr.Markdown("<h1 style='text-align: center;'> Siber Drone Tespit ve Savunma Merkezi </h1>")

    with gr.Row():
        with gr.Column(scale=5):
            imageinput = gr.Image(sources=["webcam"], type="numpy", label="Canlı Kamera Akışı", elem_id="kutu-kamera",
                                  streaming=True)
            audioinput = gr.Audio(sources=["microphone"], type="numpy",
                                  label="Canlı Mikrofon Akışı (Stereo Mix Uyumlu)", elem_id="kutu-mikrofon",
                                  streaming=True)

        with gr.Column(scale=5):
            system_status = gr.Textbox(value="Sistem Aktif", label="Anlık Sensör Analiz Matrisi", interactive=False,
                                       elem_id="kutu-sistem")
            final_box = gr.Textbox(label="Füzyon Karar Merkezi (Ağırlıklı Karar Teorisi)", lines=2,
                                   elem_id="kutu-sonuc")

            with gr.Row():
                camera_box = gr.Textbox(label="Optik Rapor (YOLOv8x)", lines=6)
                audio_box = gr.Textbox(label="Akustik Rapor (Buffer + MFCC)", lines=6)

    imageinput.change(fn=process_only_camera, inputs=[imageinput],
                      outputs=[camera_box, audio_box, final_box, system_status])
    audioinput.stream(fn=process_only_audio, inputs=[audioinput],
                      outputs=[camera_box, audio_box, final_box, system_status])

if __name__ == "__main__":
    # Gradio v6 standardına uygun olarak theme ve css launch içerisine alındı
    ui.launch(theme=dark_theme, css=css, allowed_paths=[current_dir])
