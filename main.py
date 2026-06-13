"""
Siber Drone Tespit Sistemi - Ana Kontrol Merkezi
================================================
Gradio v6.0 ve Çift Yapay Zeka Modeli (YOLOv8x + Random Forest) Entegre Canlı Akış Sistemi

Programı çalıştırmak için README.md dosyasını referans alınız!
"""

import gradio as gr

from camera_module import process_camera
from audio_module import process_audio
from yolo_module import yolo_detect
from mfcc_module import mfcc_analyze


# ──────────────────────────────────────────────────────────────
# Gerçek Zamanlı Füzyon Karar Fonksiyonu
# ──────────────────────────────────────────────────────────────

def realtime_detection(image, audio):
    camera_output = "Optik veri bekleniyor..."
    audio_output = "Akustik veri bekleniyor..."

    drone_by_visual = False
    drone_by_audio = False

    # 1. Optik Kanal İşleme (YOLOv8x)
    if image is not None:
        camera_result = process_camera(image)
        yolo_is_drone, yolo_result = yolo_detect(image)
        camera_output = f"{camera_result}\n{yolo_result}"
        if yolo_is_drone:
            drone_by_visual = True

    # 2. Akustik Kanal İşleme (MFCC + ML)
    if audio is not None:
        audio_result = process_audio(audio)
        mfcc_is_drone, mfcc_result = mfcc_analyze(audio)
        audio_output = f"{audio_result}\n{mfcc_result}"
        if mfcc_is_drone:
            drone_by_audio = True

    # 3. Ortak Siber Karar Mekanizması
    if drone_by_visual or drone_by_audio:
        final_output = " Drone Tespit Edildi!"
    else:
        final_output = "Drone Tespit Edilemedi."

    return camera_output, audio_output, final_output, "Sistem Canlı / Aktif"


# ──────────────────────────────────────────────────────────────
# Stil ve Tema Konfigürasyonları
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
    block_shadow="none",
)

css = """
html, body, .gradio-container { background: #050B1F !important; }
h1 { color: #fff !important; font-size: 36px !important; font-weight: 700 !important; }
h3 { color: #8B5CF6 !important; margin-bottom: 20px;}
textarea { background: #070E22 !important; color: #e2e8f0 !important; }

#kutu-kamera { border: 2px solid #3B82F6 !important; border-radius: 16px !important; box-shadow: 0 0 15px rgba(59,130,246,.4) !important; }
#kutu-mikrofon { border: 2px solid #8B5CF6 !important; border-radius: 16px !important; box-shadow: 0 0 15px rgba(139,92,246,.4) !important; }
#kutu-kamera-analiz { border: 1px solid #6366F1 !important; border-radius: 12px !important; }
#kutu-ses-analiz { border: 1px solid #8B5CF6 !important; border-radius: 12px !important; }
#kutu-sonuc { border: 2px solid #EF4444 !important; border-radius: 16px !important; box-shadow: 0 0 20px rgba(239,68,68,.4) !important; }
#kutu-sonuc textarea { color: #FCA5A5 !important; font-weight: bold !important; font-size: 22px !important; text-align: center !important; }
#kutu-sistem { border: 2px solid #06B6D4 !important; border-radius: 14px !important; }
#kutu-sistem textarea { color: #67E8F9 !important; font-weight: 600 !important; text-align: center !important; }
"""

# ──────────────────────────────────────────────────────────────
# Gradio v6 Arayüz İnşası
# ──────────────────────────────────────────────────────────────

with gr.Blocks(title="Real-Time Drone Detection") as ui:
    # Logo ve Başlık Düzeni
    with gr.Row():
        with gr.Column(scale=2, min_width=150):
            gr.HTML("""
                <div style="text-align: center; padding: 10px;">
                    <img src="file=drone_logo.png" style="max-width: 140px; width: 100%; border-radius: 12px; box-shadow: 0 0 15px rgba(139,92,246,0.3);">
                </div>
            """)
        with gr.Column(scale=8):
            gr.Markdown("<h1 style='text-align: left; margin-top: 15px;'> Drone Tespit Merkezi </h1>")
            gr.Markdown(
                "<h3 style='text-align: left; color: #8B5CF6;'>Kesintisiz Optik ve Akustik Spektrum Taraması</h3>")

    with gr.Row():
        # Sol Panel: Girişler (Streaming Modu)
        with gr.Column(scale=5):
            imageinput = gr.Image(
                sources=["webcam"],
                type="numpy",
                label="Canlı Kamera Akışı",
                elem_id="kutu-kamera",
                streaming=True
            )
            audioinput = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Canlı Mikrofon Akışı",
                elem_id="kutu-mikrofon",
                streaming=True
            )

        # Sağ Panel: Sonuçlar ve Füzyon Alanı
        with gr.Column(scale=5):
            system_status = gr.Textbox(
                value="Sistem Aktif, Taramalar Sürüyor",
                label="Sistem Sağlık Durumu",
                interactive=False,
                elem_id="kutu-sistem",
            )

            final_box = gr.Textbox(label="Füzyon Karar Merkezi (Real-Time Anlık)", lines=2, elem_id="kutu-sonuc")

            with gr.Row():
                camera_box = gr.Textbox(label="Optik Rapor (YOLOv8x)", lines=6, elem_id="kutu-kamera-analiz")
                audio_box = gr.Textbox(label="Akustik Rapor (MFCC/ML Model)", lines=6, elem_id="kutu-ses-analiz")

    # Canlı Tetikleyiciler
    imageinput.change(
        fn=realtime_detection,
        inputs=[imageinput, audioinput],
        outputs=[camera_box, audio_box, final_box, system_status]
    )

    audioinput.stream(
        fn=realtime_detection,
        inputs=[imageinput, audioinput],
        outputs=[camera_box, audio_box, final_box, system_status]
    )

if __name__ == "__main__":
    ui.launch(theme=dark_theme, css=css, allowed_paths=["."])
