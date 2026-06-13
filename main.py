"""
Real-Time Drone Tespit Sistemi (Gradio 6.0 Uyumlu)
==================================================
Kamera ve mikrofondan gelen anlık veri akışlarını
butonsuz, canlı olarak analiz eden güncel Gradio arayüzü.
"""

import gradio as gr

from camera_module import process_camera
from audio_module import process_audio
from yolo_module import yolo_detect
from mfcc_module import mfcc_analyze


# ──────────────────────────────────────────────────────────────
# Gerçek Zamanlı Tespit Fonksiyonu
# ──────────────────────────────────────────────────────────────

def realtime_detection(image, audio):
    """
    Kamera veya mikrofondan gelen her yeni veride otomatik tetiklenir.
    """
    camera_output = "Optik veri bekleniyor..."
    audio_output = "Akustik veri bekleniyor..."

    drone_by_visual = False
    drone_by_audio = False

    # 1. Canlı Görüntü Analizi
    if image is not None:
        camera_result = process_camera(image)
        yolo_is_drone, yolo_result = yolo_detect(image)
        camera_output = f"{camera_result}\n{yolo_result}"
        if yolo_is_drone:
            drone_by_visual = True

    # 2. Canlı Ses Analizi
    if audio is not None:
        audio_result = process_audio(audio)
        mfcc_is_drone, mfcc_result = mfcc_analyze(audio)
        audio_output = f"{audio_result}\n{mfcc_result}"
        if mfcc_is_drone:
            drone_by_audio = True

    # 3. Anlık Füzyon Kararı
    if drone_by_visual or drone_by_audio:
        final_output = "ALARM: DRONE TESPİT EDİLDİ!"
    else:
        final_output = "Tarama Temiz. Tehdit Yok."

    return camera_output, audio_output, final_output, "Sistem Canlı/Aktif"


# ──────────────────────────────────────────────────────────────
# Gradio Tema ve Stil Ayarları
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
h1 { color: #fff !important; font-size: 36px !important; font-weight: 700 !important; text-align: center;}
h3 { color: #8B5CF6 !important; text-align: center; margin-bottom: 20px;}
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
# Arayüz Yapısı (Gradio 6.0 Düzenlemesi)
# ──────────────────────────────────────────────────────────────

# theme ve css parametreleri Blocks constructor'ından kaldırıldı
with gr.Blocks(title="Real-Time Drone Detection") as ui:
    gr.Markdown("# ️SİBER DRONE TESPİT MERKEZİ")
    gr.Markdown("### Kesintisiz Optik ve Akustik Spektrum Taraması")

    with gr.Row():
        # Sol Panel: Girişler
        with gr.Column(scale=5):
            imageinput = gr.Image(
                sources=["webcam"],
                type="numpy",
                label="CANLI KAMERA AKIŞI",
                elem_id="kutu-kamera",
                streaming=True
            )
            audioinput = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="CANLI MİKROFON AKIŞI",
                elem_id="kutu-mikrofon",
                streaming=True
            )

        # Sağ Panel: Analiz ve karar
        with gr.Column(scale=5):
            system_status = gr.Textbox(
                value="Sistem Başlatılıyor...",
                label="Sistem Sağlık Durumu",
                interactive=False,
                elem_id="kutu-sistem",
            )

            final_box = gr.Textbox(label="FÜZYON KARAR MERKEZİ (ANLIK)", lines=2, elem_id="kutu-sonuc")

            with gr.Row():
                camera_box = gr.Textbox(label="Optik Rapor (YOLOv8)", lines=6, elem_id="kutu-kamera-analiz")
                audio_box = gr.Textbox(label="Akustik Rapor (MFCC/ML)", lines=6, elem_id="kutu-ses-analiz")

    # ──────────────────────────────────────────────────────────────
    # GERÇEK ZAMANLI TETİKLEME MEKANİZMASI
    # ──────────────────────────────────────────────────────────────

    # Hata veren 'every=0.1' kaldırıldı.
    # Gradio 6'da canlı akışlar (streaming=True) event bazlı olarak otomatik ve verimli yönetilir.
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
    # theme ve css parametreleri artık launch() içinde iletiliyor.
    ui.launch(theme=dark_theme, css=css)
