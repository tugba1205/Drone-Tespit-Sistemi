"""
Drone Tespit Sistemi
====================
YOLOv8 (görsel) + MFCC (ses) tabanlı çoklu veri analizi ile
gerçek zamanlı drone tespiti yapan Gradio web arayüzü.
"""

import gradio as gr

from camera_module import process_camera
from audio_module import process_audio
from yolo_module import yolo_detect
from mfcc_module import mfcc_analyze


# ──────────────────────────────────────────────────────────────
# Ana tespit fonksiyonu
# ──────────────────────────────────────────────────────────────

def dronedetection(image, audio):
    """
    Kamera görüntüsü ve ses verisini analiz ederek drone tespiti yapar.
    YOLO veya MFCC modüllerinden herhangi biri 'DRONE' döndürürse
    sistem pozitif sonuç verir.
    """
    camera_output = "Veri bekleniyor..."
    audio_output  = "Veri bekleniyor..."
    results       = []

    if image is not None:
        camera_result  = process_camera(image)
        yolo_result    = yolo_detect(image)
        camera_output  = f"{camera_result}\n\n{yolo_result}"
        results.extend([camera_result, yolo_result])

    if audio is not None:
        audio_result  = process_audio(audio)
        mfcc_result   = mfcc_analyze(audio)
        audio_output  = f"{audio_result}\n\n{mfcc_result}"
        results.extend([audio_result, mfcc_result])

    drone_detected = any("DRONE" in r.upper() for r in results)
    final_output   = "DRONE TESPİT EDİLDİ" if drone_detected else "Drone tespit edilemedi"

    return camera_output, audio_output, final_output, "🟢 Sistem Aktif"


# ──────────────────────────────────────────────────────────────
# Gradio tema — token'ları koyu arkaplan / sıfır border'a çek
# ──────────────────────────────────────────────────────────────

dark_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill            = "#050B1F",
    body_background_fill_dark       = "#050B1F",
    background_fill_primary         = "#070E22",
    background_fill_primary_dark    = "#070E22",
    background_fill_secondary       = "#070E22",
    background_fill_secondary_dark  = "#070E22",
    border_color_primary            = "#1a1a2e",
    border_color_primary_dark       = "#1a1a2e",
    input_background_fill           = "#070E22",
    input_background_fill_dark      = "#070E22",
    input_border_color              = "transparent",
    input_border_color_dark         = "transparent",
    input_border_color_focus        = "transparent",
    input_border_color_focus_dark   = "transparent",
    block_background_fill           = "#070E22",
    block_background_fill_dark      = "#070E22",
    block_border_color              = "transparent",
    block_border_color_dark         = "transparent",
    block_border_width              = "0px",
    block_label_text_color          = "#d1d5db",
    block_label_text_color_dark     = "#d1d5db",
    block_title_text_color          = "#ffffff",
    block_title_text_color_dark     = "#ffffff",
    block_shadow                    = "none",
    block_shadow_dark               = "none",
)


# ──────────────────────────────────────────────────────────────
# CSS — neon border'lar, karanlık iç alanlar
# ──────────────────────────────────────────────────────────────

css = """
html, body, .gradio-container, footer {
    background: #050B1F !important;
    margin: 0 !important;
    padding: 0 !important;
}
.gradio-container { max-width: 100% !important; }

/* Yazı renkleri */
h1 { color: #fff !important; font-size: 42px !important; font-weight: 700 !important; }
h2, h3 { color: #fff !important; }
p, label, span, .svelte-1gfkn6j { color: #d1d5db !important; }

/* Genel kutu sıfırlama */
.block, .form, .wrap, .gap, .row {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
input, textarea {
    background: #070E22 !important;
    color: #e2e8f0 !important;
}

/* Kamera — mavi neon */
#kutu-kamera {
    background: #070E22 !important;
    border: 2px solid #3B82F6 !important;
    border-radius: 16px !important;
    box-shadow: 0 0 12px #3B82F6, 0 0 30px rgba(59,130,246,.4) !important;
    overflow: hidden !important;
}
#kutu-kamera *:not(button):not(svg):not(path) {
    background: #070E22 !important;
    border-color: transparent !important;
    box-shadow: none !important;
}
#kutu-kamera canvas { background: #000 !important; }

/* Mikrofon — mor neon */
#kutu-mikrofon {
    background: #070E22 !important;
    border: 2px solid #8B5CF6 !important;
    border-radius: 16px !important;
    box-shadow: 0 0 12px #8B5CF6, 0 0 30px rgba(139,92,246,.4) !important;
    overflow: hidden !important;
}
#kutu-mikrofon *:not(button):not(svg):not(path):not(span.svelte-1gfkn6j) {
    background: #070E22 !important;
    border-color: transparent !important;
    box-shadow: none !important;
}
#kutu-mikrofon * { color: #fff !important; }
#kutu-mikrofon button,
#kutu-mikrofon button span,
#kutu-mikrofon button svg,
#kutu-mikrofon label,
#kutu-mikrofon span { color: #fff !important; fill: #fff !important; }

/* Mikrofon X butonu gizle */
#kutu-mikrofon button[title="Remove Audio"],
#kutu-mikrofon button[aria-label="Clear"],
#kutu-mikrofon button[aria-label="Remove Audio"],
#kutu-mikrofon button[aria-label="close"],
#kutu-mikrofon .clear,
#kutu-mikrofon > div > div > button:first-of-type,
#kutu-mikrofon .upload-container > button,
#kutu-mikrofon > div > button[style*="position: absolute"],
#kutu-mikrofon button.close { display: none !important; }

/* Kamera Analizi — indigo neon */
#kutu-kamera-analiz {
    background: #070E22 !important;
    border: 2px solid #6366F1 !important;
    border-radius: 16px !important;
    box-shadow: 0 0 10px #6366F1, 0 0 26px rgba(99,102,241,.35) !important;
    overflow: hidden !important;
}
#kutu-kamera-analiz textarea {
    background: #070E22 !important;
    color: #e2e8f0 !important;
    border: none !important;
    box-shadow: none !important;
}

/* Ses Analizi — mor neon */
#kutu-ses-analiz {
    background: #070E22 !important;
    border: 2px solid #8B5CF6 !important;
    border-radius: 16px !important;
    box-shadow: 0 0 10px #8B5CF6, 0 0 26px rgba(139,92,246,.35) !important;
    overflow: hidden !important;
}
#kutu-ses-analiz textarea {
    background: #070E22 !important;
    color: #e2e8f0 !important;
    border: none !important;
    box-shadow: none !important;
}

/* Nihai Sonuç — kırmızı neon */
#kutu-sonuc {
    background: #070E22 !important;
    border: 2px solid #EF4444 !important;
    border-radius: 16px !important;
    box-shadow: 0 0 10px #EF4444, 0 0 28px rgba(239,68,68,.35) !important;
    overflow: hidden !important;
}
#kutu-sonuc textarea {
    background: #070E22 !important;
    color: #FCA5A5 !important;
    border: none !important;
    box-shadow: none !important;
}

/* Sistem Durumu — cyan neon */
#kutu-sistem {
    background: #070E22 !important;
    border: 2px solid #06B6D4 !important;
    border-radius: 14px !important;
    box-shadow: 0 0 10px #06B6D4, 0 0 22px rgba(6,182,212,.3) !important;
    overflow: hidden !important;
}
#kutu-sistem textarea {
    background: #070E22 !important;
    color: #67E8F9 !important;
    border: none !important;
    box-shadow: none !important;
    font-weight: 600 !important;
}

/* Analizi Başlat butonu */
#btn-analiz {
    background: linear-gradient(90deg, #3B82F6, #8B5CF6, #D946EF) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    box-shadow: 0 0 18px rgba(59,130,246,.5), 0 0 32px rgba(217,70,239,.4) !important;
    transition: filter .2s ease, box-shadow .2s ease !important;
}
#btn-analiz:hover {
    filter: brightness(1.15) !important;
    box-shadow: 0 0 28px rgba(59,130,246,.7), 0 0 45px rgba(217,70,239,.6) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #050B1F; }
::-webkit-scrollbar-thumb { background: #8B5CF6; border-radius: 10px; }
"""


# ──────────────────────────────────────────────────────────────
# Arayüz
# ──────────────────────────────────────────────────────────────

with gr.Blocks(theme=dark_theme, css=css, title="Drone Detection System") as ui:

    with gr.Row():

        # Sol panel — sistem durumu
        with gr.Column(scale=1):
            gr.HTML("""
                <div style="text-align:center; padding:8px 0;">
                    <img src="/gradio_api/file=drone_logo.png" style="width:120px;">
                </div>
            """)
            system_status = gr.Textbox(
                value="🟢 Sistem Hazır",
                label="Sistem Durumu",
                interactive=False,
                max_lines=1,
                elem_id="kutu-sistem",
            )

        # Ana panel
        with gr.Column(scale=8):
            gr.Markdown("# Drone Tespit Sistemi\n### YOLOv8 + MFCC Tabanlı Çoklu Veri Analizi")

            with gr.Row():
                imageinput = gr.Image(
                    sources=["webcam"],
                    type="numpy",
                    label="Kamera Görüntüsü",
                    elem_id="kutu-kamera",
                )
                audioinput = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Mikrofon Sesi",
                    elem_id="kutu-mikrofon",
                )

            analyze_button = gr.Button("Analizi Başlat", elem_id="btn-analiz")

            gr.Markdown("## Analiz Sonuçları")

            with gr.Row():
                camera_box = gr.Textbox(label="Kamera Analizi", lines=10, elem_id="kutu-kamera-analiz")
                audio_box  = gr.Textbox(label="Ses Analizi",    lines=10, elem_id="kutu-ses-analiz")

            final_box = gr.Textbox(label="Nihai Sonuç", lines=3, elem_id="kutu-sonuc")

    analyze_button.click(
        fn=dronedetection,
        inputs=[imageinput, audioinput],
        outputs=[camera_box, audio_box, final_box, system_status],
    )

if __name__ == "__main__":
    ui.launch(allowed_paths=["."])