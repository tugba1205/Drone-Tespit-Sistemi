import gradio as gr

from camera_module import process_camera
from audio_module import process_audio
from yolo_module import yolo_detect
from mfcc_module import mfcc_analyze


def dronedetection(image, audio):

    results = []

    # Kamera kontrolü
    if image is not None:
        camera_result = process_camera(image)
        yolo_result = yolo_detect(image)

        results.append(camera_result)
        results.append(yolo_result)

    # Ses kontrolü
    if audio is not None:
        audio_result = process_audio(audio)
        mfcc_result = mfcc_analyze(audio)

        results.append(audio_result)
        results.append(mfcc_result)

    # Nihai karar
    drone_detected = False

    for result in results:
        if "DRONE" in result:
            drone_detected = True

    if drone_detected:
        final_result = "DRONE TESPİT EDİLDİ!\n\n"
    else:
        final_result = "Drone tespit edilemedi.\n\n"

    final_result += "\n".join(results)

    return final_result


# Arayüz
with gr.Blocks() as ui:

    gr.Markdown("# Drone Tespit Sistemi")
    gr.Markdown("Kamera + Ses + YOLO + MFCC Analizi")

    with gr.Row():

        imageinput = gr.Image(
            sources=["webcam"],
            type="numpy",
            label="Kamera Görüntüsü"
        )

        audioinput = gr.Audio(
            sources=["microphone"],
            type="numpy",
            label="Mikrofon Sesi"
        )

    button = gr.Button("Analize Başla")

    output = gr.Textbox(
        label="Analiz Sonucu",
        lines=15
    )

    button.click(
        fn=dronedetection,
        inputs=[imageinput, audioinput],
        outputs=output
    )

ui.launch()