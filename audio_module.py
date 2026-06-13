import numpy as np

def process_audio(audio):
    if audio is None:
        return "(Ses Analizi) / Veri bekleniyor..."

    sample_rate, audio_data = audio
    if audio_data.size == 0:
        return "(Ses Analizi) / Sinyal boş."

    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    energy = np.mean(np.abs(audio_data))
    return f"(Sinyal Analizi) / Örnekleme Frekansı: {sample_rate} Hz | Sinyal Enerjisi: {energy:.4f}"
