import numpy as np

def process_audio(audio):   # Ses verisinin ortalama enerjisini hesaplar.

    if audio is None:
        return "(Ses Analizi) Veri bekleniyor..."

    sample_rate, audio_data = audio

    # Kanalları birleştirme (Stereo -> Mono)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Genlik normalizasyonu (int formatından float32'ye)
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_data = audio_data.astype(np.float32)

    # Ses enerjisi (Kök Ortalama Kare - RMS benzeri basitleştirilmiş mutlak ortalama)
    energy = np.mean(np.abs(audio_data))

    result = f"""[Ses Enerji Analizi]
    Sample Rate: {sample_rate} Hz
    Sinyal Enerjisi: {energy:.4f}
    """

    return result
