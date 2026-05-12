import librosa
import numpy as np


def mfcc_analyze(audio):

    sample_rate, audio_data = audio

    # Stereo ise mono yap
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # MFCC çıkarımı
    mfccs = librosa.feature.mfcc(
        y=audio_data.astype(float),
        sr=sample_rate,
        n_mfcc=13
    )

    # Ortalama MFCC değeri
    mfcc_mean = np.mean(mfccs)

    result = f"""[MFCC ANALİZİ]
MFCC Ortalama: {mfcc_mean:.2f}
"""

    return result