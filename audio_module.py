import numpy as np


def process_audio(audio):

    sample_rate, audio_data = audio

    # Ses enerjisi
    energy = np.mean(np.abs(audio_data))

    result = f"""[SES ANALİZİ]
Sample Rate: {sample_rate}
Enerji: {energy:.2f}
"""

    return result