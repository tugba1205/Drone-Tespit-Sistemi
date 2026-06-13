import os
import joblib
import librosa
import numpy as np

MODEL_PATH = "drone_audio_model.pkl"

if os.path.exists(MODEL_PATH):
    sound_model = joblib.load(MODEL_PATH)
else:
    sound_model = None

def mfcc_analyze(audio):
    """
    Anlık akan ses parçalarını işler. Kısa paket hatalarına karşı korumalıdır.
    """
    if audio is None:
        return False, "[MFCC] Akış bekleniyor..."

    sample_rate, audio_data = audio

    # Boş veya sinyalsiz paket kontrolü
    if audio_data.size == 0:
        return False, "[MFCC] Boş ses paketi."

    # Stereo -> Mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # GERÇEK ZAMANLI KONTROL:
    # Ses paketi librosa'nın varsayılan n_fft (2048) değerinden küçükse hata vermemesi için pas geç
    if len(audio_data) < 2048:
        return False, "[MFCC] Ses verisi analiz için henüz yetersiz uzunlukta..."

    # Normalizasyon
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_data = audio_data.astype(np.float32)

    try:
        # MFCC Özellik çıkarımı
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        mfcc_features = np.mean(mfccs, axis=1).reshape(1, -1)
        mfcc_mean_all = np.mean(mfcc_features)
    except Exception as e:
        # Gerçek zamanlı akışın kopmaması için hatayı yakala ama sistemi çökertme
        return False, f"[MFCC Anlık Hata]: {str(e)}"

    drone_detected = False
    result_status = "Temiz (Çevre Sesi)"

    if sound_model is not None:
        try:
            prediction = sound_model.predict(mfcc_features)[0]
            if prediction == 1:
                drone_detected = True
                result_status = "🚨 DRONE SESİ FREKANSI!"
        except:
            result_status = "Tahmin hatası."
    else:
        # Model yoksa simülasyon eşiği
        if 15 < abs(mfcc_mean_all) < 35:
            drone_detected = False

    result = f"""[Akustik Analiz]
    Durum: {result_status}
    Anlık Spektrum Ort: {mfcc_mean_all:.2f}
    """

    return drone_detected, result
