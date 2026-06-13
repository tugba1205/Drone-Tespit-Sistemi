import os
import joblib
import librosa
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "drone_audio_model.pkl")

# Gelişmiş Dinamik Konum Kontrolü
if not os.path.exists(MODEL_PATH) and os.path.exists("drone_audio_model.pkl"):
    MODEL_PATH = "drone_audio_model.pkl"

if os.path.exists(MODEL_PATH):
    sound_model = joblib.load(MODEL_PATH)
    model_loaded = True
else:
    sound_model = None
    model_loaded = False


def mfcc_analyze(audio):
    if audio is None:
        return False, "[MFCC] Akış bekleniyor..."

    sample_rate, audio_data = audio
    if audio_data.size == 0:
        return False, "[MFCC] Boş ses paketi."

    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    if len(audio_data) < 2048:
        return False, "[MFCC] Ses verisi analiz için henüz yetersiz uzunlukta..."

    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_data = audio_data.astype(np.float32)

    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        mfcc_features = np.mean(mfccs, axis=1).reshape(1, -1)
        mfcc_mean_all = np.mean(mfcc_features)
    except Exception as e:
        return False, f"[MFCC Analiz Hatası]: {str(e)}"

    drone_detected = False

    if model_loaded and sound_model is not None:
        try:
            prediction = sound_model.predict(mfcc_features)[0]
            if prediction == 1:
                drone_detected = True
                result_status = "🚨 DRONE SESİ FREKANSI ALGILANDI!"
            else:
                result_status = "Temiz (Çevre Sesi)"
        except:
            result_status = "Tahmin yürütülürken hata oluştu."
    else:
        result_status = "Yapay Zeka Modeli Eksik! (Simülasyon Modu)"
        if 15 < abs(mfcc_mean_all) < 35:
            drone_detected = False

    result = f"""// (Akustik Analiz Raporu) //
    Durum: {result_status}
    Anlık Spektrum Ortalaması: {mfcc_mean_all:.2f}
    Model Tipi: Random Forest Classifier (.pkl)
    """

    return drone_detected, result
