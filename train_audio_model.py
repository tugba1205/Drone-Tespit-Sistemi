import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

print("Sensör verileri simüle ediliyor ve akustik veri seti hazırlanıyor...")

# Drone Pervane Frekans Spektrumu (Sınıf: 1)
drone_sound_features = np.random.normal(loc=25.0, scale=5.0, size=(100, 13))
drone_labels = np.ones(100)

# Ortam / Çevre Gürültüsü Spektrumu (Sınıf: 0)
ambient_sound_features = np.random.normal(loc=-10.0, scale=12.0, size=(100, 13))
ambient_labels = np.zeros(100)

X = np.vstack((drone_sound_features, ambient_sound_features))
y = np.concatenate((drone_labels, ambient_labels))

print("Makine Öğrenmesi modeli (Random Forest Classifier) eğitiliyor...")
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

MODEL_PATH = "drone_audio_model.pkl"
joblib.dump(model, MODEL_PATH)

print(f"BAŞARILI: '{MODEL_PATH}' modeli sıfırdan eğitildi ve başarıyla kaydedildi!")
