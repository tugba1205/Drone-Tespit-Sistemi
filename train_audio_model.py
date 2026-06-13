"""
Geliştirilmiş Akustik Simülasyon ve Model Eğitici
Gerçek dünya MFCC spektrum genliklerine uyumlu Random Forest Eğitimi
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

print("Gerçekçi akustik spektrum verileri simüle ediliyor...")

# 1. GERÇEKÇİ DRONE SESİ FREKANS SPEKTRUMU (Sınıf: 1)
# Gerçek drone seslerinde MFCC'nin ilk katsayıları negatif ve belirli bir harmonik dar aralıktadır.
drone_samples = 500
drone_features = np.zeros((drone_samples, 13))
for i in range(drone_samples):
    # Drone pervanesinin baskın frekans yapısı simüle ediliyor
    drone_features[i, 0] = np.random.normal(loc=-40.0, scale=10.0)  # Enerji seviyesi
    drone_features[i, 1:] = np.random.normal(loc=15.0, scale=8.0, size=12)  # Harmonikler

drone_labels = np.ones(drone_samples)

# 2. GERÇEKÇİ ORTAM / ÇEVRE GÜRÜLTÜSÜ SPEKTRUMU (Sınıf: 0)
# Beyaz gürültü, rüzgar veya sessizlik durumundaki geniş/dağınık spektrum
ambient_samples = 500
ambient_features = np.zeros((ambient_samples, 13))
for i in range(ambient_samples):
    ambient_features[i, 0] = np.random.normal(loc=-120.0, scale=25.0)  # Çok daha düşük veya dağınık enerji
    ambient_features[i, 1:] = np.random.normal(loc=-5.0, scale=15.0, size=12)  # Düzensiz harmonikler

ambient_labels = np.zeros(ambient_samples)

# Verileri birleştirme
X = np.vstack((drone_features, ambient_features))
y = np.concatenate((drone_labels, ambient_labels))

print("Gelişmiş Random Forest Classifier modeli eğitiliyor...")
# predict_proba'nın daha kararlı çalışması için n_estimators artırıldı ve min_samples_leaf eklendi
model = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=42)
model.fit(X, y)

MODEL_PATH = "drone_audio_model.pkl"
joblib.dump(model, MODEL_PATH)

print(f"BAŞARILI: Realist '{MODEL_PATH}' modeli sıfırdan eğitildi ve başarıyla kaydedildi!")
