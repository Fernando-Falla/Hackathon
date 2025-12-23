import re
import pickle
import numpy as np
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# Librerías necesarias para el modelo
import nltk
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# 1. Rutas de los Artefactos 
# Las rutas son relativas al archivo main.py
MODEL_PATH = 'lr_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

# Inicializar FastAPI
app = FastAPI(
    title="Análisis de Sentimiento (Baseline)",
    version="1.0.0",
    description="API para clasificar texto usando TF-IDF y Regresión Logística."
)

# 2. Variables Globales y Carga de NLTK 
lr_model: LogisticRegression = None
tfidf_vectorizer: TfidfVectorizer = None
label_encoder: LabelEncoder = None
spanish_stopwords: set = None

# Función para cargar stopwords de forma segura
def load_stopwords():
    """Carga las stopwords de NLTK y las devuelve."""
    global spanish_stopwords
    if spanish_stopwords is None:
        try:
            # Intenta encontrar las stopwords localmente
            nltk.data.find('corpora/stopwords')
        except nltk.downloader.DownloadError:
            # Si no están, las descarga (necesario si no se hizo en la imagen base)
            nltk.download('stopwords', quiet=True)
        spanish_stopwords = set(stopwords.words('spanish'))
    return spanish_stopwords


# 3. FUNCIÓN DE LIMPIEZA (IDÉNTICA AL ENTRENAMIENTO)
def clean_text(text: str) -> str:
    """Limpia el texto, eliminando ruido, puntuación y stopwords."""
    stopwords = load_stopwords() # Aseguramos que estén cargadas

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)      # URLs
    text = re.sub(r"\S+@\S+", "", text)             # emails
    text = re.sub(r"[^a-záéíóúñü\s]", " ", text)    # Caracteres especiales
    text = re.sub(r"\s+", " ", text).strip()        # Normalizar espacios

    # Remoción de Stopwords
    text_tokens = text.split()
    tokens_sin_stopwords = [
        word for word in text_tokens if word not in stopwords
    ]
    return " ".join(tokens_sin_stopwords)


# 4. CARGA DE MODELO Y ARTEFACTOS
@app.on_event("startup")
async def load_artifacts():
    """Carga los artefactos de ML al iniciar la aplicación."""
    global lr_model, tfidf_vectorizer, label_encoder

    try:
        # Cargamos los modelos serializados con pickle
        with open(MODEL_PATH, 'rb') as f:
            lr_model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)

        print("Artefactos de ML (TF-IDF + RegLog) cargados exitosamente.")
    except Exception as e:
        print(f"Error al cargar artefactos: {e}")
        # En un entorno de servicio, es crucial fallar si no se cargan los modelos.
        raise RuntimeError("Fallo crítico al cargar los artefactos.")


# 5. ESQUEMAS DE ENTRADA Y SALIDA (Pydantic)
class TextIn(BaseModel):
    """Define la estructura del JSON de entrada."""
    text: str = Field(..., example="Excelente servicio, volveremos sin duda.")

class PredictionOut(BaseModel):
    """Define la estructura del JSON de salida."""
    prevision: str = Field(..., example="positivo")
    probabilidad: float = Field(..., example=0.935)

# 6. ENDPOINT DE PREDICCIÓN (El path que se usará) 
@app.post("/predict/sentiment", response_model=PredictionOut)
async def predict_sentiment(data: TextIn):
    """
    Clasifica el sentimiento de un texto y devuelve la predicción
    y la probabilidad máxima en el formato simplificado.
    """
    if lr_model is None:
        raise HTTPException(status_code=503, detail="El modelo aún no está listo o falló al cargar.")

    # A. Limpieza
    cleaned_text = clean_text(data.text)

    # B. Vectorización
    X_new = tfidf_vectorizer.transform([cleaned_text])

    # C. Predicción de Probabilidades
    predictions_proba = lr_model.predict_proba(X_new)[0]

    # D. Post-procesamiento
    # Obtener el índice de la clase con mayor probabilidad
    predicted_class_index = np.argmax(predictions_proba)

    # Obtener la probabilidad máxima (la 'probabilidad' de la solicitud)
    probabilidad_maxima = float(predictions_proba[predicted_class_index])

    # Convertir el índice a la etiqueta de sentimiento ('prevision' de la solicitud)
    prevision_sentimiento = label_encoder.inverse_transform([predicted_class_index])[0]

    # E. Retorno simplificado según el contrato
    return PredictionOut(
        prevision=prevision_sentimiento,
        probabilidad=probabilidad_maxima,
    )
