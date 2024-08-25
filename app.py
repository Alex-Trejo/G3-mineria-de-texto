import pandas as pd
import streamlit as st
from gensim.models.phrases import Phraser
from nltk.tokenize import word_tokenize
import spacy
from transformers import pipeline

# Crear un pipeline de resumen
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Crear un pipeline de análisis de sentimientos
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

st.title("Detección de Colocaciones, Entidades Clave y Análisis de Sentimiento")

# Ingresar texto
text = st.text_area("Ingrese un texto:", height=200)

# Definir el límite de longitud para el resumen
max_length = 1024

if (input_text := text):
    # Verificar la longitud del texto
    if len(word_tokenize(input_text)) > max_length:
        st.write(f"El texto es demasiado largo (más de {max_length} tokens). Por favor, ingrese un texto más corto.")
    else:
        # Cargar el modelo de bigramas
        to_collocations = Phraser.load('bigram_model')

        # Tokenizar el texto ingresado
        tokens = word_tokenize(input_text.lower())

        # Aplicar el modelo de bigramas
        colocaciones = to_collocations[tokens]

        # Mostrar las colocaciones detectadas
        st.subheader("Colocaciones Detectadas:")
        st.write(pd.Series(colocaciones).value_counts().to_frame().reset_index().rename(columns={'index': 'Colocación', 0: 'Frecuencia'}))

        # Análisis de entidades clave usando spaCy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(input_text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        st.subheader("Entidades Clave Detectadas:")
        st.write(pd.DataFrame(entities, columns=["Texto", "Tipo"]))

        # Análisis de sentimiento usando transformers
        sentiment = sentiment_analyzer(input_text)[0]
        st.subheader("Análisis de Sentimiento:")
        st.write(f"**Etiqueta:** {sentiment['label']}")
        st.write(f"**Puntuación:** {sentiment['score']:.2f}")

        # Obtener el resumen
        try:
            summary = summarizer(input_text, max_length=100, min_length=10, do_sample=False)
            st.subheader("Resumen Automático:")
            st.write(summary[0]['summary_text'])
        except Exception as e:
            st.write("Error al generar el resumen:", e)
else:
    st.write("Por favor, ingrese un texto para analizar.")
