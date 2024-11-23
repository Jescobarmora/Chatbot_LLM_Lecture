# utils.py

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
import os

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configurar la clave de API de OpenAI
api_key = os.getenv("OPENAI_API_KEY")

# Verificar que la clave de API se haya cargado correctamente
if api_key is None:
    raise ValueError("La clave de API de OpenAI no está definida en el archivo .env")

client = OpenAI(api_key=api_key)

def text_embedding(text):
    
    embeddings = client.embeddings.create(model="text-embedding-ada-002",
                                          input=text,
                                          encoding_format="float")
    
    return embeddings.data[0].embedding

def get_context_for_question(question, vector_store, n_chunks=5):
    """
    Obtiene los fragmentos más relevantes del vector_store para una pregunta dada.
    """
    query_embedding = text_embedding(question)
    query_vector = np.array(query_embedding)
    
    def cosine_sim(row):
        return np.dot(row, query_vector) / (np.linalg.norm(row) * np.linalg.norm(query_vector))
    
    vector_store = vector_store.copy()
    vector_store['Similarity'] = vector_store['Embedding'].apply(cosine_sim)
    top_chunks = vector_store.sort_values('Similarity', ascending=False).head(n_chunks)
    return list(top_chunks['Chunks'])

def construct_prompt(context_list):
    """
    Construye el prompt personalizado para el modelo de lenguaje.
    """
    custom_prompt = f"""
Eres una Inteligencia Artificial avanzada que trabaja como asistente personal.
Utiliza los RESULTADOS DE BÚSQUEDA SEMÁNTICA para responder las preguntas del usuario.
Solo debes utilizar la información de la BÚSQUEDA SEMÁNTICA si es que hace sentido y tiene relación con la pregunta del usuario.
Si la respuesta no se encuentra dentro del contexto de la búsqueda semántica, no inventes una respuesta y responde amablemente que no tienes información para responder.

RESULTADOS DE BÚSQUEDA SEMÁNTICA:
{context_list}

Escribe una respuesta para el usuario.
"""
    return custom_prompt

def get_chatbot_response(question, prompt, temperature=0.0):
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=temperature,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content.strip()

def compute_similarity(answer1, answer2):
    vectorizer = TfidfVectorizer().fit_transform([answer1, answer2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]