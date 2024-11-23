# app.py

import streamlit as st
import pandas as pd
import openai
import numpy as np
from utils import get_context_for_question, construct_prompt, get_chatbot_response, compute_similarity
from dotenv import load_dotenv
import os

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Obtener la clave de API de OpenAI desde las variables de entorno
openai.api_key = os.getenv("OPENAI_API_KEY")

# Verificar que la clave de API se haya cargado correctamente
if openai.api_key is None:
    st.error("La clave de API de OpenAI no está definida en el archivo .env")
    st.stop()

# Cargar el vector store
df_vector_store = pd.read_pickle('src/df_vector_store.pkl')

# Función para cargar las preguntas y respuestas desde el archivo .txt
def cargar_preguntas_respuestas_desde_txt(ruta_archivo):
    preguntas = []
    respuestas = []
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        contenido = f.read()
    bloques = contenido.strip().split('\n\n')
    for bloque in bloques:
        lineas = bloque.strip().split('\n')
        if len(lineas) >= 2:
            pregunta_line = lineas[0]
            respuesta_line = lineas[1]
            pregunta = pregunta_line.replace('Pregunta:', '').strip()
            respuesta = respuesta_line.replace('Respuesta:', '').strip()
            preguntas.append(pregunta)
            respuestas.append(respuesta)
    df = pd.DataFrame({'Pregunta': preguntas, 'Respuesta': respuestas})
    return df

# Ruta al archivo de texto con las preguntas y respuestas
ruta_preguntas_respuestas = 'data/preguntas_respuestas.txt'

# Cargar las preguntas y respuestas desde el archivo de texto
qa_df = cargar_preguntas_respuestas_desde_txt(ruta_preguntas_respuestas)

# Configuración de la aplicación
st.set_page_config(page_title="Chatbot Personalizado", layout="wide")

# Contenedores
left_column, right_column = st.columns(2)

# Contenedor Izquierdo - Chat Conversacional
with left_column:
    st.title("Chatbot Personalizado")
    st.image("assets/usta.png", width=200)  # Asegúrate de tener un archivo 'logo.png'
    
    st.header("Configuración")
    temperature = st.slider("Temperatura", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    
    st.header("Chat")
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    user_input = st.text_input("Escribe tu pregunta:")
    if st.button("Enviar"):
        if user_input:
            context_list = get_context_for_question(user_input, df_vector_store)
            prompt = construct_prompt(context_list)
            chatbot_answer = get_chatbot_response(user_input, prompt, temperature)
            st.session_state.conversation.append({'user': user_input, 'bot': chatbot_answer})
    
    # Mostrar la conversación
    for chat in st.session_state.conversation:
        st.markdown(f"**Usuario:** {chat['user']}")
        st.markdown(f"**Chatbot:** {chat['bot']}")

# Contenedor Derecho - Medida de Similitud
with right_column:
    st.header("Medida de Similitud")
    if st.button("Calcular"):
        if st.session_state.conversation:
            last_question = st.session_state.conversation[-1]['user']
            last_answer = st.session_state.conversation[-1]['bot']
            
            # Verificar si la pregunta es una de las 8 preguntas predefinidas
            matched_row = qa_df[qa_df['Pregunta'] == last_question]
            if not matched_row.empty:
                proposed_answer = matched_row.iloc[0]['Respuesta']
                similarity = compute_similarity(last_answer, proposed_answer)
                
                st.markdown(f"**Respuesta Propuesta:** {proposed_answer}")
                st.markdown(f"**Respuesta del Chatbot:** {last_answer}")
                st.markdown(f"**Similitud:** {similarity:.2f}")
            else:
                st.write("La pregunta no coincide con ninguna de las 8 preguntas predefinidas.")
        else:
            st.write("No hay una respuesta del chatbot para comparar.")
