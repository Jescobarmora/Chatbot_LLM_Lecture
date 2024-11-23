# Chatbot de Respuestas sobre una Lectura de LLM

## Descripción del Proyecto

Este proyecto consiste en un **Chatbot personalizado** que responde a preguntas basadas en el contenido del documento `llm_doc.pdf`. Utiliza modelos de lenguaje de gran tamaño (LLM) de OpenAI para generar respuestas coherentes y relevantes. Además, compara las respuestas generadas por el chatbot con respuestas predefinidas para evaluar su precisión.

## Características

- **Generación de Preguntas y Respuestas:** Crea 8 preguntas y respuestas basadas en el contenido de un documento PDF.
- **Chatbot Personalizado:** Implementa un chatbot utilizando un LLM de OpenAI que puede responder a las preguntas generadas.
- **Evaluación de Respuestas:** Compara las respuestas del chatbot con las respuestas propuestas mediante métricas de similitud.
- **Visualización de Accuracy:** Genera una gráfica que muestra la distribución acumulada del accuracy a medida que se evalúan las respuestas.
- **Interfaz Web con Streamlit:** Proporciona una interfaz de usuario amigable para interactuar con el chatbot y evaluar sus respuestas.

## Configurar Variables de Entorno

Crea un archivo .env en el directorio raíz del proyecto y añade tu clave de API de OpenAI:
OPENAI_API_KEY=tu_clave_de_api_aqui

## Ejecutar la Aplicación de Streamlit

Inicia la aplicación de Streamlit con el siguiente comando:
`streamlit run src/app.py`
