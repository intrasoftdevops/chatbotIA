# Chatbot (RAG con Gemini)

Este es un chatbot conversacional que responde preguntas basadas en un conjunto de documentos proporcionados, **Retrieval Augmented Generation (RAG)** para buscar información relevante en los documentos y generar respuestas precisas y contextualmente informadas.

El proyecto está construido con:

  * **FastAPI**: Para crear la API REST que aloja el chatbot.
  * **LlamaIndex**: Para la gestión de datos, la creación del índice de conocimiento y la implementación de la lógica RAG.
  * **Gemini (Google)**: El modelo de lenguaje grande (LLM) que procesa las preguntas y genera las respuestas.

-----

### Requisitos Previos

Antes de comenzar, asegúrate de tener instalado lo siguiente:

  * **Python 3.11** o superior.
  * **Git** (para la gestión de código).
  * **Una clave API de Gemini** obtenida de [Google AI Studio](https://aistudio.google.com/app/apikey).

-----

### Configuración del Entorno

Sigue estos pasos para configurar tu proyecto y sus dependencias.

1.  **Clonar el repositorio:**

    ```bash
    git clone https://github.com/sindresorhus/del
    cd chatbotIA
    ```

2.  **Crear y activar el entorno virtual:**

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Configurar tu clave API:**

      * Crea un archivo llamado `.env` en la raíz del proyecto.
      * Añade tu clave API de Gemini en la siguiente línea:
        ```env
        GOOGLE_API_KEY="TU_CLAVE_API_DE_GEMINI"
        ```


4.  **Instalar las dependencias:**

    ```bash
    pip install llama-index-llms-gemini llama-index-embeddings-gemini llama-index pydantic "uvicorn[standard]" google-generativeai fastapi
    ```

-----

### Uso del Proyecto

Una vez que el entorno y las dependencias estén configurados, puedes ejecutar el chatbot en dos sencillos pasos.

#### Paso 1: Preparar la Base de Conocimiento

Este comando procesa todos los documentos de la carpeta `data/` y crea un índice que el chatbot utilizará. Solo necesitas ejecutarlo una vez o cada vez que añadas o modifiques documentos.

```bash
python prepare_data.py
```

  * **Verificación:** Si todo va bien, verás un mensaje que dice "Índice guardado exitosamente." Se creará una carpeta `storage/`.

#### Paso 2: Ejecutar el Servidor del Chatbot

Este comando inicia el servidor web que aloja la API del chatbot. El servidor debe estar corriendo para poder interactuar con el chatbot.

```bash
python -m uvicorn main:app --reload
```

  * **Verificación:** Si todo funciona, verás un mensaje que dice "Uvicorn running on [http://127.0.0.1:8000](http://127.0.0.1:8000)" y "Índice cargado y motor de consulta del chatbot listo para usar.".

-----

### Interactuar con el Chatbot

Mientras el servidor está activo, puedes acceder a la interfaz de documentación interactiva de FastAPI (Swagger UI) para enviar preguntas.

1.  Abre tu navegador web.
2.  Ve a la siguiente dirección:
    ```
    http://127.0.0.1:8000/docs
    ```
3.  En la interfaz, expande el endpoint `POST /chat`, haz clic en "Try it out" e introduce tu pregunta en el campo `query`.
4.  Haz clic en "Execute" para obtener una respuesta de tu chatbot.


