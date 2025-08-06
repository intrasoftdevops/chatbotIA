# Chatbot de Política y Tribus (RAG con Gemini)

Este es un chatbot conversacional que responde preguntas sobre política colombiana y el sistema de tribus políticas, utilizando **Retrieval Augmented Generation (RAG)** para buscar información relevante en documentos y generar respuestas precisas y contextualmente informadas.

## 🎯 **Funcionalidades Principales**

### **1. Respuestas sobre Política Colombiana**
- Consultas sobre partidos políticos, democracia, sistema electoral
- Información basada en documentos políticos disponibles
- Respuestas contextuales y empáticas

### **2. Sistema de Tribus Políticas**
- **Detección automática** de preguntas sobre tribus
- **Información sobre coordinadores** y acceso a grupos
- **Explicación del sistema** de voluntarios organizados por región
- **Orientación** para contactar coordinadores locales

### **3. Patrones de Detección de Tribus/Referidos**
El chatbot reconoce automáticamente **60+ variaciones** de preguntas sobre tribus y referidos, incluyendo:
- "Mándame el link de mi tribu"
- "Envíame el link de mi tribu"
- "¿Me puedes mandar el enlace de mi tribu?"
- "Pásame el link de la tribu"
- "¿Dónde está el link de mi tribu?"
- "Mandame el link d mi tribu"
- "Enlace tribu porfa"
- "Link tribu ya"
- "Dame el enlace de mi grupo"
- "¿Cómo entro a la tribu?"
- "Parce, mándame el link de mi tribu"
- "Oe, ¿tenés el enlace de la tribu?"
- "Mijo, pásame el link del parche"
- "Necesito el link pa entrar a mi tribu"
- "¿Dónde está el bendito link de la tribu?"
- "Quiero entrar a mi tribu"
- "Perdí el link de la tribu"
- "Ayúdame con el link de la tribu"
- "Como es que invito gente?"
- "Dame el link"
- **"Mándame el link de mis referidos"**
- **"Envíame el enlace de mis referidos"**
- **"¿Me puedes mandar el link de referidos?"**
- **"Pásame el link de referidos"**
- **"¿Dónde está mi enlace de referidos?"**
- **"Parce, mándame el link de mis referidos"**
- **"Link de referidos porfa"**
- **"¿Cómo obtengo mi link de referidos?"**

**Cobertura completa** de expresiones colombianas, jerga local y variaciones informales. **"Tribu" y "referidos" son sinónimos** en el sistema.

### **4. Respuestas Inteligentes**
- **Contextuales**: Adaptadas al tipo de pregunta
- **Informativas**: Explican el sistema de tribus
- **Orientadoras**: Guían hacia coordinadores locales
- **Empáticas**: Mantienen un tono amigable

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

  * **Verificación:** Si todo funciona, verás un mensaje que dice "Uvicorn running on [http://127.0.0.1:8000](http://127.0.0.1:8000)" y "Índice cargado y motor de conversación del chatbot listo para usar.".

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

-----

### Testing de Funcionalidades

Para probar que el chatbot responde correctamente a preguntas sobre tribus:

```bash
python test_tribal_responses.py
```

Este script prueba:
- **Preguntas sobre tribus**: Verifica que el chatbot detecte y responda apropiadamente
- **Preguntas de control**: Confirma que no mencione tribus en preguntas generales
- **Respuestas contextuales**: Valida que las respuestas sean relevantes

-----

### Ejemplos de Uso

#### Preguntas sobre Tribus:
```
Usuario: "¿Me puedes mandar el link de mi tribu?"
Bot: "Las tribus son grupos de voluntarios organizados por región. Para obtener el enlace de tu tribu específica, necesitas contactar a tu coordinador local. ¿En qué ciudad vives? Te puedo ayudar a contactar al coordinador de tu zona."
```

#### Preguntas Generales de Política:
```
Usuario: "¿Qué es la democracia?"
Bot: "La democracia es un sistema político donde el poder reside en el pueblo..."
```

-----

### API Endpoints

- **GET /**: Mensaje de bienvenida
- **POST /chat**: Endpoint principal para enviar preguntas
  - Body: `{"query": "tu pregunta", "session_id": "identificador_sesion"}`
  - Response: `{"response": {"response": "respuesta del bot"}}`


