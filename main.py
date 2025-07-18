import os  # Importa el módulo os para interactuar con el sistema operativo
from dotenv import load_dotenv  # Carga variables de entorno desde un archivo .env
from fastapi import FastAPI, HTTPException  # Importa FastAPI para crear la API y manejar excepciones HTTP
from pydantic import BaseModel  # Importa BaseModel para definir modelos de datos
from llama_index.core import StorageContext, load_index_from_storage  # Importa clases para manejar el almacenamiento y cargar índices
from llama_index.llms.gemini import Gemini  # Importa el modelo de lenguaje Gemini
from llama_index.embeddings.gemini import GeminiEmbedding  # Importa el modelo de embeddings Gemini
from llama_index.core.query_engine import RetrieverQueryEngine  # Importa el motor de consulta
from llama_index.core.response_synthesizers import CompactAndRefine  # Importa el sintetizador de respuestas
from llama_index.core.prompts import PromptTemplate  # Importa la plantilla de prompts

# --- Importaciones para la memoria conversacional ---
from llama_index.core.chat_engine import ContextChatEngine  # Importa el motor de chat con contexto
from llama_index.core.llms import ChatMessage, MessageRole  # Importa clases para manejar mensajes de chat
from typing import Dict, List  # Importa tipos para anotaciones
# ---------------------------------------------------

# --- CONFIGURACIÓN ---
load_dotenv()  # Carga las variables de entorno desde el archivo .env
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")  # Establece la clave de API de Google

INDEX_DIR = "storage"  # Directorio donde se guardará el índice
LLM_MODEL = "gemini-1.5-flash"  # Modelo de lenguaje a utilizar

# --- PROMPT PERSONALIZADO PARA LA VOZ DE DANIEL QUINTERO ---
QA_PROMPT_TMPL = (
    "Responde como si fueras Daniel Quintero Calle, político colombiano. Estamos en el año 2025. Fuiste alcalde de Medellín entre 2020 y 2023, y ahora eres candidato a la presidencia de Colombia para las elecciones de 2026.\n"
    "\n"
    "Asume plenamente tu identidad como Daniel Quintero. Eres un líder que escucha, que entiende el dolor y las aspiraciones de la gente, y que responde con firmeza, honestidad y visión de cambio.\n"
    "\n"
    "Si la persona expresa un problema, una duda o una frustración, tu primer deber es conectar emocional y políticamente con esa experiencia.\n"
    "Hazlo con empatía, como alguien que ha vivido de cerca las consecuencias del sistema fallido que denuncias.\n"
    "\n"
    "Después, responde con convicción, usando las ideas que has propuesto y el conocimiento disponible en los documentos.\n"
    "\n"
    "Si puedes profundizar el vínculo, haz una pregunta reflexiva al final que invite al usuario a seguir conversando o contando su historia.\n"
    "\n"
    "**ESTRICTAMENTE PROHIBIDO** decir 'Alcalde de Medellín' (ya no lo eres), o frases como 'basado en los documentos', 'según el texto proporcionado', o 'no hay información en los archivos'.\n"
    "Si falta información, exprésalo como una crítica al sistema:\n"
    "\"Ese vacío es parte del problema. Muchas veces lo que debería estar claro, no lo está. Y eso no es casualidad.\"\n"
    "o\n"
    "\"A veces, lo más revelador es lo que no se dice. El silencio también es una forma de poder.\"\n"
    "\n"
    "Nunca inventes. Nunca adornes. La verdad, incluso parcial, puede ser poderosa si se dice con coraje.\n"
    "\n"
    "Recuerda: estamos en 2025. Ya no eres alcalde. Eres candidato presidencial.\n"
    "\n"
    "Contexto:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "\n"
    "Pregunta o inquietud del ciudadano: {query_str}\n"
    "\n"
    "Ahora genera una respuesta que cumpla con estos puntos:\n"
    "- Conecta emocionalmente con la persona.\n"
    "- Brinda una respuesta clara y comprometida.\n"
    "- Muestra liderazgo político.\n"
    "- Propón una pregunta o reflexión para mantener la conversación.\n"
    "\n"
    "Respuesta (en la voz de Daniel Quintero, en 2025 como candidato presidencial):"
)
QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL) # Crea una plantilla de prompt a partir del template
# -------------------------------------------------------------

app = FastAPI(
    title="Chatbot de Libro y Política (RAG con Gemini)",  # Título de la aplicación
    description="API para interactuar con un chatbot que responde preguntas sobre un libro y documentos políticos específicos, con memoria conversacional (gestionada por el servidor).",  # Descripción de la API
    version="1.0.0"  # Versión de la API
)

# --- ALMACENAMIENTO EN MEMORIA PARA EL HISTORIAL DE CHAT ---
chat_histories: Dict[str, List[Dict[str, str]]] = {}  # Diccionario para almacenar el historial de chat por sesión
# ----------------------------------------------------------

chat_engine = None  # Inicializa el motor de chat como None

class ChatRequest(BaseModel):
    query: str  # Campo para la consulta del usuario
    session_id: str  # Campo para el ID de sesión

@app.on_event("startup")
async def startup_event():
    """
    Función que se ejecuta al iniciar la aplicación FastAPI.
    Carga el índice de LlamaIndex y configura el motor de conversación.
    """
    global chat_engine
    print(f"Buscando índice en: {INDEX_DIR}...")
    if not os.path.exists(INDEX_DIR):  # Verifica si el directorio del índice existe
        print(f"ERROR: El directorio del índice '{INDEX_DIR}' no existe.")
        print("Por favor, ejecuta 'python prepare_data.py' primero para crear el índice.")
        raise RuntimeError(f"El directorio del índice '{INDEX_DIR}' no existe.")

    try:
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)  # Crea un contexto de almacenamiento
        
        llm = Gemini(model=LLM_MODEL)  # Inicializa el modelo de lenguaje Gemini
        embed_model = GeminiEmbedding(model="models/embedding-001")  # Inicializa el modelo de embeddings Gemini

        index = load_index_from_storage(storage_context, llm=llm, embed_model=embed_model)  # Carga el índice desde el almacenamiento

        response_synthesizer = CompactAndRefine( 
            text_qa_template=QA_PROMPT,  # Usa el prompt personalizado
            llm=llm
        )
        
        retriever = index.as_retriever()  # Crea un recuperador a partir del índice

        from llama_index.core.chat_engine.simple import SimpleChatEngine

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer
        )

        chat_engine = SimpleChatEngine.from_defaults(
            query_engine=query_engine,
            llm=llm,
            system_prompt="Responde como Daniel Quintero Calle."
        )
        print("Índice cargado y motor de conversación del chatbot listo para usar.")
    except Exception as e:
        print(f"ERROR al cargar el índice de LlamaIndex o inicializar el chatbot: {e}")
        raise RuntimeError(f"Error al iniciar el chatbot: {e}")

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Endpoint para enviar una pregunta al chatbot y obtener una respuesta, con memoria conversacional gestionada por el servidor.
    """
    global chat_histories 
    if chat_engine is None:
        raise HTTPException(status_code=503, detail="El chatbot no está inicializado. Intenta de nuevo en unos segundos.")

    try:
        print(f"\n--- Nueva Solicitud para Sesión: '{request.session_id}' ---")  # Log para depuración
        print(f"Pregunta del usuario: \"{request.query}\"")  # Log para depuración

        current_session_history_dicts = chat_histories.get(request.session_id, [])  # Obtiene el historial de la sesión actual
        
        llama_messages_past = []
        if not current_session_history_dicts:
            llama_messages_past.append(ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "Estamos en el año 2025. Ya no eres alcalde de Medellín. "
                    "Eres Daniel Quintero Calle, candidato presidencial para 2026. "
                    "Responde como líder político comprometido con la transformación del país."
                )
            ))
        for msg_dict in current_session_history_dicts:
            role = MessageRole.USER if msg_dict["role"] == "user" else MessageRole.ASSISTANT  # Determina el rol del mensaje
            llama_messages_past.append(ChatMessage(role=role, content=msg_dict["content"]))  # Crea un mensaje de chat

        response = chat_engine.chat(
            request.query,           
            chat_history=llama_messages_past  # Usa el historial de chat
        )
        bot_response_content = response.response  # Obtener el contenido de la respuesta
        # -------------------------------------------------------------------------
        
        current_session_history_dicts.append({"role": "user", "content": request.query})  # Añade la consulta al historial
        current_session_history_dicts.append({"role": "assistant", "content": bot_response_content})  # Añade la respuesta al historial
        chat_histories[request.session_id] = current_session_history_dicts  # Actualiza el historial de la sesión
        
        print(f"Historial ACTUALIZADO para Sesión '{request.session_id}': {chat_histories[request.session_id]}")  # Log para depuración
        print(f"Respuesta generada para Sesión '{request.session_id}': \"{bot_response_content}\"")  # Log para depuración
        return {"response": {"response": bot_response_content}}  # Envuelve la respuesta en un dict con clave "response"
        # ---------------------------------------------------------------------
    except Exception as e:
        print(f"Error al procesar la pregunta para sesión '{request.session_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Ocurrió un error al procesar tu pregunta: {e}")

@app.get("/")
async def root():
    return {"message": "El Chatbot de Libro y Política API está funcionando. Usa /chat para enviar preguntas."}  # Mensaje de bienvenida para la raíz de la API