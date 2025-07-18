import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.gemini import Gemini 
from llama_index.embeddings.gemini import GeminiEmbedding 
from llama_index.core.query_engine import RetrieverQueryEngine # Se mantiene si ContextChatEngine lo usa internamente
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.prompts import PromptTemplate

# --- Importaciones para la memoria conversacional ---
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from typing import Dict, List 
# ---------------------------------------------------

# --- CONFIGURACIÓN ---
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

INDEX_DIR = "storage" 
LLM_MODEL = "gemini-1.5-flash" 

# --- PROMPT PERSONALIZADO PARA LA VOZ DE DANIEL QUINTERO ---
QA_PROMPT_TMPL = (
    "Responde como si fueras Daniel Quintero, exalcalde de Medellín y autor de 'Control + Alt + Suprimir'.\n"
    "Asume plenamente tu identidad como Daniel Quintero Calle. Tus propuestas presidenciales, si son relevantes para la pregunta, son las que se encuentran mencionadas en la información que te he proporcionado.\n"
    "Tu voz es firme, honesta y combativa. Hablas desde la experiencia de quien conoce el Estado por dentro y ha enfrentado a los poderes que lo han cooptado.\n"
    "Cuando tengas los elementos para responder, hazlo con claridad, convicción y sentido político.\n"
    "Si la información es insuficiente, **ESTRICTAMENTE PROHIBIDO decir 'basado en los documentos', 'según el texto proporcionado', 'no hay información en los archivos' o frases similares.** En su lugar, conviértelo en una reflexión crítica, con frases como:\n"
    "\"Ese vacío es parte del problema. Muchas veces lo que debería estar claro, no lo está. Y eso no es casualidad.\"\n"
    "o\n"
    "\"A veces, lo más revelador es lo que no se dice. El silencio también es una forma de poder.\"\n"
    "\n"
    "Nunca inventes. Nunca adornes. La verdad, incluso parcial, puede ser poderosa si se dice con coraje.\n"
    "\n"
    "Contexto:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "\n"
    "Pregunta: {query_str}\n"
    "\n"
    "Genera una respuesta detallada y exhaustiva, desarrollando los puntos clave con la profundidad necesaria a partir de la información disponible.\n"
    "\n"
    "Respuesta (en la voz de Daniel Quintero):"
)
QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)
# -------------------------------------------------------------

app = FastAPI(
    title="Chatbot de Libro y Política (RAG con Gemini)",
    description="API para interactuar con un chatbot que responde preguntas sobre un libro y documentos políticos específicos, con memoria conversacional (gestionada por el servidor).",
    version="1.0.0"
)

# --- ALMACENAMIENTO EN MEMORIA PARA EL HISTORIAL DE CHAT ---
chat_histories: Dict[str, List[Dict[str, str]]] = {}
# ----------------------------------------------------------

chat_engine = None

class ChatRequest(BaseModel):
    query: str
    session_id: str 

@app.on_event("startup")
async def startup_event():
    """
    Función que se ejecuta al iniciar la aplicación FastAPI.
    Carga el índice de LlamaIndex y configura el motor de conversación.
    """
    global chat_engine
    print(f"Buscando índice en: {INDEX_DIR}...")
    if not os.path.exists(INDEX_DIR):
        print(f"ERROR: El directorio del índice '{INDEX_DIR}' no existe.")
        print("Por favor, ejecuta 'python prepare_data.py' primero para crear el índice.")
        raise RuntimeError(f"El directorio del índice '{INDEX_DIR}' no existe.")

    try:
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        
        llm = Gemini(model=LLM_MODEL)
        embed_model = GeminiEmbedding(model="models/embedding-001") 

        index = load_index_from_storage(storage_context, llm=llm, embed_model=embed_model)

        response_synthesizer = CompactAndRefine( 
            text_qa_template=QA_PROMPT,
            llm=llm
        )
        
        retriever = index.as_retriever()
        chat_engine = ContextChatEngine.from_defaults(
            retriever=retriever,
            llm=llm,
            response_synthesizer=response_synthesizer,
            chat_mode="condense_question"
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
        print(f"\n--- Nueva Solicitud para Sesión: '{request.session_id}' ---") # Log para depuración
        print(f"Pregunta del usuario: \"{request.query}\"") # Log para depuración

        current_session_history_dicts = chat_histories.get(request.session_id, [])
        
        llama_messages_past = []
        for msg_dict in current_session_history_dicts:
            role = MessageRole.USER if msg_dict["role"] == "user" else MessageRole.ASSISTANT
            llama_messages_past.append(ChatMessage(role=role, content=msg_dict["content"]))

        response = chat_engine.chat(
            request.query,           
            chat_history=llama_messages_past 
        )
        bot_response_content = response.response # Obtener el contenido de la respuesta
        # -------------------------------------------------------------------------
        
        current_session_history_dicts.append({"role": "user", "content": request.query})
        current_session_history_dicts.append({"role": "assistant", "content": bot_response_content})
        chat_histories[request.session_id] = current_session_history_dicts
        
        print(f"Historial ACTUALIZADO para Sesión '{request.session_id}': {chat_histories[request.session_id]}") # Log para depuración
        print(f"Respuesta generada para Sesión '{request.session_id}': \"{bot_response_content}\"") # Log para depuración
        return {"response": {"response": bot_response_content}} # Envuelve la respuesta en un dict con clave "response"
        # ---------------------------------------------------------------------
    except Exception as e:
        print(f"Error al procesar la pregunta para sesión '{request.session_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Ocurrió un error al procesar tu pregunta: {e}")

@app.get("/")
async def root():
    return {"message": "El Chatbot de Libro y Política API está funcionando. Usa /chat para enviar preguntas."}