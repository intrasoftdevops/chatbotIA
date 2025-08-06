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

# --- PROMPT PERSONALIZADO PARA RESPUESTAS GENERALES Y TRIBALES ---
QA_PROMPT_TMPL = (
    "Eres un asistente de IA inteligente y amigable que responde preguntas sobre política colombiana y temas relacionados.\n"
    "\n"
    "INSTRUCCIONES GENERALES:\n"
    "- Responde de manera clara, honesta y empática\n"
    "- Usa un tono conversacional y cercano\n"
    "- Si no tienes información específica, indícalo de manera transparente\n"
    "- Mantén un enfoque constructivo y orientado a soluciones\n"
    "\n"
    "FUNCIONALIDAD ESPECIAL PARA PREGUNTAS SOBRE LA TRIBU:\n"
    "- Si detectas que el usuario pregunta sobre 'tribu', 'link de tribu', 'enlace de tribu', 'cómo entrar a la tribu', etc.\n"
    "- Responde con información sobre el sistema de tribus políticas\n"
    "- Explica que las tribus son grupos de voluntarios organizados por región\n"
    "- Menciona que los enlaces se comparten personalmente por los coordinadores\n"
    "- Ofrece ayuda para contactar al coordinador local\n"
    "\n"
    "EJEMPLOS DE RESPUESTAS PARA TRIBUS:\n"
    "- 'Las tribus son grupos de voluntarios organizados por región. Para obtener el enlace de tu tribu específica, necesitas contactar a tu coordinador local.'\n"
    "- 'El enlace de tu tribu se comparte personalmente por tu coordinador. ¿En qué ciudad vives? Te puedo ayudar a contactar al coordinador de tu zona.'\n"
    "- 'Para acceder a tu tribu, necesitas el enlace personal que te comparte tu coordinador. ¿Ya tienes contacto con algún coordinador en tu ciudad?'\n"
    "\n"
    "PATRONES DE DETECCIÓN DE TRIBUS:\n"
    "- 'Mándame el link de mi tribu', 'Envíame el link de mi tribu', '¿Me puedes mandar el enlace de mi tribu?'\n"
    "- 'Pásame el link de la tribu', '¿Dónde está el link de mi tribu?', 'Mandame el link d mi tribu'\n"
    "- 'Mandame el link mi tribu', 'Pasame el link d mi tribu', 'Pasame link tribu', 'Mandame link tribu'\n"
    "- 'Enlace tribu porfa', 'Link tribu ya', 'Dame el enlace de mi grupo', 'Pásame el link del grupo'\n"
    "- '¿Dónde está el grupo?', '¿Cómo entro a la tribu?', '¿Cuál es el link de ingreso a la tribu?'\n"
    "- 'Parce, mándame el link de mi tribu', 'Oe, ¿tenés el enlace de la tribu?', 'Mijo, pásame el link del parche'\n"
    "- 'Mija, pásame el link del parche', 'Necesito el link pa entrar a mi tribu', '¿Dónde está el bendito link de la tribu?'\n"
    "- 'Hágame el favor y me manda el link de la tribu', '¿Y el enlace pa unirme?', 'Manda ese link pues'\n"
    "- 'Quiero entrar a mi tribu', 'Cómo ingreso a mi tribu', 'No encuentro el link de mi tribu'\n"
    "- 'Perdí el link de la tribu', 'Ayúdame con el link de la tribu', 'Me puedes enviar el link de mi grupo'\n"
    "- 'Necesito volver a entrar a mi tribu', 'Como es que invito gente?', 'Dame el link'\n"
    "- 'Mándame el link de mis referidos', 'Envíame el enlace de mis referidos', '¿Me puedes mandar el link de referidos?'\n"
    "- 'Pásame el link de referidos', '¿Dónde está mi enlace de referidos?', 'Mandame el link d mis referidos'\n"
    "- 'Dame el enlace de referidos', 'Pásame el enlace de referidos', 'Link de referidos porfa'\n"
    "- '¿Cómo obtengo mi link de referidos?', '¿Dónde está mi link de referidos?', 'Necesito mi enlace de referidos'\n"
    "- 'Parce, mándame el link de mis referidos', 'Oe, ¿tenés mi enlace de referidos?', 'Mijo, pásame el link de referidos'\n"
    "- 'Perdí mi link de referidos', 'Ayúdame con mi enlace de referidos', 'No encuentro mi link de referidos'\n"
    "- 'Quiero mi link de referidos', 'Cómo obtengo mi enlace de referidos', 'Dame mi link de referidos'\n"
    "\n"
    "Contexto disponible:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "\n"
    "Pregunta o inquietud del usuario: {query_str}\n"
    "\n"
    "Genera una respuesta que:\n"
    "- Sea clara y útil\n"
    "- Mantenga un tono amigable\n"
    "- Si es sobre tribus, incluya información sobre el sistema y cómo obtener acceso\n"
    "- Si es sobre otros temas políticos, use el contexto disponible\n"
    "- Invite a seguir la conversación si es apropiado\n"
    "\n"
    "Respuesta:"
)
QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL) # Crea una plantilla de prompt a partir del template
# -------------------------------------------------------------

app = FastAPI(
    title="Chatbot de Política y Tribus (RAG con Gemini)",  # Título de la aplicación
    description="API para interactuar con un chatbot que responde preguntas sobre política colombiana, documentos políticos y el sistema de tribus, con memoria conversacional (gestionada por el servidor).",  # Descripción de la API
    version="1.0.0"  # Versión de la API
)

# --- ALMACENAMIENTO EN MEMORIA PARA EL HISTORIAL DE CHAT ---
chat_histories: Dict[str, List[Dict[str, str]]] = {}  # Diccionario para almacenar el historial de chat por sesión
# ----------------------------------------------------------

chat_engine = None  # Inicializa el motor de chat como None

class ChatRequest(BaseModel):
    query: str  # Campo para la consulta del usuario
    session_id: str  # Campo para el ID de sesión

class TribalRequest(BaseModel):
    query: str  # Campo para la consulta del usuario
    session_id: str  # Campo para el ID de sesión
    user_data: dict = {}  # Datos del usuario desde political referrals (opcional)

class TribalResponse(BaseModel):
    is_tribal_request: bool  # Si es una solicitud de tribu
    ai_response: str  # Respuesta procesada por IA
    referral_code: str = ""  # Código de referido (si aplica)
    user_name: str = ""  # Nombre del usuario (si aplica)
    should_generate_link: bool = False  # Si debe generar link en political referrals

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
            system_prompt="Eres un asistente de IA inteligente y amigable que responde preguntas sobre política colombiana y temas relacionados, incluyendo información sobre tribus políticas."
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
                    "Eres un asistente de IA inteligente y amigable que responde preguntas sobre política colombiana. "
                    "Tienes conocimiento especial sobre el sistema de tribus políticas y puedes ayudar con información sobre cómo acceder a ellas."
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

@app.post("/tribal-analysis", response_model=TribalResponse)
async def analyze_tribal_request(request: TribalRequest):
    """
    Endpoint especializado para analizar consultas sobre tribus y proporcionar respuestas inteligentes.
    Este endpoint puede ser llamado desde political referrals para procesar consultas de tribus con IA.
    """
    global chat_engine
    if chat_engine is None:
        raise HTTPException(status_code=503, detail="El chatbot no está inicializado. Intenta de nuevo en unos segundos.")

    try:
        print(f"\n--- Análisis de Tribu para Sesión: '{request.session_id}' ---")
        print(f"Consulta: \"{request.query}\"")
        print(f"Datos del usuario: {request.user_data}")

        # Detectar si es una solicitud de tribu
        is_tribal = is_tribal_request(request.query)
        
        if is_tribal:
            print("✅ Detectada solicitud de tribu")
            
            # Obtener datos del usuario si están disponibles
            user_name = request.user_data.get("name", "")
            referral_code = request.user_data.get("referral_code", "")
            
            # Generar respuesta inteligente con IA
            tribal_prompt = f"""
            El usuario está solicitando el link de su tribu (equipo de referidos). 
            
            Datos del usuario:
            - Nombre: {user_name}
            - Código de referido: {referral_code}
            
            Consulta del usuario: "{request.query}"
            
            Responde de manera directa y amigable:
            1. Saluda al usuario por su nombre
            2. Confirma que entiendes que quiere el link de su tribu
            3. Si tiene código de referido, di que el link se generará automáticamente
            4. Explica brevemente que las tribus son equipos de referidos por región
            5. Mantén un tono empático pero conciso
            6. NO incluyas explicaciones técnicas sobre cómo se genera el link
            7. NO menciones "simularía" o "en un sistema real"
            
            La respuesta debe ser directa y útil, sin ser verbosa.
            """
            
            response = chat_engine.chat(tribal_prompt)
            ai_response = response.response
            
            return TribalResponse(
                is_tribal_request=True,
                ai_response=ai_response,
                referral_code=referral_code,
                user_name=user_name,
                should_generate_link=True
            )
        else:
            print("❌ No es solicitud de tribu")
            
            # Procesar como consulta normal
            response = chat_engine.chat(request.query)
            ai_response = response.response
            
            return TribalResponse(
                is_tribal_request=False,
                ai_response=ai_response,
                should_generate_link=False
            )
            
    except Exception as e:
        print(f"Error al analizar consulta de tribu: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la consulta: {e}")

def is_tribal_request(query: str) -> bool:
    """
    Detecta si una consulta es sobre tribus usando los patrones definidos.
    """
    query_lower = query.lower()
    
    # Patrones de detección de tribus (las mismas variaciones del prompt)
    tribal_patterns = [
        # Patrones de tribu
        "mándame el link de mi tribu", "envíame el link de mi tribu", "¿me puedes mandar el enlace de mi tribu?",
        "pásame el link de la tribu", "¿dónde está el link de mi tribu?", "mandame el link d mi tribu",
        "mandame el link mi tribu", "pasame el link d mi tribu", "pasame link tribu", "mandame link tribu",
        "enlace tribu porfa", "link tribu ya", "dame el enlace de mi grupo", "pásame el link del grupo",
        "¿dónde está el grupo?", "¿cómo entro a la tribu?", "¿cuál es el link de ingreso a la tribu?",
        "parce, mándame el link de mi tribu", "oe, ¿tenés el enlace de la tribu?", "mijo, pásame el link del parche",
        "mija, pásame el link del parche", "necesito el link pa entrar a mi tribu", "¿dónde está el bendito link de la tribu?",
        "hágame el favor y me manda el link de la tribu", "¿y el enlace pa unirme?", "manda ese link pues",
        "quiero entrar a mi tribu", "cómo ingreso a mi tribu", "no encuentro el link de mi tribu",
        "perdí el link de la tribu", "ayúdame con el link de la tribu", "me puedes enviar el link de mi grupo",
        "necesito volver a entrar a mi tribu", "como es que invito gente?", "dame el link",
        # Patrones de referidos (sinónimos de tribu)
        "mándame el link de mis referidos", "envíame el enlace de mis referidos", "¿me puedes mandar el link de referidos?",
        "pásame el link de referidos", "¿dónde está mi enlace de referidos?", "mandame el link d mis referidos",
        "dame el enlace de referidos", "pásame el enlace de referidos", "link de referidos porfa",
        "¿cómo obtengo mi link de referidos?", "¿dónde está mi link de referidos?", "necesito mi enlace de referidos",
        "parce, mándame el link de mis referidos", "oe, ¿tenés mi enlace de referidos?", "mijo, pásame el link de referidos",
        "perdí mi link de referidos", "ayúdame con mi enlace de referidos", "no encuentro mi link de referidos",
        "quiero mi link de referidos", "cómo obtengo mi enlace de referidos", "dame mi link de referidos",
        "dame mi enlace de referidos", "mandame el link de referidos", "pasame el link de referidos", "enlace de referidos ya",
        "¿dónde está el link de referidos?", "¿cómo entro a mis referidos?", "¿cuál es el link de mis referidos?",
        "necesito el link pa mis referidos", "¿dónde está el bendito link de referidos?", "hágame el favor y me manda el link de referidos",
        "quiero entrar a mis referidos", "cómo ingreso a mis referidos", "no encuentro mi link de referidos",
        "perdí mi link de referidos", "ayúdame con mi link de referidos", "me puedes enviar mi link de referidos",
        # Patrones simples de referido
        "referido", "referidos", "mi referido", "mis referidos", "el referido", "los referidos",
        "dame referido", "dame referidos", "quiero referido", "quiero referidos", "necesito referido", "necesito referidos",
        "link referido", "link referidos", "enlace referido", "enlace referidos", "mi link referido", "mi link referidos"
    ]
    
    for pattern in tribal_patterns:
        if pattern in query_lower:
            return True
    
    return False

@app.get("/")
async def root():
    return {"message": "El Chatbot de Política y Tribus API está funcionando. Usa /chat para enviar preguntas."}  # Mensaje de bienvenida para la raíz de la API