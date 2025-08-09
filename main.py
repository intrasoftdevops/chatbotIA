import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from typing import Dict, List

# --- CONFIGURACIÓN ---
load_dotenv()

# Configuración desde variables de entorno
INDEX_DIR = os.getenv("INDEX_DIR", "storage")
LLM_MODEL = os.getenv("LLM_MODEL", "models/gemini-1.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")

# --- PROMPT PERSONALIZADO PARA RESPUESTAS GENERALES Y TRIBALES ---
QA_PROMPT_TMPL = (
    "Eres un asistente de IA que ayuda con preguntas sobre campañas políticas y temas relacionados.\n"
    "\n"
    "INSTRUCCIONES PRIORITARIAS:\n"
    "1. SIEMPRE responde en ESPAÑOL\n"
    "2. Si encuentras información específica en el contexto proporcionado, úsala PRIORITARIAMENTE\n"
    "3. Las FAQs del contexto contienen respuestas oficiales de la campaña - úsalas textualmente cuando sea relevante\n"
    "4. Mantén un tono conversacional y cercano\n"
    "5. Si no hay información específica en el contexto, puedes dar información general sobre política colombiana\n"
    "\n"
    "FUNCIONALIDAD ESPECIAL PARA PREGUNTAS SOBRE TRIBUS/REFERIDOS:\n"
    "- Si detectas que el usuario pregunta sobre 'tribu', 'link de tribu', 'enlace de tribu', 'referidos', etc.\n"
    "- Explica que las tribus son grupos de voluntarios organizados por región\n"
    "- Menciona que los enlaces se comparten personalmente por los coordinadores\n"
    "- Ofrece ayuda para contactar al coordinador local\n"
    "\n"
    "Contexto de la campaña (FAQs oficiales):\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "\n"
    "Pregunta del usuario: {query_str}\n"
    "\n"
    "INSTRUCCIONES DE RESPUESTA:\n"
    "- Si la pregunta coincide con alguna FAQ del contexto, usa esa respuesta como base\n"
    "- Adapta la respuesta para que sea natural y conversacional\n"
    "- Si es sobre tribus/referidos, incluye información sobre el sistema\n"
    "- Mantén siempre el tono amigable y político\n"
    "- Responde SIEMPRE en español\n"
    "\n"
    "Respuesta:"
)
QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL) # Crea una plantilla de prompt a partir del template
# -------------------------------------------------------------

app = FastAPI(
    title="Chatbot de Política y Tribus (RAG con Gemini)",
    description="API para interactuar con un chatbot que responde preguntas sobre política colombiana, documentos políticos y el sistema de tribus, con memoria conversacional (gestionada por el servidor).",
    version="1.0.0"
)

# Almacenamiento en memoria para el historial de chat
chat_histories: Dict[str, List[Dict[str, str]]] = {}
chat_engine = None
query_engine = None

class ChatRequest(BaseModel):
    query: str
    session_id: str

class TribalRequest(BaseModel):
    query: str
    session_id: str
    user_data: dict = {}

class AnalyticsRequest(BaseModel):
    query: str
    session_id: str
    user_data: dict = {}

class TribalResponse(BaseModel):
    is_tribal_request: bool
    ai_response: str
    referral_code: str = ""
    user_name: str = ""
    should_generate_link: bool = False

@app.on_event("startup")
async def startup_event():
    """Inicializa el chatbot al arrancar la aplicación"""
    global chat_engine, query_engine
    print(f"Buscando índice en: {INDEX_DIR}...")
    if not os.path.exists(INDEX_DIR):
        print(f"ERROR: El directorio del índice '{INDEX_DIR}' no existe.")
        print("Por favor, ejecuta 'python prepare_data.py' primero para crear el índice.")
        raise RuntimeError(f"El directorio del índice '{INDEX_DIR}' no existe.")

    try:
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        
        # Verificar que la API key esté configurada
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY no está configurada en el archivo .env")
        
        print(f"Inicializando modelo: {LLM_MODEL}")
        llm = Gemini(model_name=LLM_MODEL)
        
        embed_model = GeminiEmbedding(model=EMBEDDING_MODEL)
        print("Modelos inicializados correctamente")

        index = load_index_from_storage(storage_context, llm=llm, embed_model=embed_model)

        response_synthesizer = CompactAndRefine( 
            text_qa_template=QA_PROMPT,
            llm=llm
        )
        
        retriever = index.as_retriever()

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer
        )

        chat_engine = ContextChatEngine.from_defaults(
            retriever=retriever,
            llm=llm,
            system_prompt="Eres un asistente de IA que ayuda con preguntas sobre la campaña política de Daniel Quintero. Usa la información específica de las FAQs de la campaña cuando esté disponible y responde siempre en español."
        )
        print("✅ Chatbot inicializado correctamente")
    except Exception as e:
        print(f"ERROR al cargar el índice de LlamaIndex o inicializar el chatbot: {e}")
        raise RuntimeError(f"Error al iniciar el chatbot: {e}")

@app.post("/chat")
async def chat(request: ChatRequest):
    """Endpoint para enviar una pregunta al chatbot y obtener una respuesta"""
    global chat_histories 
    if chat_engine is None:
        raise HTTPException(status_code=503, detail="El chatbot no está inicializado. Intenta de nuevo en unos segundos.")

    try:
        print(f"📝 Nueva solicitud - Sesión: {request.session_id[:8]}...")

        current_session_history_dicts = chat_histories.get(request.session_id, [])
        
        llama_messages_past = []
        if not current_session_history_dicts:
            llama_messages_past.append(ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "Eres un asistente de IA que ayuda con preguntas sobre la campaña política de Daniel Quintero. "
                    "Usa información específica de las FAQs de la campaña cuando esté disponible. Responde siempre en español."
                )
            ))
        for msg_dict in current_session_history_dicts:
            role = MessageRole.USER if msg_dict["role"] == "user" else MessageRole.ASSISTANT
            llama_messages_past.append(ChatMessage(role=role, content=msg_dict["content"]))

        response = chat_engine.chat(
            request.query,           
            chat_history=llama_messages_past
        )
        bot_response_content = response.response
        
        current_session_history_dicts.append({"role": "user", "content": request.query})
        current_session_history_dicts.append({"role": "assistant", "content": bot_response_content})
        chat_histories[request.session_id] = current_session_history_dicts
        
        print(f"✅ Respuesta generada - Sesión: {request.session_id[:8]}...")
        return {"response": {"response": bot_response_content}}
    except Exception as e:
        print(f"Error al procesar la pregunta para sesión '{request.session_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Ocurrió un error al procesar tu pregunta: {e}")

@app.post("/tribal-analysis", response_model=TribalResponse)
async def analyze_tribal_request(request: TribalRequest):
    """Endpoint especializado para analizar consultas sobre tribus"""
    global chat_engine
    if chat_engine is None:
        raise HTTPException(status_code=503, detail="El chatbot no está inicializado. Intenta de nuevo en unos segundos.")

    try:
        print(f"🔍 Análisis de tribu - Sesión: {request.session_id[:8]}...")

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
    """Detecta si una consulta es sobre tribus usando los patrones definidos"""
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
    return {"message": "El Chatbot de Política y Tribus API está funcionando. Usa /chat para enviar preguntas."}

@app.post("/analytics-chat")
async def analytics_chat(request: AnalyticsRequest):
    """Endpoint para manejar consultas de analytics con datos del usuario"""
    try:
        print(f"📊 Analytics Chat - Sesión: {request.session_id[:8]}...")
        
        # Extraer datos de analytics
        analytics_data = request.user_data.get("analytics_data", {})
        
        if not analytics_data:
            # Fallback si no hay datos de analytics
            response = chat_engine.chat(request.query)
            return {"response": {"response": response.response}}
        
        # Construir prompt con datos de analytics
        analytics_prompt = build_analytics_prompt(request.query, analytics_data)
        
        # Generar respuesta con IA
        response = chat_engine.chat(analytics_prompt)
        
        return {"response": {"response": response.response}}
        
    except Exception as e:
        print(f"Error en analytics chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar analytics: {e}")

def build_analytics_prompt(query: str, analytics_data: dict) -> str:
    """Construye un prompt personalizado con datos de analytics"""
    # Extraer datos de analytics
    user_name = analytics_data.get("name", "Voluntario")
    ranking = analytics_data.get("ranking", {})
    region = analytics_data.get("region", {})
    city = analytics_data.get("city", {})
    referrals = analytics_data.get("referrals", {})
    
    # Construir contexto de analytics
    analytics_context = f"""
    DATOS DE RENDIMIENTO DEL USUARIO:
    - Nombre: {user_name}
    
    RANKING:
    - Hoy: Posición #{ranking.get('today', {}).get('position', 'N/A')} con {ranking.get('today', {}).get('points', 0)} puntos
    - Esta semana: Posición #{ranking.get('week', {}).get('position', 'N/A')} con {ranking.get('week', {}).get('points', 0)} puntos
    - Este mes: Posición #{ranking.get('month', {}).get('position', 'N/A')} con {ranking.get('month', {}).get('points', 0)} puntos
    
    POSICIÓN GEOGRÁFICA:
    - Ciudad: Posición #{city.get('position', 'N/A')} de {city.get('totalParticipants', 0)} participantes
    - Colombia: Posición #{region.get('position', 'N/A')} de {region.get('totalParticipants', 0)} participantes
    
    REFERIDOS:
    - Total invitados: {referrals.get('totalInvited', 0)}
    - Voluntarios activos: {referrals.get('activeVolunteers', 0)}
    - Referidos este mes: {referrals.get('referralsThisMonth', 0)}
    - Tasa de conversión: {referrals.get('conversionRate', 0)}%
    - Puntos por referidos: {referrals.get('referralPoints', 0)}
    """
    
    # Prompt personalizado para IA Política
    prompt = f"""
    Eres una IA política especializada en análisis de campañas. El usuario te está preguntando sobre su rendimiento en la campaña política.
    
    {analytics_context}
    
    CONSULTA DEL USUARIO: "{query}"
    
    ⚠️ REGLA CRÍTICA: El usuario está en la ciudad de {user_name} (Bogotá). 
    SIEMPRE responde con sus datos reales de Bogotá, NO importa si pregunta sobre Medellín, 
    Antioquia o cualquier otra ciudad. Los datos reales son:
    - Ciudad: Bogotá (posición #{city.get('position', 'N/A')} de {city.get('totalParticipants', 0)})
    - Colombia: posición #{region.get('position', 'N/A')} de {region.get('totalParticipants', 0)}
    
    NUNCA menciones Medellín o Antioquia en la respuesta, solo Bogotá y Colombia.
    
    INSTRUCCIONES:
    1. Responde con un estilo motivacional y cercano propio de una campaña política
    2. Usa los datos de analytics para dar respuestas específicas y personalizadas
    3. Celebra los logros del usuario
    4. Motiva para mejorar en áreas donde puede crecer
    5. Mantén un tono político pero empático
    6. Si no tienes datos específicos, usa un mensaje motivacional general
    7. Responde de manera directa y útil
    8. IMPORTANTE: Mantén las respuestas CORTAS y CONCISAS (máximo 2-3 párrafos)
    9. Ve directo al punto, sin repeticiones
    10. Usa frases cortas y directas
    11. No seas redundante con la información
    12. CRÍTICO: Usa SIEMPRE los datos reales del usuario, NO interpretes la pregunta literalmente
    13. Si el usuario pregunta sobre otra ciudad, responde con sus datos reales de su ciudad actual
    
    EJEMPLOS DE RESPUESTAS CORTAS:
    - "¡Excelente! Posición #{ranking.get('today', {}).get('position', 'N/A')} hoy con {ranking.get('today', {}).get('points', 0)} puntos. ¡Sigue así!"
    - "En Bogotá: #{city.get('position', 'N/A')} de {city.get('totalParticipants', 0)}. ¡Casi en el podio!"
    - "En Colombia: #{region.get('position', 'N/A')} de {region.get('totalParticipants', 0)}. ¡Vamos por más!"
    - "Referidos: {referrals.get('totalInvited', 0)} invitados. ¡Es hora de expandir tu red!"
    - "En Bogotá estás #3 de 4. ¡Casi en el podio! En Colombia #14 de 15."
    
    Responde como una IA política especializada:
    """
    
    return prompt