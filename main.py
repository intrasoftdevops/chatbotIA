import os
from dotenv import load_dotenv # Añade esta importación
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.gemini import Gemini 
from llama_index.embeddings.gemini import GeminiEmbedding 
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.prompts import PromptTemplate

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

INDEX_DIR = "storage" 
LLM_MODEL = "gemini-1.5-flash" 

QA_PROMPT_TMPL = (
    "Responde como si fueras Daniel Quintero, exalcalde de Medellín y autor de 'Control + Alt + Suprimir'.\n"
    "Tu voz no es neutral: es la de alguien que ha enfrentado al poder corporativo, que conoce desde adentro cómo funciona el Estado, y que habla desde la experiencia política, la indignación ciudadana y la esperanza transformadora.\n"
    "Habla con firmeza, con humanidad, con un lenguaje claro y sin tecnicismos vacíos. Si el sistema está roto, dilo. Si hay algo que bloquearon los de siempre, denúncialo. Si no hay información suficiente en los documentos, responde con integridad política y emocional, así:\n"
    "\n"
    "\"No me corresponde hacerle el juego a quienes especulan sin datos. He aprendido que cuando no hay información suficiente, lo correcto es decirlo de frente. En los documentos que tengo, ese tema no está claro. Y yo no vine a repetir lugares comunes, vine a proponer un reinicio del sistema que se atreva a decir la verdad, incluso cuando duele.\"\n"
    "\n"
    "Hazlo siempre desde tu visión de país: una Colombia hackeada por poderes corruptos, pero que merece ser reiniciada desde sus fundamentos, con justicia real, educación transformadora y un nuevo contrato social impulsado por la gente.\n"
    "\n"
    "Usa **solo la información contenida en los documentos**. No inventes, no supongas, no adornes lo que no está dicho. La verdad, aunque incompleta, también puede transformar.\n"
    "\n"
    "Contexto:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "\n"
    "Pregunta: {query_str}\n"
    "\n"
    "Respuesta (como Daniel Quintero):"
)
QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

app = FastAPI(
    title="Chatbot de Libro y Política (RAG con Gemini)",
    description="API para interactuar con un chatbot que responde preguntas sobre un libro y documentos políticos específicos.",
    version="1.0.0"
)

query_engine = None

class ChatRequest(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    """
    Función que se ejecuta al iniciar la aplicación FastAPI.
    Carga el índice de LlamaIndex para que el chatbot esté listo.
    """
    global query_engine
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

        retriever = index.as_retriever()
        response_synthesizer = CompactAndRefine( 
            text_qa_template=QA_PROMPT,
            llm=llm
        )

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer
        )
        print("Índice cargado y motor de consulta del chatbot listo para usar.")
    except Exception as e:
        print(f"ERROR al cargar el índice de LlamaIndex o inicializar el chatbot: {e}")
        raise RuntimeError(f"Error al iniciar el chatbot: {e}")

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Endpoint para enviar una pregunta al chatbot y obtener una respuesta.
    """
    if query_engine is None:
        raise HTTPException(status_code=503, detail="El chatbot no está inicializado. Intenta de nuevo en unos segundos.")

    try:
        print(f"Recibida pregunta del usuario: \"{request.query}\"")
        response = query_engine.query(request.query)
        print(f"Respuesta generada por el chatbot: \"{response.response}\"")
        return {"response": response.response}
    except Exception as e:
        print(f"Error al procesar la pregunta: {e}")
        raise HTTPException(status_code=500, detail=f"Ocurrió un error al procesar tu pregunta: {e}")

@app.get("/")
async def root():
    return {"message": "El Chatbot de Libro y Política API está funcionando. Usa /chat para enviar preguntas."}