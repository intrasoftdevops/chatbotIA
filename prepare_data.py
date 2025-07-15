import os
from dotenv import load_dotenv # Añade esta importación
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.gemini import Gemini  
from llama_index.embeddings.gemini import GeminiEmbedding 
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

DATA_DIR = "data"  
INDEX_DIR = "storage" 
LLM_MODEL = "gemini-1.5-flash"

def prepare_documents():
    """
    Carga documentos de un directorio y crea un índice que se guarda localmente.
    """
    if not os.path.exists(DATA_DIR):
        print(f"Error: El directorio de datos '{DATA_DIR}' no existe.")
        print(f"Por favor, crea la carpeta '{DATA_DIR}' y coloca tus documentos (libros, políticas, etc.) allí.")
        return

    print(f"Cargando documentos de: {DATA_DIR}...")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    print(f"Se cargaron {len(documents)} documentos.")

    print("Dividiendo documentos en 'chunks' y creando nodos...")
    parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    nodes = parser.get_nodes_from_documents(documents)
    print(f"Se generaron {len(nodes)} nodos (trozos de información).")

    print("Inicializando LLM de Gemini para generación de texto y Embeddings...")
    llm = Gemini(model=LLM_MODEL) 
    
    embed_model = GeminiEmbedding(model="models/embedding-001") 

    index = VectorStoreIndex(nodes, llm=llm, embed_model=embed_model)

    print(f"Guardando índice en: {INDEX_DIR}...")
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print("Índice guardado exitosamente.")

if __name__ == "__main__":
    prepare_documents()