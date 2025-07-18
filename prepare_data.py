import os  # Importa el módulo os para interactuar con el sistema operativo
from dotenv import load_dotenv  # Carga variables de entorno desde un archivo .env
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader  # Importa clases para manejar índices y leer directorios
from llama_index.llms.gemini import Gemini  # Importa el modelo de lenguaje Gemini
from llama_index.embeddings.gemini import GeminiEmbedding  # Importa el modelo de embeddings Gemini
from llama_index.core.node_parser import SentenceSplitter  # Importa el divisor de oraciones para crear chunks

load_dotenv()  # Carga las variables de entorno desde el archivo .env

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")  # Establece la clave de API de Google

DATA_DIR = "data"  # Directorio donde se encuentran los documentos
INDEX_DIR = "storage"  # Directorio donde se guardará el índice
LLM_MODEL = "gemini-1.5-flash"  # Modelo de lenguaje a utilizar

def prepare_documents():
    """
    Carga documentos de un directorio y crea un índice que se guarda localmente.
    """
    if not os.path.exists(DATA_DIR):  # Verifica si el directorio de datos existe
        print(f"Error: El directorio de datos '{DATA_DIR}' no existe.")
        print(f"Por favor, crea la carpeta '{DATA_DIR}' y coloca tus documentos (libros, políticas, etc.) allí.")
        return

    print(f"Cargando documentos de: {DATA_DIR}...")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()  # Carga los documentos desde el directorio
    print(f"Se cargaron {len(documents)} documentos.")

    print("Dividiendo documentos en 'chunks' y creando nodos...")
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)  # Divide los documentos en chunks
    nodes = parser.get_nodes_from_documents(documents)  # Crea nodos a partir de los chunks
    print(f"Se generaron {len(nodes)} nodos (trozos de información).")

    print("Inicializando LLM de Gemini para generación de texto y Embeddings...")
    llm = Gemini(model=LLM_MODEL)  # Inicializa el modelo de lenguaje Gemini
    
    embed_model = GeminiEmbedding(model="models/embedding-001")  # Inicializa el modelo de embeddings Gemini

    index = VectorStoreIndex(nodes, llm=llm, embed_model=embed_model)  # Crea un índice a partir de los nodos

    print(f"Guardando índice en: {INDEX_DIR}...")
    index.storage_context.persist(persist_dir=INDEX_DIR)  # Guarda el índice en el directorio especificado
    print("Índice guardado exitosamente.")

if __name__ == "__main__":
    prepare_documents()  # Ejecuta la función para preparar los documentos