# Ambiente de Desarrollo - ChatbotIA

Este documento describe cómo configurar y trabajar en el ambiente de desarrollo para el proyecto ChatbotIA.

## 🏗️ Configuración Inicial

### Prerrequisitos
- Python 3.11+
- Git
- Google API Key para Gemini
- Cuenta de Google Cloud (opcional para deploy)

### Setup Rápido
```bash
# 1. Clonar el repositorio y cambiar a rama dev
git clone <repository-url>
cd chatbotIA
git checkout dev

# 2. Ejecutar script de configuración automática
./scripts/setup-dev.sh

# 3. Editar variables de entorno
cp .env.example .env
# Edita .env con tus configuraciones específicas

# 4. Preparar datos (si es necesario)
python prepare_data.py

# 5. Iniciar servidor de desarrollo
python run_server.py
```

## 🔧 Configuración Manual

### 1. Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # En Linux/Mac
# o
venv\Scripts\activate     # En Windows
```

### 2. Dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Variables de Entorno
Crea un archivo `.env` basado en `.env.example`:
```env
GOOGLE_API_KEY=tu_google_api_key_aqui
LLM_MODEL=models/gemini-1.5-flash
EMBEDDING_MODEL=models/embedding-001
INDEX_DIR=storage
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=development
LOG_LEVEL=DEBUG
```

## 🧪 Pruebas

### Ejecutar Pruebas
```bash
# Todas las pruebas
pytest tests/

# Con cobertura
pytest tests/ --cov=. --cov-report=html

# Pruebas específicas
pytest tests/test_main.py::TestTribalDetection -v
```

### Estructura de Pruebas
```
tests/
├── __init__.py
├── test_main.py          # Pruebas principales de la API
├── test_tribal.py        # Pruebas específicas de detección tribal
└── test_analytics.py     # Pruebas de analytics
```

## 🚀 Desarrollo

### Estructura del Proyecto
```
chatbotIA/
├── main.py              # Aplicación principal FastAPI
├── prepare_data.py      # Script de preparación de datos
├── run_server.py        # Script para ejecutar servidor
├── requirements.txt     # Dependencias Python
├── storage/             # Índices de LlamaIndex
├── data/               # Datos fuente
├── tests/              # Pruebas unitarias
├── scripts/            # Scripts de utilidad
├── logs/               # Archivos de log
└── .env                # Variables de entorno (no versionado)
```

### Comandos Útiles

#### Desarrollo
```bash
# Iniciar servidor con hot reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Verificar sintaxis
flake8 .

# Formatear código
black .

# Ejecutar pruebas con watch
pytest-watch
```

#### Datos y Storage
```bash
# Recrear índice
rm -rf storage/ && python prepare_data.py

# Verificar índice
python -c "from llama_index.core import StorageContext, load_index_from_storage; print('Index OK')"
```

#### Docker
```bash
# Construir imagen
docker build -t chatbotia-dev .

# Ejecutar contenedor
docker run -p 8000:8000 --env-file .env chatbotia-dev
```

## 🔀 Flujo de Trabajo Git

### Ramas
- `main`: Producción
- `dev`: Desarrollo principal
- `feature/*`: Nuevas características
- `bugfix/*`: Correcciones de errores
- `hotfix/*`: Correcciones urgentes

### Proceso de Desarrollo
```bash
# 1. Crear nueva rama desde dev
git checkout dev
git pull origin dev
git checkout -b feature/nueva-funcionalidad

# 2. Desarrollar y hacer commits
git add .
git commit -m "feat: agregar nueva funcionalidad"

# 3. Push y crear PR a dev
git push origin feature/nueva-funcionalidad
# Crear Pull Request hacia dev en GitHub
```

## 📊 CI/CD

### GitHub Actions
El pipeline de CI/CD se ejecuta automáticamente en:
- Push a rama `dev`
- Pull Requests hacia `dev`

Incluye:
- ✅ Linting con flake8
- 🧪 Ejecución de pruebas
- 📊 Reporte de cobertura
- 🐳 Build de imagen Docker
- 🚀 Deploy a Cloud Run (ambiente dev)

### Variables de Entorno en CI/CD
Configurar en GitHub Secrets:
- `GOOGLE_API_KEY_DEV`
- `GCP_SA_KEY_DEV`
- `GCP_PROJECT_ID_DEV`

## 🐛 Debugging

### Logs
```bash
# Ver logs en tiempo real
tail -f logs/app.log

# Logs con nivel DEBUG
export LOG_LEVEL=DEBUG
python run_server.py
```

### Endpoints de Debug
- `GET /`: Health check
- `POST /chat`: Chat normal
- `POST /tribal-analysis`: Análisis tribal
- `POST /analytics-chat`: Chat con analytics

### Pruebas Manuales
```bash
# Test básico
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "Hola", "session_id": "test_session"}'

# Test tribal
curl -X POST "http://localhost:8000/tribal-analysis" \
  -H "Content-Type: application/json" \
  -d '{"query": "mándame el link de mi tribu", "session_id": "test", "user_data": {"name": "Test User"}}'
```

## 📚 Recursos

### Documentación
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [LlamaIndex Docs](https://docs.llamaindex.ai/)
- [Google Gemini API](https://ai.google.dev/docs)

### Herramientas Recomendadas
- **IDE**: VS Code con extensiones Python
- **API Testing**: Postman o Insomnia
- **DB Browser**: Para revisar storage si se usa SQLite

## 🆘 Troubleshooting

### Problemas Comunes

#### Error: "INDEX_DIR no existe"
```bash
python prepare_data.py
```

#### Error: "GOOGLE_API_KEY no configurada"
```bash
# Verificar .env
cat .env | grep GOOGLE_API_KEY
```

#### Puerto 8000 en uso
```bash
# Cambiar puerto en .env o matar proceso
lsof -ti:8000 | xargs kill -9
```

#### Dependencias desactualizadas
```bash
pip install --upgrade -r requirements.txt
```

## 📞 Soporte

Para soporte o preguntas:
1. Revisar este README
2. Buscar en Issues de GitHub
3. Crear nuevo Issue con template correspondiente
4. Contactar al equipo de desarrollo