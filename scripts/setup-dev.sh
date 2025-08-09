#!/bin/bash
# Script de configuración para ambiente de desarrollo - chatbotIA

set -e

echo "🚀 Configurando ambiente de desarrollo para chatbotIA..."

# Verificar si Python 3.11+ está instalado
python_version=$(python3 --version | cut -d' ' -f2)
echo "📋 Versión de Python detectada: $python_version"

# Crear directorio de scripts si no existe
mkdir -p scripts

# Verificar si virtual environment existe
if [ ! -d "venv" ]; then
    echo "📦 Creando virtual environment..."
    python3 -m venv venv
fi

# Activar virtual environment
echo "🔄 Activando virtual environment..."
source venv/bin/activate

# Actualizar pip
echo "📥 Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
echo "📚 Instalando dependencias..."
pip install -r requirements.txt

# Verificar si existe archivo .env
if [ ! -f ".env" ]; then
    echo "⚙️ Configurando variables de entorno..."
    if [ -f "env.dev.example" ]; then
        cp env.dev.example .env
        echo "✅ Archivo .env creado desde env.dev.example"
        echo "⚠️  Por favor, edita el archivo .env con tus configuraciones específicas"
    elif [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Archivo .env creado desde .env.example"
        echo "⚠️  Por favor, edita el archivo .env con tus configuraciones específicas"
    else
        echo "❌ No se encontró archivo de ejemplo para .env"
    fi
fi

# Verificar si existe directorio storage
if [ ! -d "storage" ]; then
    echo "📁 Creando directorio storage..."
    mkdir -p storage
fi

# Crear directorio de logs si no existe
mkdir -p logs

# Verificar configuración
echo "🔍 Verificando configuración..."
if [ -f ".env" ]; then
    echo "✅ Archivo .env encontrado"
else
    echo "❌ Archivo .env no encontrado"
fi

if [ -d "storage" ]; then
    echo "✅ Directorio storage encontrado"
else
    echo "❌ Directorio storage no encontrado"
fi

# Ejecutar pruebas si están configuradas
echo "🧪 Ejecutando pruebas..."
if [ -d "tests" ]; then
    pytest tests/ -v || echo "⚠️  Algunas pruebas fallaron o no hay pruebas configuradas"
else
    echo "⚠️  Directorio de pruebas no encontrado"
fi

echo ""
echo "✅ ¡Configuración de desarrollo completada!"
echo ""
echo "📋 Próximos pasos:"
echo "   1. Edita el archivo .env con tus configuraciones específicas"
echo "   2. Ejecuta 'python prepare_data.py' para preparar los datos"
echo "   3. Ejecuta 'python run_server.py' para iniciar el servidor"
echo ""
echo "🔧 Comandos útiles:"
echo "   - Activar venv: source venv/bin/activate"
echo "   - Ejecutar pruebas: pytest tests/"
echo "   - Iniciar servidor: python run_server.py"
echo "   - Ver logs: tail -f logs/app.log"
echo ""