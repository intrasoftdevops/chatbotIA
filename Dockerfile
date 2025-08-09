# Usar una imagen base de python 3.11
FROM python:3.11-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo requirements.txt y el script de preparación de datos
COPY requirements.txt ./
COPY prepare_data.py ./

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de la aplicación
COPY . .

# Nota: Los datos deben estar pre-procesados en el directorio storage/
# Si no existen, el script prepare_data.py se ejecutará al iniciar la aplicación

# Exponer el puerto en el que correrá la aplicación
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 