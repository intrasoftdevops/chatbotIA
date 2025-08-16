# Etapa 1: Builder - Instalar dependencias
FROM python:3.11-slim as builder

WORKDIR /app

# Instalar dependencias primero para cacheo
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Argumento para la API key durante el build (en la etapa builder)
ARG GOOGLE_API_KEY
ENV GOOGLE_API_KEY=$GOOGLE_API_KEY

# Etapa 2: Final - Construir la imagen final
FROM python:3.11-slim

WORKDIR /app

# Argumento para la API key durante el build (en la etapa final)
ARG GOOGLE_API_KEY
ENV GOOGLE_API_KEY=$GOOGLE_API_KEY

# Copiar dependencias de la etapa builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copiar datos PRIMERO
COPY data/ ./data/

# Copiar el resto de la aplicación
COPY . .

# Generar el índice ANTES de iniciar (con API key del build)
RUN python prepare_data.py

# Verificar que el índice se generó correctamente
RUN ls -la storage/ && echo "Índice generado exitosamente"

# Exponer el puerto en el que correrá la aplicación
EXPOSE 8080

# Comando para ejecutar la aplicación
CMD ["python", "run_server.py"] 