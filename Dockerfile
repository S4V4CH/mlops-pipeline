# Dockerfile para MLOps Pipeline - California Housing Prediction API
# Basado en Python 3.12 oficial

FROM python:3.12-slim

# Metadata
LABEL maintainer="MLOps Pipeline Project"
LABEL description="API REST para predicción de precios de viviendas en California"
LABEL version="1.0.0"

# Establecer directorio de trabajo
WORKDIR /app

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivo de requisitos
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar archivos del proyecto
COPY config.json .
COPY src/ ./src/
COPY models/ ./models/

# Crear directorios necesarios si no existen
RUN mkdir -p models monitoring_reports

# Exponer puerto de la API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Comando para ejecutar la aplicación
CMD ["uvicorn", "src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
