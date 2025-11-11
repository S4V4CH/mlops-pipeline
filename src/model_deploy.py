"""
Model Deployment API - FastAPI Service
=======================================

API REST para servir el modelo de predicción de precios de viviendas en California.
Incluye endpoints para predicciones individuales y por lotes, health checks,
y documentación automática con Swagger.

Autor: MLOps Pipeline Project
Fecha: Noviembre 2025
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import joblib
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Agregar directorio padre al path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# Configuración de la aplicación
# ============================================================================

app = FastAPI(
    title="California Housing Price Prediction API",
    description="API REST para predicciones de precios de viviendas usando Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Variables globales para modelo y preprocessor
model = None
preprocessor = None
feature_names = None
config = None

# ============================================================================
# Modelos de datos (Pydantic)
# ============================================================================

class HousingFeatures(BaseModel):
    """
    Esquema de entrada para predicción individual.
    Incluye todas las features necesarias para el modelo.
    """
    longitude: float = Field(..., description="Longitud geográfica", ge=-125, le=-114)
    latitude: float = Field(..., description="Latitud geográfica", ge=32, le=42)
    housing_median_age: float = Field(..., description="Edad mediana de las viviendas", ge=1, le=100)
    total_rooms: float = Field(..., description="Total de habitaciones", ge=1)
    total_bedrooms: float = Field(..., description="Total de dormitorios", ge=1)
    population: float = Field(..., description="Población del área", ge=1)
    households: float = Field(..., description="Número de hogares", ge=1)
    median_income: float = Field(..., description="Ingreso mediano (en $10,000)", ge=0)
    ocean_proximity: str = Field(..., description="Proximidad al océano")
    
    @validator('ocean_proximity')
    def validate_ocean_proximity(cls, v):
        """Validar que ocean_proximity esté en las categorías permitidas"""
        valid_categories = ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']
        if v not in valid_categories:
            raise ValueError(f"ocean_proximity debe ser uno de: {valid_categories}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 41.0,
                "total_rooms": 880.0,
                "total_bedrooms": 129.0,
                "population": 322.0,
                "households": 126.0,
                "median_income": 8.3252,
                "ocean_proximity": "NEAR BAY"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Esquema para predicciones por lotes"""
    instances: List[HousingFeatures] = Field(..., description="Lista de instancias para predicción")
    
    class Config:
        schema_extra = {
            "example": {
                "instances": [
                    {
                        "longitude": -122.23,
                        "latitude": 37.88,
                        "housing_median_age": 41.0,
                        "total_rooms": 880.0,
                        "total_bedrooms": 129.0,
                        "population": 322.0,
                        "households": 126.0,
                        "median_income": 8.3252,
                        "ocean_proximity": "NEAR BAY"
                    }
                ]
            }
        }

class PredictionResponse(BaseModel):
    """Esquema de respuesta para predicción individual"""
    prediction: float = Field(..., description="Precio predicho de la vivienda")
    model_name: str = Field(..., description="Nombre del modelo utilizado")
    timestamp: str = Field(..., description="Timestamp de la predicción")
    
class BatchPredictionResponse(BaseModel):
    """Esquema de respuesta para predicciones por lotes"""
    predictions: List[float] = Field(..., description="Lista de predicciones")
    model_name: str = Field(..., description="Nombre del modelo utilizado")
    count: int = Field(..., description="Número de predicciones realizadas")
    timestamp: str = Field(..., description="Timestamp de las predicciones")

class HealthResponse(BaseModel):
    """Esquema de respuesta para health check"""
    status: str = Field(..., description="Estado del servicio")
    model_loaded: bool = Field(..., description="Indica si el modelo está cargado")
    model_name: Optional[str] = Field(None, description="Nombre del modelo cargado")
    timestamp: str = Field(..., description="Timestamp del health check")

# ============================================================================
# Funciones auxiliares
# ============================================================================

def load_model_and_config():
    """
    Carga el modelo entrenado, preprocessor y configuración.
    """
    global model, preprocessor, feature_names, config
    
    try:
        # Cargar configuración
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Directorio de modelos
        model_dir = os.path.join(os.path.dirname(__file__), '..', 
                                 config.get('model_output_dir', 'models/'))
        
        # Buscar el último modelo guardado
        model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_') and f.endswith('.pkl')]
        
        if not model_files:
            raise FileNotFoundError(f"No se encontraron modelos en {model_dir}")
        
        # Ordenar por fecha de modificación y tomar el más reciente
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
        latest_model_file = os.path.join(model_dir, model_files[0])
        
        # Cargar modelo
        model = joblib.load(latest_model_file)
        logger.info(f"Modelo cargado exitosamente: {model_files[0]}")
        
        # Buscar preprocessor
        preprocessor_files = [f for f in os.listdir(model_dir) 
                             if f.startswith('preprocessor_') and f.endswith('.pkl')]
        
        if preprocessor_files:
            preprocessor_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
            preprocessor_path = os.path.join(model_dir, preprocessor_files[0])
            preprocessor_data = joblib.load(preprocessor_path)
            
            if isinstance(preprocessor_data, dict):
                preprocessor = preprocessor_data['preprocessor']
                feature_names = preprocessor_data['feature_names']
            else:
                preprocessor = preprocessor_data
                feature_names = None
            
            logger.info(f"Preprocessor cargado exitosamente: {preprocessor_files[0]}")
        else:
            logger.warning("No se encontró preprocessor. Se usará el modelo directamente.")
            preprocessor = None
            feature_names = None
        
        return True
        
    except Exception as e:
        logger.error(f"Error cargando modelo: {str(e)}")
        return False

def preprocess_input(input_data: pd.DataFrame) -> np.ndarray:
    """
    Preprocesa los datos de entrada usando el preprocessor cargado.
    
    Args:
        input_data: DataFrame con los datos de entrada
        
    Returns:
        Array numpy con los datos preprocesados
    """
    try:
        # Crear features derivados (misma lógica que en ft_engineering.py)
        df = input_data.copy()
        
        # Features derivados
        df['rooms_per_household'] = df['total_rooms'] / df['households']
        df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
        df['population_per_household'] = df['population'] / df['households']
        
        if preprocessor is not None:
            # Usar preprocessor si está disponible
            X_processed = preprocessor.transform(df)
        else:
            # Si no hay preprocessor, convertir directamente a array
            # (esto requeriría que el modelo acepte datos sin procesar)
            X_processed = df.values
        
        return X_processed
        
    except Exception as e:
        logger.error(f"Error en preprocesamiento: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en preprocesamiento de datos: {str(e)}"
        )

# ============================================================================
# Eventos de inicio y cierre
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Evento ejecutado al iniciar la aplicación.
    Carga el modelo y configuración.
    """
    logger.info("Iniciando servicio de predicción...")
    success = load_model_and_config()
    
    if success:
        logger.info("✓ Servicio iniciado correctamente")
        logger.info(f"✓ Modelo: {type(model).__name__}")
        logger.info(f"✓ Preprocessor: {'Cargado' if preprocessor else 'No disponible'}")
    else:
        logger.error("✗ Error al iniciar el servicio")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Evento ejecutado al cerrar la aplicación.
    """
    logger.info("Cerrando servicio de predicción...")

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """
    Endpoint raíz con información básica de la API.
    """
    return {
        "service": "California Housing Price Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint para verificar el estado del servicio.
    
    Returns:
        Estado del servicio y disponibilidad del modelo
    """
    is_healthy = model is not None
    
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=is_healthy,
        model_name=type(model).__name__ if is_healthy else None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: HousingFeatures):
    """
    Endpoint para predicción individual.
    
    Args:
        features: Características de la vivienda
        
    Returns:
        Predicción del precio de la vivienda
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible. El servicio no está listo."
        )
    
    try:
        # Convertir entrada a DataFrame
        input_dict = features.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Preprocesar datos
        X_processed = preprocess_input(input_df)
        
        # Realizar predicción
        prediction = float(model.predict(X_processed)[0])
        
        logger.info(f"Predicción realizada: ${prediction:,.2f}")
        
        return PredictionResponse(
            prediction=prediction,
            model_name=type(model).__name__,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al realizar predicción: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Endpoint para predicciones por lotes.
    
    Args:
        request: Lista de instancias para predicción
        
    Returns:
        Lista de predicciones
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible. El servicio no está listo."
        )
    
    try:
        # Convertir lista de instancias a DataFrame
        instances_dict = [instance.dict() for instance in request.instances]
        input_df = pd.DataFrame(instances_dict)
        
        # Preprocesar datos
        X_processed = preprocess_input(input_df)
        
        # Realizar predicciones
        predictions = model.predict(X_processed)
        predictions_list = [float(p) for p in predictions]
        
        logger.info(f"Predicciones por lotes realizadas: {len(predictions_list)} instancias")
        
        return BatchPredictionResponse(
            predictions=predictions_list,
            model_name=type(model).__name__,
            count=len(predictions_list),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error en predicción por lotes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al realizar predicciones: {str(e)}"
        )

@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Obtener información sobre el modelo cargado.
    
    Returns:
        Información detallada del modelo y configuración
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible"
        )
    
    info = {
        "model_type": type(model).__name__,
        "model_loaded": True,
        "preprocessor_available": preprocessor is not None,
        "feature_count": len(feature_names) if feature_names else "unknown",
        "config": config
    }
    
    # Agregar parámetros del modelo si están disponibles
    if hasattr(model, 'get_params'):
        info['model_parameters'] = model.get_params()
    
    return info

# ============================================================================
# Manejo de errores
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Manejador global de excepciones.
    """
    logger.error(f"Error no manejado: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Error interno del servidor",
            "timestamp": datetime.now().isoformat()
        }
    )

# ============================================================================
# Ejecución principal
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Configuración del servidor
    uvicorn.run(
        "model_deploy:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
