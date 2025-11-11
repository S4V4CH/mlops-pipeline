# 🏠 MLOps Pipeline - California Housing Price Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.121-green)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

Proyecto MLOps completo con pipeline de Machine Learning para predicción de precios de viviendas en California, incluyendo entrenamiento, despliegue, monitoreo y análisis de calidad de código.

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Arquitectura](#-arquitectura)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Modelo](#-modelo)
- [API](#-api)
- [Monitoreo](#-monitoreo)
- [Docker](#-docker)
- [Calidad de Código](#-calidad-de-código)
- [Resultados](#-resultados)
- [Contribución](#-contribución)
- [Licencia](#-licencia)

## ✨ Características

- 🤖 **Modelo de ML**: XGBoost Regressor optimizado (R² = 0.84)
- 🚀 **API REST**: FastAPI con documentación automática
- 📊 **Dashboard de Monitoreo**: Detección de drift con 4 tests estadísticos
- 🎨 **Interfaz de Predicción**: UI intuitiva con Streamlit
- 🐳 **Dockerizado**: Contenedores listos para producción
- 📈 **Análisis de Datos**: EDA completo con visualizaciones
- 🔍 **Calidad de Código**: SonarCloud integrado (Quality Gate: Passed)
- 📝 **CI/CD**: GitHub Actions automatizado

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────┐
│                  Frontend Layer                  │
│  ┌──────────────┐        ┌──────────────┐       │
│  │  Streamlit   │        │  Streamlit   │       │
│  │  Monitoring  │        │  Prediction  │       │
│  │   :8501      │        │    :8502     │       │
│  └──────┬───────┘        └──────┬───────┘       │
└─────────┼────────────────────────┼───────────────┘
          │                        │
          └────────────┬───────────┘
                       │
┌──────────────────────┼───────────────────────────┐
│                  API Layer                       │
│              ┌───────▼────────┐                  │
│              │   FastAPI      │                  │
│              │   :8000        │                  │
│              │  /predict      │                  │
│              │  /health       │                  │
│              │  /model/info   │                  │
│              └───────┬────────┘                  │
└──────────────────────┼───────────────────────────┘
                       │
┌──────────────────────┼───────────────────────────┐
│                 Model Layer                      │
│              ┌───────▼────────┐                  │
│              │  XGBoost Model │                  │
│              │  Preprocessor  │                  │
│              │  RMSE: $45,710 │                  │
│              │  R²: 0.8432    │                  │
│              └────────────────┘                  │
└──────────────────────────────────────────────────┘
```

## 🔧 Requisitos

- Python 3.12+
- pip
- virtualenv (recomendado)
- Docker & Docker Compose (opcional)

## 📦 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/S4V4CH/mlops-pipeline.git
cd mlops-pipeline
```

### 2. Crear entorno virtual

```bash
python -m venv mlops-pipeline-venv
source mlops-pipeline-venv/bin/activate  # Linux/Mac
# mlops-pipeline-venv\Scripts\activate   # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar setup (opcional)

```bash
chmod +x setup.sh
./setup.sh
```

## 🚀 Uso

### Entrenar el Modelo

```bash
# Opción 1: Usando Jupyter Notebook
jupyter notebook src/model_training.ipynb

# Opción 2: Usando el script
python src/model_training.py
```

### Iniciar la API

```bash
uvicorn src.model_deploy:app --host 0.0.0.0 --port 8000 --reload
```

Acceder a:
- **API**: http://localhost:8000
- **Documentación Swagger**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Iniciar Dashboard de Monitoreo

```bash
streamlit run src/streamlit_app.py --server.port 8501
```

Acceder a: http://localhost:8501

### Iniciar Interfaz de Predicción

```bash
streamlit run src/prediction_interface.py --server.port 8502
```

Acceder a: http://localhost:8502

## 📁 Estructura del Proyecto

```
mlops-pipeline/
├── src/
│   ├── Cargar_datos.ipynb          # Carga y exploración inicial
│   ├── comprension_eda.ipynb       # Análisis exploratorio
│   ├── ft_engineering.py           # Feature engineering
│   ├── model_training.ipynb        # Entrenamiento del modelo
│   ├── model_evaluation.ipynb      # Evaluación y métricas
│   ├── model_deploy.py             # API FastAPI
│   ├── streamlit_app.py            # Dashboard de monitoreo
│   ├── prediction_interface.py     # Interfaz de predicción
│   └── model_monitoring.ipynb      # Monitoreo de drift
├── models/                         # Modelos entrenados (.pkl)
├── monitoring_reports/             # Reportes de drift
├── .github/
│   └── workflows/
│       └── sonarcloud.yml         # CI/CD
├── Dockerfile                      # Imagen Docker
├── docker-compose.yml             # Orquestación
├── requirements.txt               # Dependencias Python
├── config.json                    # Configuración
└── README.md                      # Este archivo
```

## 🤖 Modelo

### Algoritmo

**XGBoost Regressor** - Gradient Boosting optimizado

### Características (Features)

- `longitude`: Coordenada de longitud
- `latitude`: Coordenada de latitud  
- `housing_median_age`: Edad mediana de las viviendas
- `total_rooms`: Total de habitaciones
- `total_bedrooms`: Total de dormitorios
- `population`: Población del área
- `households`: Número de hogares
- `median_income`: Ingreso mediano (en $10,000)
- `ocean_proximity`: Proximidad al océano (categórica)

### Feature Engineering

- Rooms per household
- Bedrooms to rooms ratio
- Population per household
- One-Hot Encoding para variables categóricas

### Métricas de Rendimiento

| Métrica | Valor |
|---------|-------|
| **RMSE** | $45,710 |
| **MAE** | $30,041 |
| **R² Score** | 0.8432 |
| **MAPE** | 15.23% |

### Hyperparámetros

```python
{
    'n_estimators': 200,
    'max_depth': 7,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

## 🌐 API

### Endpoints

#### `GET /health`
Health check del servicio

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "model_name": "XGBRegressor",
    "timestamp": "2025-11-10T20:00:00"
}
```

#### `POST /predict`
Realizar predicción de precio

**Request:**
```json
{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
}
```

**Response:**
```json
{
    "prediction": 419304.53,
    "model_name": "XGBRegressor",
    "timestamp": "2025-11-10T20:00:00"
}
```

#### `GET /model/info`
Información del modelo

**Response:**
```json
{
    "model_name": "XGBRegressor",
    "version": "1.0",
    "features": [...],
    "trained_date": "2025-11-10"
}
```

## 📊 Monitoreo

### Dashboard de Drift Detection

El sistema incluye monitoreo automatizado de drift con 4 tests estadísticos:

1. **Kolmogorov-Smirnov Test**
   - Detecta cambios en la distribución general
   - Umbral: p-value < 0.05

2. **Population Stability Index (PSI)**
   - Mide estabilidad de poblaciones
   - Umbral: PSI > 0.1 (warning), PSI > 0.25 (critical)

3. **Jensen-Shannon Distance**
   - Distancia entre distribuciones
   - Umbral: JSD > 0.1

4. **Chi-Squared Test**
   - Para variables categóricas
   - Umbral: p-value < 0.05

### Alertas

- 🟢 **Sin drift**: Todos los tests pasan
- 🟡 **Warning**: 1-2 tests fallan
- 🔴 **Critical**: 3+ tests fallan

## 🐳 Docker

### Construir Imagen

```bash
docker build -t mlops-pipeline:latest .
```

### Ejecutar Contenedor

```bash
# Solo API
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  mlops-pipeline:latest

# Con Docker Compose (todos los servicios)
docker-compose up -d
```

### Servicios en Docker Compose

- **api**: FastAPI (puerto 8000)
- **dashboard**: Streamlit Monitoring (puerto 8501)
- **prediction**: Streamlit Prediction (puerto 8502)

### Detener servicios

```bash
docker-compose down
```

## 🔍 Calidad de Código

### SonarCloud

El proyecto está integrado con SonarCloud para análisis continuo de calidad:

- **Quality Gate**: ✅ Passed
- **Bugs**: 0
- **Code Smells**: 17 → 4 (mejorado)
- **Coverage**: 0.0% (sin tests unitarios)
- **Duplications**: 0.0%
- **Lines of Code**: 237

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=S4V4CH_mlops-pipeline&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=S4V4CH_mlops-pipeline)

### GitHub Actions

Workflow automático que ejecuta:
- Análisis de SonarCloud en cada push
- Verificación de calidad de código
- Reporte de métricas

## 📈 Resultados

### Rendimiento del Modelo

- **Precisión**: El modelo predice precios con un error promedio de $30,041
- **Explicabilidad**: R² de 0.84 indica que el modelo explica el 84% de la varianza
- **Rapidez**: Predicciones en menos de 100ms

### Visualizaciones

El proyecto incluye:
- Distribuciones de features
- Matriz de correlación
- Importancia de features
- Residual plots
- Learning curves
- Drift reports

## 🤝 Contribución

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 👤 Autor

- GitHub: [@S4V4CH](https://github.com/S4V4CH)
- Proyecto: [mlops-pipeline](https://github.com/S4V4CH/mlops-pipeline)


