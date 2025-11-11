"""
Interfaz Web de Predicción - California Housing
================================================

Interfaz interactiva con Streamlit para realizar predicciones
de precios de viviendas usando la API de FastAPI.

Autor: MLOps Pipeline Project
Fecha: Noviembre 2025
"""

import streamlit as st
import requests
import json
import pandas as pd

# ============================================================================
# Configuración de la página
# ============================================================================

st.set_page_config(
    page_title="Predicción de Precios de Viviendas",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Variables globales
# ============================================================================

API_URL = "http://localhost:8000"

# ============================================================================
# Funciones auxiliares
# ============================================================================

def check_api_status():
    """Verifica si la API está disponible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def make_prediction(features):
    """Hace una predicción usando la API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=features,
            timeout=5
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# Interfaz principal
# ============================================================================

# Título
st.title("🏠 Predicción de Precios de Viviendas en California")
st.markdown("---")

# Verificar estado de la API
api_status = check_api_status()

if not api_status:
    st.error("⚠️ La API de predicción no está disponible. Asegúrate de que esté corriendo en http://localhost:8000")
    st.info("💡 Para iniciar la API, ejecuta: `uvicorn model_deploy:app --host 0.0.0.0 --port 8000`")
    st.stop()

st.success("✅ API conectada correctamente")

# ============================================================================
# Sidebar con información
# ============================================================================

with st.sidebar:
    st.header("ℹ️ Información")
    st.markdown("""
    Esta interfaz te permite predecir precios de viviendas en California 
    usando un modelo de Machine Learning entrenado con XGBoost.
    
    **Instrucciones:**
    1. Completa los campos del formulario
    2. Haz clic en "Predecir Precio"
    3. Obtén la predicción instantánea
    
    **Categorías de Ocean Proximity:**
    - `<1H OCEAN`: Menos de 1 hora del océano
    - `INLAND`: Tierra adentro
    - `NEAR OCEAN`: Cerca del océano
    - `NEAR BAY`: Cerca de la bahía
    - `ISLAND`: En una isla
    """)

# ============================================================================
# Formulario de entrada
# ============================================================================

st.header("📝 Datos de la Vivienda")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Ubicación")
    longitude = st.number_input(
        "Longitud",
        min_value=-125.0,
        max_value=-114.0,
        value=-122.23,
        step=0.01,
        help="Coordenada de longitud geográfica"
    )
    
    latitude = st.number_input(
        "Latitud",
        min_value=32.0,
        max_value=42.0,
        value=37.88,
        step=0.01,
        help="Coordenada de latitud geográfica"
    )
    
    ocean_proximity = st.selectbox(
        "Proximidad al Océano",
        options=['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'],
        index=3,
        help="Categoría de proximidad al océano"
    )
    
    st.subheader("🏘️ Características del Área")
    
    population = st.number_input(
        "Población",
        min_value=1,
        max_value=50000,
        value=322,
        step=1,
        help="Población total del área"
    )
    
    households = st.number_input(
        "Hogares",
        min_value=1,
        max_value=10000,
        value=126,
        step=1,
        help="Número de hogares en el área"
    )

with col2:
    st.subheader("🏠 Características de la Vivienda")
    
    housing_median_age = st.slider(
        "Edad Mediana de las Viviendas (años)",
        min_value=1,
        max_value=100,
        value=41,
        help="Edad mediana de las viviendas en el área"
    )
    
    total_rooms = st.number_input(
        "Total de Habitaciones",
        min_value=1,
        max_value=50000,
        value=880,
        step=1,
        help="Número total de habitaciones"
    )
    
    total_bedrooms = st.number_input(
        "Total de Dormitorios",
        min_value=1,
        max_value=10000,
        value=129,
        step=1,
        help="Número total de dormitorios"
    )
    
    st.subheader("💰 Datos Económicos")
    
    median_income = st.number_input(
        "Ingreso Mediano (en $10,000)",
        min_value=0.0,
        max_value=20.0,
        value=8.33,
        step=0.01,
        help="Ingreso mediano en unidades de $10,000 USD"
    )

st.markdown("---")

# ============================================================================
# Botón de predicción
# ============================================================================

if st.button("🔮 Predecir Precio", type="primary", use_container_width=True):
    
    # Preparar datos
    features = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }
    
    # Hacer predicción
    with st.spinner("Realizando predicción..."):
        result = make_prediction(features)
    
    # Mostrar resultado
    if "error" in result:
        st.error(f"❌ Error: {result['error']}")
    else:
        prediction = result.get("prediction", 0)
        model_name = result.get("model_name", "Unknown")
        
        # Mostrar predicción destacada
        st.success("✅ Predicción completada")
        
        st.markdown("### 🎯 Resultado de la Predicción")
        
        # Métrica grande
        col_pred1, col_pred2, col_pred3 = st.columns([2, 1, 1])
        
        with col_pred1:
            st.metric(
                label="💵 Precio Estimado de la Vivienda",
                value=f"${prediction:,.2f}",
                delta=None
            )
        
        with col_pred2:
            st.metric(
                label="🤖 Modelo",
                value=model_name
            )
        
        with col_pred3:
            # Calcular precio por habitación
            price_per_room = prediction / total_rooms if total_rooms > 0 else 0
            st.metric(
                label="📊 Precio/Habitación",
                value=f"${price_per_room:,.2f}"
            )
        
        # Información adicional
        st.markdown("---")
        st.markdown("#### 📋 Detalles de la Predicción")
        
        # Crear DataFrame con los datos ingresados
        input_data = pd.DataFrame([{
            "Longitud": longitude,
            "Latitud": latitude,
            "Edad Viviendas (años)": housing_median_age,
            "Total Habitaciones": total_rooms,
            "Total Dormitorios": total_bedrooms,
            "Población": population,
            "Hogares": households,
            "Ingreso Mediano": f"${median_income * 10000:,.0f}",
            "Proximidad Océano": ocean_proximity
        }])
        
        st.dataframe(input_data.T, use_container_width=True)
        
        # Métricas calculadas
        st.markdown("#### 📈 Métricas Calculadas")
        col_met1, col_met2, col_met3 = st.columns(3)
        
        with col_met1:
            rooms_per_household = total_rooms / households if households > 0 else 0
            st.metric("Habitaciones/Hogar", f"{rooms_per_household:.2f}")
        
        with col_met2:
            bedrooms_per_room = total_bedrooms / total_rooms if total_rooms > 0 else 0
            st.metric("Dormitorios/Habitación", f"{bedrooms_per_room:.2%}")
        
        with col_met3:
            population_per_household = population / households if households > 0 else 0
            st.metric("Población/Hogar", f"{population_per_household:.2f}")

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>🏠 <strong>California Housing Price Prediction</strong></p>
    <p>MLOps Pipeline Project | Noviembre 2025</p>
    <p>Modelo: XGBoost Regressor | API: FastAPI</p>
</div>
""", unsafe_allow_html=True)
