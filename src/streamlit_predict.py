"""
Interfaz de Predicción - California Housing Price Prediction
=============================================================

Interfaz web interactiva con Streamlit para realizar predicciones
de precios de viviendas usando el modelo entrenado.

Autor: MLOps Pipeline Project
Fecha: Noviembre 2025
"""

import streamlit as st
import requests
import pandas as pd
import json

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Precios - California Housing",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL de la API
API_URL = "http://localhost:8000"

# ============================================================================
# Funciones auxiliares
# ============================================================================

def check_api_health():
    """Verificar si la API está disponible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def make_prediction(features):
    """Hacer una predicción usando la API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=features,
            headers={"Content-Type": "application/json"},
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

st.title("🏠 Predicción de Precios de Viviendas")
st.markdown("### California Housing Dataset - Predictor en Tiempo Real")

# Verificar estado de la API
if check_api_health():
    st.success("✅ API conectada y funcionando")
else:
    st.error("❌ API no disponible. Asegúrate de que el servidor esté corriendo en puerto 8000")
    st.info("Ejecuta: `uvicorn model_deploy:app --host 0.0.0.0 --port 8000 --reload`")
    st.stop()

st.markdown("---")

# ============================================================================
# Formulario de entrada
# ============================================================================

st.header("📝 Ingresa los datos de la vivienda")

col1, col2, col3 = st.columns(3)

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
        "Proximidad al océano",
        options=['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'],
        index=3,
        help="Cercanía de la vivienda al océano"
    )

with col2:
    st.subheader("🏘️ Características de la Vivienda")
    housing_median_age = st.slider(
        "Edad mediana de las viviendas (años)",
        min_value=1,
        max_value=100,
        value=41,
        help="Edad promedio de las viviendas en el bloque"
    )
    
    total_rooms = st.number_input(
        "Total de habitaciones",
        min_value=1,
        value=880,
        step=10,
        help="Número total de habitaciones en el bloque"
    )
    
    total_bedrooms = st.number_input(
        "Total de dormitorios",
        min_value=1,
        value=129,
        step=5,
        help="Número total de dormitorios en el bloque"
    )

with col3:
    st.subheader("👥 Población y Economía")
    population = st.number_input(
        "Población",
        min_value=1,
        value=322,
        step=10,
        help="Población total del bloque"
    )
    
    households = st.number_input(
        "Número de hogares",
        min_value=1,
        value=126,
        step=5,
        help="Número de hogares en el bloque"
    )
    
    median_income = st.number_input(
        "Ingreso mediano ($10,000)",
        min_value=0.0,
        max_value=15.0,
        value=8.3252,
        step=0.1,
        help="Ingreso mediano en unidades de $10,000"
    )

st.markdown("---")

# ============================================================================
# Botón de predicción
# ============================================================================

col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    predict_button = st.button("🔮 Predecir Precio", type="primary", use_container_width=True)

if predict_button:
    # Preparar datos para la API
    features = {
        "longitude": float(longitude),
        "latitude": float(latitude),
        "housing_median_age": float(housing_median_age),
        "total_rooms": float(total_rooms),
        "total_bedrooms": float(total_bedrooms),
        "population": float(population),
        "households": float(households),
        "median_income": float(median_income),
        "ocean_proximity": ocean_proximity
    }
    
    # Hacer predicción
    with st.spinner("🤖 Realizando predicción..."):
        result = make_prediction(features)
    
    # Mostrar resultados
    if "error" in result:
        st.error(f"❌ Error: {result['error']}")
    else:
        st.success("✅ Predicción realizada exitosamente")
        
        # Resultado principal
        st.markdown("---")
        st.markdown("## 💰 Precio Predicho")
        
        predicted_price = result['prediction']
        
        # Mostrar precio en formato grande
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                f"""
                <div style='text-align: center; padding: 30px; background-color: #f0f2f6; border-radius: 10px;'>
                    <h1 style='color: #1f77b4; font-size: 60px; margin: 0;'>
                        ${predicted_price:,.2f}
                    </h1>
                    <p style='color: #666; font-size: 18px; margin-top: 10px;'>
                        Precio estimado de la vivienda
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Información adicional
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Detalles de la Predicción")
            st.write(f"**Modelo utilizado:** {result.get('model_name', 'N/A')}")
            st.write(f"**Timestamp:** {result.get('timestamp', 'N/A')}")
            
            # Calcular métricas derivadas
            rooms_per_household = total_rooms / households
            bedrooms_per_room = total_bedrooms / total_rooms
            population_per_household = population / households
            
            st.write(f"**Habitaciones por hogar:** {rooms_per_household:.2f}")
            st.write(f"**Proporción dormitorios:** {bedrooms_per_room:.2f}")
            st.write(f"**Personas por hogar:** {population_per_household:.2f}")
        
        with col2:
            st.markdown("### 💵 Análisis del Precio")
            
            # Rangos de referencia
            if predicted_price < 150000:
                st.info("🟢 Precio bajo - Vivienda económica")
            elif predicted_price < 300000:
                st.info("🟡 Precio medio - Vivienda accesible")
            elif predicted_price < 450000:
                st.warning("🟠 Precio alto - Vivienda premium")
            else:
                st.error("🔴 Precio muy alto - Vivienda de lujo")
            
            # Precio por habitación
            price_per_room = predicted_price / total_rooms
            st.write(f"**Precio por habitación:** ${price_per_room:,.2f}")
            
            # Precio por hogar
            price_per_household = predicted_price / households
            st.write(f"**Precio por hogar:** ${price_per_household:,.2f}")

# ============================================================================
# Sidebar con información
# ============================================================================

with st.sidebar:
    st.header("ℹ️ Información")
    
    st.markdown("""
    ### Acerca de esta aplicación
    
    Esta interfaz permite predecir precios de viviendas en California 
    usando un modelo de Machine Learning entrenado con XGBoost.
    
    **Características del modelo:**
    - Algoritmo: XGBoost Regressor
    - R² Score: ~0.84
    - RMSE: ~$45,710
    - MAE: ~$30,041
    
    ### 📝 Instrucciones
    
    1. Ingresa las características de la vivienda
    2. Haz clic en "Predecir Precio"
    3. Obtén el precio estimado
    
    ### 🔗 Enlaces
    
    - [API Docs](http://localhost:8000/docs)
    - [API Health](http://localhost:8000/health)
    - [Dashboard de Monitoreo](http://localhost:8501)
    """)
    
    st.markdown("---")
    
    # Mostrar ejemplo de datos
    with st.expander("📋 Ver ejemplo de datos"):
        example_data = {
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
        st.json(example_data)

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🏠 California Housing Price Prediction | MLOps Pipeline Project 2025</p>
    </div>
    """,
    unsafe_allow_html=True
)
