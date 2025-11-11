"""
Test API - Script de Prueba para Model Deployment API
======================================================

Script para validar los endpoints de la API de predicción.
Incluye tests para predicción individual, batch, y health checks.

Autor: MLOps Pipeline Project
Fecha: Noviembre 2025

Uso:
    python test_api.py
"""

import requests
import json
from datetime import datetime

# Configuración
API_BASE_URL = "http://localhost:8000"

# Colores para output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_header(text):
    """Imprimir encabezado"""
    print(f"\n{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BLUE}{text.center(80)}{Colors.END}")
    print(f"{Colors.BLUE}{'='*80}{Colors.END}\n")

def print_success(text):
    """Imprimir éxito"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    """Imprimir error"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text):
    """Imprimir información"""
    print(f"{Colors.YELLOW}ℹ {text}{Colors.END}")

def test_root_endpoint():
    """Test del endpoint raíz"""
    print_header("TEST 1: Root Endpoint")
    
    try:
        response = requests.get(f"{API_BASE_URL}/")
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Status Code: {response.status_code}")
            print_info(f"Service: {data.get('service')}")
            print_info(f"Version: {data.get('version')}")
            print_info(f"Status: {data.get('status')}")
            print(f"\nEndpoints disponibles:")
            for key, value in data.get('endpoints', {}).items():
                print(f"  • {key}: {value}")
            return True
        else:
            print_error(f"Status Code: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_health_endpoint():
    """Test del endpoint de health check"""
    print_header("TEST 2: Health Check Endpoint")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Status Code: {response.status_code}")
            print_info(f"Status: {data.get('status')}")
            print_info(f"Model Loaded: {data.get('model_loaded')}")
            print_info(f"Model Name: {data.get('model_name', 'N/A')}")
            print_info(f"Timestamp: {data.get('timestamp')}")
            
            if data.get('model_loaded'):
                print_success("Modelo cargado correctamente")
                return True
            else:
                print_error("Modelo no está cargado")
                return False
        else:
            print_error(f"Status Code: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_predict_endpoint():
    """Test del endpoint de predicción individual"""
    print_header("TEST 3: Predict Endpoint (Individual)")
    
    # Datos de ejemplo
    test_data = {
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
    
    print_info("Datos de entrada:")
    print(json.dumps(test_data, indent=2))
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Status Code: {response.status_code}")
            print_success(f"Predicción: ${data.get('prediction'):,.2f}")
            print_info(f"Modelo: {data.get('model_name')}")
            print_info(f"Timestamp: {data.get('timestamp')}")
            return True
        else:
            print_error(f"Status Code: {response.status_code}")
            print_error(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_predict_batch_endpoint():
    """Test del endpoint de predicción por lotes"""
    print_header("TEST 4: Predict Batch Endpoint")
    
    # Datos de ejemplo para batch
    test_data = {
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
            },
            {
                "longitude": -118.25,
                "latitude": 34.05,
                "housing_median_age": 35.0,
                "total_rooms": 1500.0,
                "total_bedrooms": 300.0,
                "population": 800.0,
                "households": 280.0,
                "median_income": 5.5,
                "ocean_proximity": "<1H OCEAN"
            },
            {
                "longitude": -121.89,
                "latitude": 37.34,
                "housing_median_age": 28.0,
                "total_rooms": 2000.0,
                "total_bedrooms": 400.0,
                "population": 1200.0,
                "households": 390.0,
                "median_income": 7.8,
                "ocean_proximity": "INLAND"
            }
        ]
    }
    
    print_info(f"Número de instancias: {len(test_data['instances'])}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Status Code: {response.status_code}")
            print_success(f"Predicciones realizadas: {data.get('count')}")
            print_info(f"Modelo: {data.get('model_name')}")
            print_info(f"Timestamp: {data.get('timestamp')}")
            
            print("\nPredicciones:")
            for i, pred in enumerate(data.get('predictions', []), 1):
                print(f"  {i}. ${pred:,.2f}")
            
            return True
        else:
            print_error(f"Status Code: {response.status_code}")
            print_error(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_model_info_endpoint():
    """Test del endpoint de información del modelo"""
    print_header("TEST 5: Model Info Endpoint")
    
    try:
        response = requests.get(f"{API_BASE_URL}/model/info")
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Status Code: {response.status_code}")
            print_info(f"Model Type: {data.get('model_type')}")
            print_info(f"Preprocessor Available: {data.get('preprocessor_available')}")
            print_info(f"Feature Count: {data.get('feature_count')}")
            
            if 'model_parameters' in data:
                print("\nParámetros del modelo:")
                for key, value in list(data['model_parameters'].items())[:5]:  # Solo primeros 5
                    print(f"  • {key}: {value}")
            
            return True
        else:
            print_error(f"Status Code: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_invalid_input():
    """Test con entrada inválida"""
    print_header("TEST 6: Invalid Input Handling")
    
    # Datos inválidos (ocean_proximity incorrecto)
    invalid_data = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "ocean_proximity": "INVALID_CATEGORY"  # Categoría inválida
    }
    
    print_info("Probando con datos inválidos (ocean_proximity incorrecto)")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=invalid_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 422:  # Validation Error
            print_success(f"Status Code: {response.status_code} (Validación correcta)")
            print_info("La API rechazó correctamente los datos inválidos")
            return True
        else:
            print_error(f"Status Code inesperado: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def run_all_tests():
    """Ejecutar todos los tests"""
    print_header("INICIANDO TESTS DE LA API")
    print(f"URL Base: {API_BASE_URL}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Lista de tests
    tests = [
        ("Root Endpoint", test_root_endpoint),
        ("Health Check", test_health_endpoint),
        ("Predicción Individual", test_predict_endpoint),
        ("Predicción Batch", test_predict_batch_endpoint),
        ("Model Info", test_model_info_endpoint),
        ("Invalid Input", test_invalid_input)
    ]
    
    results = []
    
    # Ejecutar tests
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"Error ejecutando {test_name}: {str(e)}")
            results.append((test_name, False))
    
    # Resumen
    print_header("RESUMEN DE TESTS")
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    print(f"\nTests ejecutados: {len(results)}")
    print_success(f"Exitosos: {passed}")
    if failed > 0:
        print_error(f"Fallidos: {failed}")
    else:
        print_info(f"Fallidos: {failed}")
    
    print("\nDetalle:")
    for test_name, result in results:
        status = f"{Colors.GREEN}✓ PASS{Colors.END}" if result else f"{Colors.RED}✗ FAIL{Colors.END}"
        print(f"  {status} - {test_name}")
    
    print("\n" + "="*80)
    
    if failed == 0:
        print_success("\n🎉 ¡Todos los tests pasaron exitosamente!")
    else:
        print_error(f"\n⚠️  {failed} test(s) fallaron. Revisa los logs arriba.")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                    API Testing Script - MLOps Pipeline                    ║
    ║                   California Housing Price Prediction API                 ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print_info("Asegúrate de que la API está corriendo en http://localhost:8000")
    print_info("Inicia la API con: uvicorn src.model_deploy:app --reload\n")
    
    input("Presiona Enter para comenzar los tests...")
    
    run_all_tests()
