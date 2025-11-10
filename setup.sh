#!/bin/bash

echo "=========================================="
echo "  MLOps Pipeline - Setup Script"
echo "=========================================="
echo

# Leer configuración
PROJECT_CODE="mlops-pipeline"
VENV_NAME="${PROJECT_CODE}-venv"

# Crear entorno virtual
echo "📦 Creando entorno virtual: $VENV_NAME"
python3 -m venv "$VENV_NAME"

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source "$VENV_NAME/bin/activate"

# Actualizar pip
echo "⬆️  Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
echo "📥 Instalando dependencias desde requirements.txt..."
pip install -r requirements.txt

# Registrar kernel de Jupyter
echo "📓 Registrando kernel de Jupyter..."
python -m ipykernel install --user --name="$VENV_NAME" --display-name="$VENV_NAME"

echo
echo "=========================================="
echo "  ✅ Setup completado"
echo "=========================================="
echo "Para activar el entorno virtual:"
echo "  source $VENV_NAME/bin/activate"
echo
echo "Para usar en VS Code Jupyter:"
echo "  Selecciona el kernel: $VENV_NAME"
echo "=========================================="

echo "=== Python Virtual Environment Setup ==="
echo

# Desactivar entorno actual si existe
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Desactivando ambiente virtual actual: $VIRTUAL_ENV"
    deactivate 2>/dev/null || echo "(No se pudo desactivar, continuando...)"
fi

echo "Buscando código del proyecto en config.json..."

# Cambiar al directorio donde está config.json
cd etl_scripts/src || { echo "Directorio etl_scripts/src no encontrado"; exit 1; }

# Leer el valor de project_code desde config.json
project_code=$(grep -oP '(?<="project_code":\s*")[^"]+' config.json)

if [[ -z "$project_code" ]]; then
    echo "Error: No se pudo leer 'project_code' de config.json"
    exit 1
fi

# Volver al directorio raíz
cd ../..

echo "Creando nuevo ambiente virtual: ${project_code}-venv"
python3 -m venv "${project_code}-venv"

echo "Activando ambiente virtual..."
source "${project_code}-venv/bin/activate"

if [[ $? -eq 0 ]]; then
    echo
    echo "Ambiente virtual creado con éxito."
    echo "Python actual:"
    which python

    echo
    echo "=== Instalando requisitos ==="
    if [[ -f requirements.txt ]]; then
        echo "requirements.txt encontrado, instalando librerías..."
        pip install --no-cache-dir -r requirements.txt

        if [[ $? -eq 0 ]]; then
            echo
            echo "Todas las librerías instaladas correctamente."
            echo
            echo "=== Registrando ambiente virtual con Jupyter ==="
            python -m ipykernel install --user --name="${project_code}-venv" --display-name="${project_code}-venv Python ETL"

            if [[ $? -eq 0 ]]; then
                echo "Ambiente virtual registrado como kernel de Jupyter correctamente."
                echo "Ahora puedes seleccionar '${project_code}-venv Python ETL' en Jupyter notebook."
            else
                echo "Advertencia: Falló el registro del kernel de Jupyter."
            fi
        else
            echo "Error instalando las librerías desde requirements.txt."
        fi
    else
        echo "Advertencia: requirements.txt no encontrado en el directorio actual."
    fi
else
    echo "Error activando el ambiente virtual."
fi
