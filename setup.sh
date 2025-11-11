#!/bin/bash

echo "=========================================="
echo "  MLOps Pipeline - Setup Script"
echo "=========================================="
echo

# Desactivar entorno actual si existe
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Desactivando ambiente virtual actual: $VIRTUAL_ENV"
    deactivate 2>/dev/null || true
fi

# Leer configuración del proyecto
PROJECT_CODE="mlops-pipeline"
VENV_NAME="${PROJECT_CODE}-venv"

echo "📋 Configuración del proyecto:"
echo "  • Nombre del proyecto: $PROJECT_CODE"
echo "  • Ambiente virtual: $VENV_NAME"
echo

# Crear entorno virtual
echo "📦 Creando entorno virtual: $VENV_NAME"
python3 -m venv "$VENV_NAME"

if [[ $? -ne 0 ]]; then
    echo "❌ Error: No se pudo crear el ambiente virtual"
    exit 1
fi

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source "$VENV_NAME/bin/activate"

if [[ $? -ne 0 ]]; then
    echo "❌ Error: No se pudo activar el ambiente virtual"
    exit 1
fi

echo "✓ Ambiente virtual activado"
echo "  Python: $(which python)"
echo "  Versión: $(python --version)"
echo

# Actualizar pip
echo "⬆️  Actualizando pip..."
pip install --upgrade pip --quiet

# Verificar requirements.txt
if [[ ! -f requirements.txt ]]; then
    echo "❌ Error: requirements.txt no encontrado"
    exit 1
fi

# Instalar dependencias
echo "📥 Instalando dependencias desde requirements.txt..."
echo "  (Esto puede tomar varios minutos...)"
pip install -r requirements.txt

if [[ $? -ne 0 ]]; then
    echo "❌ Error: Fallo en la instalación de dependencias"
    exit 1
fi

echo "✓ Todas las dependencias instaladas correctamente"
echo

# Registrar kernel de Jupyter
echo "📓 Registrando kernel de Jupyter..."
python -m ipykernel install --user --name="$VENV_NAME" --display-name="$VENV_NAME"

if [[ $? -eq 0 ]]; then
    echo "✓ Kernel de Jupyter registrado exitosamente"
else
    echo "⚠️  Advertencia: Falló el registro del kernel de Jupyter"
fi

echo
echo "=========================================="
echo "  ✅ Setup completado exitosamente"
echo "=========================================="
echo
echo "Para activar el entorno virtual manualmente:"
echo "  source $VENV_NAME/bin/activate"
echo
echo "Para usar en VS Code Jupyter:"
echo "  Selecciona el kernel: $VENV_NAME"
echo
echo "Para verificar la instalación:"
echo "  python --version"
echo "  pip list"
echo
echo "=========================================="
