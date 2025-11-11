"""
Feature Engineering Pipeline - California Housing Dataset

Este módulo contiene las funciones para el procesamiento y transformación de datos:
- Imputación de valores nulos
- Creación de features derivados
- Encoding de variables categóricas
- Escalado de features
- División train/test
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
from datetime import datetime


class FeatureEngineer:
    """
    Clase para realizar feature engineering en el dataset de California Housing.
    """
    
    def __init__(self, config_path='../config.json'):
        """
        Inicializa el Feature Engineer con la configuración del proyecto.
        
        Args:
            config_path (str): Ruta al archivo de configuración JSON
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.random_state = self.config.get('random_state', 42)
        self.preprocessor = None
        self.feature_names = None
        
    def load_data(self):
        """
        Carga el dataset desde el archivo CSV.
        
        Returns:
            pd.DataFrame: Dataset cargado
        """
        data_path = f"../{self.config['data_file']}"
        df = pd.read_csv(data_path)
        print(f"✓ Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
        return df
    
    def create_derived_features(self, df):
        """
        Crea features derivados a partir de las variables originales.
        
        Features creados:
        - rooms_per_household: Promedio de habitaciones por hogar
        - bedrooms_per_room: Proporción de dormitorios respecto al total de habitaciones
        - population_per_household: Promedio de personas por hogar
        
        Args:
            df (pd.DataFrame): Dataset original
            
        Returns:
            pd.DataFrame: Dataset con features derivados
        """
        df = df.copy()
        
        # Feature 1: Habitaciones por hogar
        df['rooms_per_household'] = df['total_rooms'] / df['households']
        
        # Feature 2: Proporción de dormitorios
        df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
        
        # Feature 3: Población por hogar
        df['population_per_household'] = df['population'] / df['households']
        
        # Manejar valores infinitos (división por cero)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        print("✓ Features derivados creados:")
        print("  • rooms_per_household")
        print("  • bedrooms_per_room")
        print("  • population_per_household")
        
        return df
    
    def prepare_features(self, df, target_col='median_house_value'):
        """
        Separa features y target, y clasifica columnas por tipo.
        
        Args:
            df (pd.DataFrame): Dataset completo
            target_col (str): Nombre de la variable objetivo
            
        Returns:
            tuple: (X, y, numeric_features, categorical_features)
        """
        # Separar features y target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Identificar columnas numéricas y categóricas
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"\n✓ Features preparados:")
        print(f"  • Features numéricos: {len(numeric_features)}")
        print(f"  • Features categóricos: {len(categorical_features)}")
        print(f"  • Variable objetivo: {target_col}")
        
        return X, y, numeric_features, categorical_features
    
    def build_preprocessor(self, numeric_features, categorical_features):
        """
        Construye el pipeline de preprocesamiento con transformadores.
        
        Pipeline numérico:
        - SimpleImputer (estrategia: mediana)
        - StandardScaler
        
        Pipeline categórico:
        - SimpleImputer (estrategia: constante con valor 'missing')
        - OneHotEncoder
        
        Args:
            numeric_features (list): Lista de features numéricos
            categorical_features (list): Lista de features categóricos
            
        Returns:
            ColumnTransformer: Preprocessor completo
        """
        # Pipeline para features numéricos
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Pipeline para features categóricos
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combinar transformadores
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        
        print("\n✓ Pipeline de preprocesamiento construido:")
        print("  • Numéricos: Imputación (mediana) + StandardScaler")
        print("  • Categóricos: Imputación (constante) + OneHotEncoder")
        
        return preprocessor
    
    def fit_transform(self, X_train):
        """
        Ajusta y transforma los datos de entrenamiento.
        
        Args:
            X_train (pd.DataFrame): Features de entrenamiento
            
        Returns:
            np.ndarray: Features transformados
        """
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        
        # Obtener nombres de features después de la transformación
        self._extract_feature_names(X_train)
        
        print(f"\n✓ Datos de entrenamiento transformados: {X_train_transformed.shape}")
        
        return X_train_transformed
    
    def transform(self, X_test):
        """
        Transforma los datos de prueba usando el preprocessor ajustado.
        
        Args:
            X_test (pd.DataFrame): Features de prueba
            
        Returns:
            np.ndarray: Features transformados
        """
        if self.preprocessor is None:
            raise ValueError("El preprocessor no ha sido ajustado. Ejecuta fit_transform primero.")
        
        X_test_transformed = self.preprocessor.transform(X_test)
        print(f"✓ Datos de prueba transformados: {X_test_transformed.shape}")
        
        return X_test_transformed
    
    def _extract_feature_names(self, X):
        """
        Extrae los nombres de features después de la transformación.
        
        Args:
            X (pd.DataFrame): Features originales
        """
        feature_names = []
        
        for name, transformer, features in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(features)
            elif name == 'cat':
                # Obtener nombres después de OneHotEncoder
                onehot = transformer.named_steps['onehot']
                cat_features = onehot.get_feature_names_out(features)
                feature_names.extend(cat_features)
        
        self.feature_names = feature_names
        print(f"✓ Total de features después de transformación: {len(self.feature_names)}")
    
    def save_preprocessor(self, output_dir=None):
        """
        Guarda el preprocessor entrenado en disco.
        
        Args:
            output_dir (str): Directorio de salida (usa config si no se especifica)
        """
        if self.preprocessor is None:
            raise ValueError("No hay preprocessor para guardar. Ejecuta fit_transform primero.")
        
        if output_dir is None:
            output_dir = f"../{self.config.get('model_output_dir', 'models')}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        preprocessor_path = os.path.join(output_dir, f'preprocessor_{timestamp}.pkl')
        
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"\n✓ Preprocessor guardado en: {preprocessor_path}")
        
        # Guardar también los nombres de features
        feature_names_path = os.path.join(output_dir, f'feature_names_{timestamp}.pkl')
        joblib.dump(self.feature_names, feature_names_path)
        print(f"✓ Nombres de features guardados en: {feature_names_path}")
        
        return preprocessor_path
    
    def load_preprocessor(self, preprocessor_path):
        """
        Carga un preprocessor previamente guardado.
        
        Args:
            preprocessor_path (str): Ruta al archivo del preprocessor
        """
        self.preprocessor = joblib.load(preprocessor_path)
        print(f"✓ Preprocessor cargado desde: {preprocessor_path}")


def prepare_data_for_training(config_path='../config.json', test_size=0.2, save_preprocessor=True):
    """
    Función principal para preparar los datos para entrenamiento.
    
    Realiza todo el pipeline de feature engineering:
    1. Carga de datos
    2. Creación de features derivados
    3. Separación de features y target
    4. Train/test split
    5. Construcción y aplicación del preprocessor
    6. Guardado del preprocessor (opcional)
    
    Args:
        config_path (str): Ruta al archivo de configuración
        test_size (float): Proporción de datos para test (default: 0.2)
        save_preprocessor (bool): Si se debe guardar el preprocessor (default: True)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor, feature_names)
    """
    print("=" * 80)
    print("FEATURE ENGINEERING PIPELINE - CALIFORNIA HOUSING")
    print("=" * 80)
    
    # Inicializar Feature Engineer
    fe = FeatureEngineer(config_path)
    
    # 1. Cargar datos
    print("\n[1/6] Cargando datos...")
    df = fe.load_data()
    
    # 2. Crear features derivados
    print("\n[2/6] Creando features derivados...")
    df = fe.create_derived_features(df)
    
    # 3. Preparar features y target
    print("\n[3/6] Preparando features y target...")
    X, y, numeric_features, categorical_features = fe.prepare_features(df)
    
    # 4. Train/test split
    print(f"\n[4/6] Dividiendo datos (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=fe.random_state,
        shuffle=True
    )
    print(f"  • Train: {X_train.shape[0]:,} muestras")
    print(f"  • Test: {X_test.shape[0]:,} muestras")
    
    # 5. Construir y aplicar preprocessor
    print("\n[5/6] Aplicando transformaciones...")
    fe.preprocessor = fe.build_preprocessor(numeric_features, categorical_features)
    X_train_transformed = fe.fit_transform(X_train)
    X_test_transformed = fe.transform(X_test)
    
    # 6. Guardar preprocessor
    if save_preprocessor:
        print("\n[6/6] Guardando preprocessor...")
        fe.save_preprocessor()
    else:
        print("\n[6/6] Preprocessor no guardado (save_preprocessor=False)")
    
    print("\n" + "=" * 80)
    print("✅ FEATURE ENGINEERING COMPLETADO")
    print("=" * 80)
    print(f"\nDimensiones finales:")
    print(f"  • X_train: {X_train_transformed.shape}")
    print(f"  • X_test: {X_test_transformed.shape}")
    print(f"  • y_train: {y_train.shape}")
    print(f"  • y_test: {y_test.shape}")
    print(f"  • Total features: {len(fe.feature_names)}")
    print("=" * 80)
    
    return X_train_transformed, X_test_transformed, y_train, y_test, fe.preprocessor, fe.feature_names


# Ejemplo de uso
if __name__ == "__main__":
    # Ejecutar pipeline completo
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data_for_training()
    
    print("\n📊 Primeros 10 features:")
    for i, name in enumerate(feature_names[:10], 1):
        print(f"  {i:2d}. {name}")
    
    if len(feature_names) > 10:
        print(f"  ... y {len(feature_names) - 10} más")
