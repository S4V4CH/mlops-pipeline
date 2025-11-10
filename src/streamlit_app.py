"""
Streamlit Dashboard - Data Drift Monitoring
=============================================

Dashboard interactivo para visualización de métricas de data drift
y monitoreo del modelo en producción.

Autor: MLOps Pipeline Project
Fecha: Noviembre 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import glob

# Configuración de la página
st.set_page_config(
    page_title="Data Drift Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("📊 Data Drift Monitoring Dashboard")
st.markdown("---")

# Sidebar - Configuración
st.sidebar.header("⚙️ Configuración")

# Cargar archivos de monitoreo disponibles
MONITORING_DIR = '../monitoring_reports/'

if os.path.exists(MONITORING_DIR):
    drift_files = sorted(glob.glob(os.path.join(MONITORING_DIR, 'drift_summary_*.csv')), reverse=True)
    alert_files = sorted(glob.glob(os.path.join(MONITORING_DIR, 'alerts_*.json')), reverse=True)
    
    if len(drift_files) > 0:
        # Selector de archivo
        selected_file = st.sidebar.selectbox(
            "Seleccionar reporte:",
            drift_files,
            format_func=lambda x: os.path.basename(x)
        )
        
        # Cargar datos
        drift_summary = pd.read_csv(selected_file)
        
        # Cargar alertas correspondientes
        timestamp = os.path.basename(selected_file).replace('drift_summary_', '').replace('.csv', '')
        alert_file = os.path.join(MONITORING_DIR, f'alerts_{timestamp}.json')
        
        if os.path.exists(alert_file):
            with open(alert_file, 'r') as f:
                alerts_data = json.load(f)
        else:
            alerts_data = None
        
        # Mostrar fecha del reporte
        st.sidebar.info(f"📅 Reporte: {timestamp}")
        
        # Configuración de umbrales
        st.sidebar.markdown("### Umbrales de Alerta")
        threshold_critical = st.sidebar.slider("Crítico", 50, 100, 75)
        threshold_high = st.sidebar.slider("Alto", 25, 75, 50)
        threshold_moderate = st.sidebar.slider("Moderado", 0, 50, 25)
        
        # Métricas principales en el sidebar
        st.sidebar.markdown("### 📈 Métricas Generales")
        st.sidebar.metric("Total Features", len(drift_summary))
        st.sidebar.metric("Features con Drift", 
                         len(drift_summary[drift_summary['drift_score'] >= threshold_moderate]))
        st.sidebar.metric("Drift Score Promedio", 
                         f"{drift_summary['drift_score'].mean():.2f}")
        
        # Sección 1: Resumen Ejecutivo
        st.header("🎯 Resumen Ejecutivo")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_features = len(drift_summary)
            st.metric("Total Features", total_features)
        
        with col2:
            critical_count = len(drift_summary[drift_summary['drift_score'] >= threshold_critical])
            st.metric("Features Críticos", critical_count, 
                     delta=None if critical_count == 0 else "⚠️",
                     delta_color="inverse")
        
        with col3:
            avg_psi = drift_summary['psi'].mean()
            st.metric("PSI Promedio", f"{avg_psi:.4f}")
        
        with col4:
            avg_js = drift_summary['js_divergence'].mean()
            st.metric("JS Div Promedio", f"{avg_js:.4f}")
        
        # Mostrar alertas si existen
        if alerts_data:
            st.markdown("### 🚨 Alertas Activas")
            for alert in alerts_data.get('alerts', []):
                if alert['level'] == '🔴 CRÍTICO':
                    st.error(f"**{alert['level']}**: {alert['message']}\n\n"
                            f"**Recomendación**: {alert['recommendation']}")
                elif alert['level'] == '🟠 ALTO':
                    st.warning(f"**{alert['level']}**: {alert['message']}\n\n"
                              f"**Recomendación**: {alert['recommendation']}")
                elif alert['level'] == '🟡 MODERADO':
                    st.info(f"**{alert['level']}**: {alert['message']}\n\n"
                           f"**Recomendación**: {alert['recommendation']}")
                else:
                    st.success(f"**{alert['level']}**: {alert['message']}\n\n"
                              f"**Recomendación**: {alert['recommendation']}")
        
        st.markdown("---")
        
        # Sección 2: Distribución de Riesgo
        st.header("📊 Distribución de Riesgo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart de distribución de riesgo
            risk_counts = drift_summary['risk_level'].value_counts()
            fig_pie = go.Figure(data=[go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.3,
                marker=dict(colors=['green', 'yellow', 'orange', 'red'])
            )])
            fig_pie.update_layout(title="Distribución de Features por Nivel de Riesgo")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart de top 10 features
            top_10 = drift_summary.nlargest(10, 'drift_score')
            fig_bar = go.Figure(data=[go.Bar(
                x=top_10['feature'],
                y=top_10['drift_score'],
                marker=dict(
                    color=top_10['drift_score'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Drift Score")
                )
            )])
            fig_bar.update_layout(
                title="Top 10 Features con Mayor Drift",
                xaxis_title="Feature",
                yaxis_title="Drift Score",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("---")
        
        # Sección 3: Análisis Detallado por Métrica
        st.header("🔍 Análisis Detallado por Métrica")
        
        tab1, tab2, tab3 = st.tabs(["📈 PSI", "📉 Jensen-Shannon", "📊 KS Test"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                # Histograma de PSI
                fig_psi_hist = px.histogram(
                    drift_summary, 
                    x='psi', 
                    nbins=30,
                    title="Distribución de PSI",
                    labels={'psi': 'Population Stability Index'}
                )
                fig_psi_hist.add_vline(x=0.1, line_dash="dash", line_color="yellow", 
                                      annotation_text="Umbral Moderado")
                fig_psi_hist.add_vline(x=0.2, line_dash="dash", line_color="red", 
                                      annotation_text="Umbral Crítico")
                st.plotly_chart(fig_psi_hist, use_container_width=True)
            
            with col2:
                # Top features por PSI
                top_psi = drift_summary.nlargest(10, 'psi')
                fig_psi_bar = px.bar(
                    top_psi,
                    x='feature',
                    y='psi',
                    title="Top 10 Features por PSI",
                    color='psi',
                    color_continuous_scale='Reds'
                )
                fig_psi_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_psi_bar, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                # Histograma de JS Divergence
                fig_js_hist = px.histogram(
                    drift_summary, 
                    x='js_divergence', 
                    nbins=30,
                    title="Distribución de Jensen-Shannon Divergence",
                    labels={'js_divergence': 'JS Divergence'}
                )
                fig_js_hist.add_vline(x=0.1, line_dash="dash", line_color="yellow", 
                                     annotation_text="Umbral Moderado")
                fig_js_hist.add_vline(x=0.3, line_dash="dash", line_color="red", 
                                     annotation_text="Umbral Crítico")
                st.plotly_chart(fig_js_hist, use_container_width=True)
            
            with col2:
                # Top features por JS
                top_js = drift_summary.nlargest(10, 'js_divergence')
                fig_js_bar = px.bar(
                    top_js,
                    x='feature',
                    y='js_divergence',
                    title="Top 10 Features por JS Divergence",
                    color='js_divergence',
                    color_continuous_scale='Reds'
                )
                fig_js_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_js_bar, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                # KS Test - Features con drift detectado
                ks_drift_count = drift_summary['ks_drift'].sum()
                fig_ks_pie = go.Figure(data=[go.Pie(
                    labels=['Drift Detectado', 'Sin Drift'],
                    values=[ks_drift_count, len(drift_summary) - ks_drift_count],
                    hole=0.3,
                    marker=dict(colors=['red', 'green'])
                )])
                fig_ks_pie.update_layout(title="KS Test - Detección de Drift")
                st.plotly_chart(fig_ks_pie, use_container_width=True)
            
            with col2:
                # Top features por KS statistic
                top_ks = drift_summary.nlargest(10, 'ks_statistic')
                fig_ks_bar = px.bar(
                    top_ks,
                    x='feature',
                    y='ks_statistic',
                    title="Top 10 Features por KS Statistic",
                    color='ks_drift',
                    color_discrete_map={True: 'red', False: 'green'}
                )
                fig_ks_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_ks_bar, use_container_width=True)
        
        st.markdown("---")
        
        # Sección 4: Tabla Detallada
        st.header("📋 Tabla Detallada de Drift por Feature")
        
        # Filtros
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_filter = st.multiselect(
                "Filtrar por nivel de riesgo:",
                options=drift_summary['risk_level'].unique(),
                default=drift_summary['risk_level'].unique()
            )
        with col2:
            psi_filter = st.multiselect(
                "Filtrar por PSI status:",
                options=drift_summary['psi_status'].unique(),
                default=drift_summary['psi_status'].unique()
            )
        with col3:
            ks_filter = st.selectbox(
                "Filtrar por KS drift:",
                options=['Todos', 'Con drift', 'Sin drift']
            )
        
        # Aplicar filtros
        filtered_df = drift_summary[
            (drift_summary['risk_level'].isin(risk_filter)) &
            (drift_summary['psi_status'].isin(psi_filter))
        ]
        
        if ks_filter == 'Con drift':
            filtered_df = filtered_df[filtered_df['ks_drift'] == True]
        elif ks_filter == 'Sin drift':
            filtered_df = filtered_df[filtered_df['ks_drift'] == False]
        
        # Mostrar tabla
        st.dataframe(
            filtered_df.sort_values('drift_score', ascending=False),
            use_container_width=True,
            height=400
        )
        
        # Botón de descarga
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar datos filtrados (CSV)",
            data=csv,
            file_name=f'drift_report_filtered_{timestamp}.csv',
            mime='text/csv'
        )
        
        st.markdown("---")
        
        # Sección 5: Recomendaciones
        st.header("💡 Recomendaciones")
        
        max_drift_score = drift_summary['drift_score'].max()
        
        if max_drift_score >= threshold_critical:
            st.error("""
            ### 🔴 ACCIÓN CRÍTICA REQUERIDA
            
            **Se ha detectado drift crítico que compromete la precisión del modelo.**
            
            **Acciones recomendadas:**
            1. ⚠️ Pausar predicciones en producción si es posible
            2. 🔄 Reentrenar el modelo INMEDIATAMENTE con datos actuales
            3. 📊 Realizar análisis profundo de los features con mayor drift
            4. ✅ Validar el nuevo modelo antes de deployment
            5. 📝 Documentar las causas del drift detectado
            """)
        elif max_drift_score >= threshold_high:
            st.warning("""
            ### 🟠 ALERTA: Drift Significativo Detectado
            
            **Se recomienda planificar reentrenamiento en corto plazo.**
            
            **Acciones recomendadas:**
            1. 📅 Programar reentrenamiento del modelo en los próximos 7 días
            2. 📈 Monitorear métricas de performance del modelo diariamente
            3. 🔍 Investigar causas del drift en los features más afectados
            4. 📊 Preparar dataset actualizado para reentrenamiento
            5. 🔔 Configurar alertas automáticas
            """)
        elif max_drift_score >= threshold_moderate:
            st.info("""
            ### 🟡 ADVERTENCIA: Drift Moderado
            
            **Monitorear de cerca la evolución del drift.**
            
            **Acciones recomendadas:**
            1. 📊 Continuar monitoreo diario de drift
            2. 📈 Evaluar tendencias de drift a lo largo del tiempo
            3. 🔍 Analizar si el drift es temporal o persistente
            4. 📝 Documentar patrones observados
            5. ⏰ Considerar reentrenamiento si el drift persiste por 2+ semanas
            """)
        else:
            st.success("""
            ### 🟢 Estado Normal: Sin Drift Significativo
            
            **El modelo está operando dentro de parámetros normales.**
            
            **Acciones recomendadas:**
            1. ✅ Continuar monitoreo regular semanal
            2. 📊 Mantener registro histórico de métricas
            3. 🔄 Planificar reentrenamiento periódico preventivo (cada 3 meses)
            4. 📈 Monitorear métricas de negocio relacionadas
            5. 📝 Mantener documentación actualizada
            """)
        
    else:
        st.warning("⚠️ No se encontraron reportes de monitoreo. Ejecuta `model_monitoring.ipynb` primero.")
else:
    st.error(f"❌ Directorio de monitoreo no encontrado: {MONITORING_DIR}")
    st.info("Ejecuta el notebook `model_monitoring.ipynb` para generar reportes.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>MLOps Pipeline - Data Drift Monitoring Dashboard</strong></p>
    <p>Desarrollado para monitoreo continuo de modelos en producción</p>
</div>
""", unsafe_allow_html=True)
