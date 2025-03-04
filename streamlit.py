import streamlit as st
import pandas as pd
import joblib  
import numpy as np 
import os

# Debug function to check file existence
def check_file_exists(filepath):
    if not os.path.exists(filepath):
        st.error(f"File not found: {filepath}")
        return False
    return True

# Cargar modelos con manejo de errores
@st.cache_resource
def load_pipeline():
    pipeline_path = "modelo_correcto.joblib"
    if not check_file_exists(pipeline_path):
        st.stop()
    try:
        return joblib.load(pipeline_path)
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        st.stop()

@st.cache_resource
def load_price_model():
    price_model_path = "random_forest_pipeline_compressed.joblib"
    if not check_file_exists(price_model_path):
        st.stop()
    try:
        return joblib.load(price_model_path)
    except Exception as e:
        st.error(f"Error loading price model: {e}")
        st.stop()

# Cargar datos con manejo de errores
def load_cars_data():
    data_path = "df_modelo_limpio.csv"
    if not check_file_exists(data_path):
        st.stop()
    try:
        return pd.read_csv(data_path)
    except Exception as e:
        st.error(f"Error loading car data: {e}")
        st.stop()

# Configuración de página
st.set_page_config(layout="wide", page_title="AutoMatch", page_icon="🚗")

# Sistema de debugging de archivos
def debug_file_system():
    st.header("🔍 Debugging de Archivos")
    
    files_to_check = [
        "modelo_correcto.joblib",
        "random_forest_pipeline_compressed.joblib",
        "df_modelo_limpio.csv"
    ]
    
    st.write("### Archivos Necesarios:")
    for file in files_to_check:
        if os.path.exists(file):
            st.success(f"✅ {file} encontrado")
        else:
            st.error(f"❌ {file} no encontrado")
    
    st.write("### Directorio Actual:")
    st.write(os.getcwd())
    
    st.write("### Contenido del Directorio:")
    st.write(os.listdir())

# Resto del código de la aplicación sigue igual...
# (Incluye todas las funciones anteriores: pagina_inicio, buscador_coches, valoracion_coches, etc.)

def main():
    st.sidebar.header("Menú")
    pagina = st.sidebar.radio(
        "Selecciona una opción", 
        ["Inicio", "Buscador de Coches", "Valoración de Coches", "Debugging de Sistema"]
    )

    # Intentar cargar datos antes de mostrar páginas que los necesitan
    try:
        df_cars = load_cars_data()
    except Exception as e:
        st.error(f"No se pudieron cargar los datos: {e}")
        df_cars = None

    if pagina == "Inicio":
        pagina_inicio()
    elif pagina == "Buscador de Coches":
        if df_cars is not None:
            buscador_coches()
        else:
            st.error("No se pueden mostrar recomendaciones sin datos.")
    elif pagina == "Valoración de Coches":
        valoracion_coches()
    elif pagina == "Debugging de Sistema":
        debug_file_system()

# Ejecutar la aplicación
if __name__ == "__main__":
    main()