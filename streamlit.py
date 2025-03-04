import streamlit as st
import pandas as pd
import joblib  
import numpy as np 

# Cargar el modelo
@st.cache_resource
def load_pipeline():
    pipeline_path = "modelo_correcto.joblib"  # Path to your saved pipeline
    return joblib.load(pipeline_path)

pipeline = load_pipeline()

#@st.cache_resource
#def load_price_model():
#    price_model_path = "modelo_precio.joblib"  
#    return joblib.load(price_model_path)

#price_model = load_price_model()

# Cargar datos
df_cars = pd.read_csv("df_modelo_limpio.csv")

# Configuración de página
st.set_page_config(layout="wide")
st.title("🚗 Recomendador de Coches")

# Crear columnas
col1, col2 = st.columns(2)

# Columna de Características
with col1:
    st.header("Recomendador por Características") 

    # Sidebar para características del coche
    st.sidebar.header("Características del coche")

    year = st.sidebar.slider("Año del coche", 2000, 2024, 2015)
    kms = st.sidebar.number_input("Kilómetros recorridos", min_value=0, max_value=500000, value=50000, step=1000)
    power = st.sidebar.number_input("Potencia (CV)", min_value=50, max_value=600, value=150, step=10)
    vehicle_age = 2024 - year

    fuel = st.sidebar.selectbox("Tipo de combustible", ["Gasolina", "Diésel", "Eléctrico", "Híbrido"], index=0)
    shift = st.sidebar.selectbox("Tipo de cambio", ["Manual", "Automático"], index=0)

    make = st.sidebar.text_input("Marca")
    model_input = st.sidebar.text_input("Modelo")
    version = st.sidebar.text_input("Versión")

    # Crear dataframe con los valores introducidos
    input_data = pd.DataFrame({
        'year': [year],
        'kms': [kms],
        'power': [power],
        'vehicle_age': [vehicle_age],
        'fuel': [fuel],
        'shift': [shift],
        'make': [make if make else None],
        'model': [model_input if model_input else None],
        'version': [version if version else None]
    })

    # Add the same engineered features you used during training
    input_data['power_to_age'] = input_data['power'] / (input_data['vehicle_age'] + 1)  
    input_data['kms_per_year'] = input_data['kms'] / (input_data['vehicle_age'] + 1)
    input_data['log_kms'] = np.log1p(input_data['kms'])
    input_data['log_power'] = np.log1p(input_data['power'])
    input_data['log_vehicle_age'] = np.log1p(input_data['vehicle_age'])

    # Add the missing columns with default values
    # Popularity metrics (can set to average values or 0)
    input_data['model_popularity'] = 0
    input_data['make_popularity'] = 0

    # Additional calculated fields
    input_data['price_per_power'] = 0  
    input_data['power_per_kms'] = input_data['power'] / (input_data['kms'] + 1)
    input_data['price_per_year'] = 0  
    input_data['price_range'] = 'medium'  

    # Dealer information (using placeholders)
    input_data['dealer_zip_code'] = '00000'
    input_data['dealer_city'] = 'unknown'
    input_data['province'] = 'unknown'
    input_data['dealer_info'] = 'unknown'
    input_data['dealer_name'] = 'unknown'
    input_data['dealer_address'] = 'unknown'

    # Other features
    input_data['big_city_dealer'] = 0  
    input_data['normalized_version'] = input_data['version']  

    # Predicción de precio
    if st.button("Predecir recomendaciones de coches"):
        predicted_price = pipeline.predict(input_data)[0]
        st.write(f"### Precio estimado: {predicted_price:,.2f} €")
    
    # Filtrar coches dentro de un margen de ±5% del precio predicho 
        margin = 0.05  
        min_price = predicted_price * (1 - margin)
        max_price = predicted_price * (1 + margin)
    
    # Crear una copia profunda para evitar advertencias de modificación
        recommended_cars = df_cars.copy()
    
    # Aplicar filtro de precio
        recommended_cars = recommended_cars[(recommended_cars['price'] >= min_price) & (recommended_cars['price'] <= max_price)]
    
    # Filtrar coches dentro de un margen de ±5% de los kilómetros ingresados 
        km_margin = 0.05  
        min_kms = kms * (1 - km_margin)
        max_kms = kms * (1 + km_margin)
    
        recommended_cars = recommended_cars[(recommended_cars['kms'] >= min_kms) & (recommended_cars['kms'] <= max_kms)]
    
    # Aplicar filtro de potencia con un margen (±10%)
        power_margin = 0.10 
        min_power = power * (1 - power_margin)
        max_power = power * (1 + power_margin)
    
    # Asegurar que la potencia sea numérica antes de filtrar
        if 'power' in recommended_cars.columns:
        # Convertir a numérico si aún no lo es
            recommended_cars['power'] = pd.to_numeric(recommended_cars['power'], errors='coerce')
            recommended_cars = recommended_cars[(recommended_cars['power'] >= min_power) & 
                                          (recommended_cars['power'] <= max_power)]
    
    # Aplicar filtros de tipo de combustible y tipo de cambio con coincidencia insensible a mayúsculas/minúsculas
        if 'fuel' in recommended_cars.columns:
            recommended_cars = recommended_cars[recommended_cars['fuel'].fillna('').str.lower() == fuel.lower()]
    
        if 'shift' in recommended_cars.columns:
            recommended_cars = recommended_cars[recommended_cars['shift'].fillna('').str.lower() == shift.lower()]
    
    # Muestra hasta 5 recomendaciones
        if len(recommended_cars) > 0:
        # Determinar cuántas recomendaciones mostrar (hasta 5)
            num_recommendations = min(5, len(recommended_cars))
        
        # Muestreo sin reemplazo
            recommended_cars = recommended_cars.sample(num_recommendations)
        
        # Restablecer índice para evitar que los valores de índice aparezcan en la tabla
            recommended_cars = recommended_cars.reset_index(drop=True)
        
        # Formatear datos de visualización
            formatted_cars = recommended_cars.copy()
        
        # Cambiar nombres de columnas en inglés a español
            column_mapping = {
                'make': 'marca',
                'model': 'modelo',
                'version': 'versión',
                'power': 'potencia',
                'shift': 'cambio',
                'fuel': 'combustible',
                'kms': 'kilómetros',
                'price': 'precio'
            }
        
        # Renombrar columnas
            formatted_cars = formatted_cars.rename(columns=column_mapping)
        
        # Formatear kilómetros con punto como separador de miles
            formatted_cars['kilómetros'] = formatted_cars['kilómetros'].apply(lambda x: f"{x:,.0f}".replace(",", "."))
        
        # Formatear precio con coma como separador decimal
            formatted_cars['precio'] = formatted_cars['precio'].apply(lambda x: f"{x:,.2f}".replace(",", ".").replace(".", ",", 1))
        
        # Formatear potencia sin decimales
            formatted_cars['potencia'] = formatted_cars['potencia'].apply(lambda x: f"{x:.0f}")
        
            st.write("### Recomendaciones según las características introducidas:")
        
        # Definir columnas de visualización - solo columnas que queremos mostrar
            display_columns = [
                'marca', 'modelo', 'versión', 'potencia', 'cambio', 'combustible', 
                'kilómetros', 'precio'
            ]
        
        # Solo incluir columnas que existen en el dataframe
            display_columns = [col for col in display_columns if col in formatted_cars.columns]
        
        # Usar st.dataframe en lugar de st.table para tener más control
            st.dataframe(
                formatted_cars[display_columns],
            # Ocultar el índice
                hide_index=True
            )
        
        # Mostrar información detallada del vendedor para cada recomendación
            st.write("### Información detallada de los vendedores")
            for i, car in enumerate(recommended_cars.itertuples(), 1):
                st.write(f"**Opción {i}: {car.make} {car.model}**")
                st.write(f"📍 **Ubicación:** {getattr(car, 'dealer_city', 'N/A')}, {getattr(car, 'province', 'N/A')}")
                st.write(f"🏬 **Concesionario:** {getattr(car, 'dealer_name', 'N/A')}")
                st.write(f"🗺️ **Dirección:** {getattr(car, 'dealer_address', 'N/A')}")
                if hasattr(car, 'dealer_zip_code'):
                    st.write(f"📮 **Código Postal:** {car.dealer_zip_code}")
                st.write("---")
        else:
            st.write("No se encontraron recomendaciones que coincidan con los criterios.")
            
# Columna de Precio
with col2:
    st.header("Predicción de Precio")
    
    # Inputs para predicción de precio
    price_year = st.slider("Año del coche", 2000, 2024, 2015)
    price_kms = st.number_input("Kilómetros", min_value=0, max_value=500000, value=50000)
    price_power = st.number_input("Potencia (CV)", min_value=50, max_value=600, value=150)
    price_fuel = st.selectbox("Tipo de combustible", ["Gasolina", "Diésel", "Eléctrico", "Híbrido"])
    price_shift = st.selectbox("Tipo de cambio", ["Manual", "Automático"])

    # Preparar datos para modelo de precio
    price_input_data = pd.DataFrame({
        'year': [price_year],
        'kms': [price_kms],
        'power': [price_power],
        'fuel': [price_fuel],
        'shift': [price_shift]
    })

    # Botón de predicción de precio
    if st.button("Predecir Precio"):
        # Cargar modelo de precio
        price_model = load_price_model()
        
        # Predecir precio
        predicted_price = price_model.predict(price_input_data)[0]
        
        # Mostrar precio predicho
        st.metric("Precio Estimado", f"{predicted_price:,.2f} €")
        
        # Opcional: Añadir información adicional
        st.write("### Detalles de la Predicción")
        st.write(f"- Año: {price_year}")
        st.write(f"- Kilómetros: {price_kms:,}")
        st.write(f"- Potencia: {price_power} CV")
        st.write(f"- Combustible: {price_fuel}")
        st.write(f"- Tipo de Cambio: {price_shift}")

# Estilos CSS
st.markdown("""
<style>
.stColumn {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)