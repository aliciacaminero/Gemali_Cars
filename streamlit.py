import streamlit as st
import pandas as pd
import joblib  
import numpy as np 
import os

# Cargar modelos
@st.cache_resource
def load_pipeline():
    pipeline_path = "modelo_correcto.joblib"
    return joblib.load(pipeline_path)

@st.cache_resource
def load_price_model():
    price_model_path = "model_with_encoders.joblib"  
    return joblib.load(price_model_path)

# Cargar datos
df_cars = pd.read_csv("df_modelo_limpio.csv")

# Configuración de página
st.set_page_config(layout="wide", page_title="AutoMatch", page_icon="🚗")

# Borrar el estado de la aplicación
def reset_state():
    for key in st.session_state.keys():
        del st.session_state[key]

# Función para la página de inicio
def pagina_inicio():
    st.title("AutoMatch: Tu Asistente Inteligente de Coches de Segunda Mano")

    st.markdown("""
    ## Bienvenido a AutoMatch 🚗

    ### ¿Qué hacemos?
    AutoMatch es una herramienta inteligente diseñada para ayudarte a encontrar el coche de segunda mano perfecto en el mercado español. Utilizamos análisis de datos avanzados para ofrecerte recomendaciones personalizadas y valoraciones precisas.

    ### Nuestros Servicios
    - **Buscador Inteligente de Coches**: Encuentra el vehículo de ocasión que mejor se adapta a tus necesidades.
    - **Valoración Precisa de Coches**: Estima el valor real de un coche en el mercado español.

    ### Cómo Funciona
    1. **Personalización**: Introduce las características que buscas en un coche.
    2. **Análisis Inteligente**: Procesamos datos de coches de segunda mano utilizando nuestros modelos entrenados para ofrecerte recomendaciones personalizadas.
    3. **Resultados Precisos**: Obtén recomendaciones y valoraciones exactas basadas en datos reales del mercado español.

    ### Ventajas
    - 🎯 Recibe recomendaciones 100% personalizadas a tus necesidades
    - 💡 Valoraciones basadas en datos reales del mercado español
    - 🔍 Búsqueda inteligente de coches de ocasión
    """)

# Función para añadir las características del coche en el sidebar
def sidebar_car_characteristics(mode='search'):
    st.sidebar.header("Características del Coche")
    
    # Si es modo de búsqueda
    if mode == 'search':
        year = st.sidebar.slider("Año del coche", 2000, 2024, 2015, key="characteristics_year")
        kms = st.sidebar.number_input("Kilómetros recorridos", min_value=0, max_value=500000, value=50000, step=1000, key="characteristics_kms")
        power = st.sidebar.number_input("Potencia (CV)", min_value=50, max_value=600, value=150, step=10, key="characteristics_power")
        fuel = st.sidebar.selectbox("Tipo de combustible", ["Gasolina", "Diésel", "Eléctrico", "Híbrido"], index=0, key="characteristics_fuel")
        shift = st.sidebar.selectbox("Tipo de cambio", ["Manual", "Automático"], index=0, key="characteristics_shift")
        make = st.sidebar.text_input("Marca", key="characteristics_make")
        model_input = st.sidebar.text_input("Modelo", key="characteristics_model")
        version = st.sidebar.text_input("Versión", key="characteristics_version")
        
        return {
            'year': year, 
            'kms': kms, 
            'power': power, 
            'fuel': fuel, 
            'shift': shift, 
            'make': make, 
            'model': model_input, 
            'version': version
        }
    
    # Si es modo de valoración
    elif mode == 'valuation':
        price_year = st.sidebar.slider("Año del coche", 2000, 2024, 2015, key="price_prediction_year")
        price_kms = st.sidebar.number_input("Kilómetros", min_value=0, max_value=500000, value=50000, key="price_prediction_kms")
        price_power = st.sidebar.number_input("Potencia (CV)", min_value=50, max_value=600, value=150, key="price_prediction_power")
        price_fuel = st.sidebar.selectbox("Tipo de combustible", ["Gasolina", "Diésel", "Eléctrico", "Híbrido"], key="price_prediction_fuel")
        price_shift = st.sidebar.selectbox("Tipo de cambio", ["Manual", "Automático"], key="price_prediction_shift")
        price_make = st.sidebar.text_input("Marca", key="price_prediction_make")
        price_model = st.sidebar.text_input("Modelo", key="price_prediction_model")
        
        return {
            'year': price_year, 
            'kms': price_kms, 
            'power': price_power, 
            'fuel': price_fuel, 
            'shift': price_shift, 
            'make': price_make, 
            'model': price_model
        }

# Función para el Buscador Inteligente de Coches
def buscador_coches():
    # Estado inicial de la aplicación de búsqueda
    if 'search_submitted' not in st.session_state:
        st.session_state.search_submitted = False

    # Título principal
    st.title("Buscador Inteligente de Coches de Segunda Mano")

    # Recoger características desde el sidebar
    car_characteristics = sidebar_car_characteristics(mode='search')

    # Columna de resultados
    col2 = st.container()

    with col2:
        # Encabezado de resultados
        st.header("Resultados de Búsqueda")
        
        # Botón de búsqueda
        if st.button("Buscar Coches"):
            st.session_state.search_submitted = True

        # Lógica de búsqueda y resultados
        if st.session_state.search_submitted:
            # Extraer características
            year = car_characteristics['year']
            kms = car_characteristics['kms']
            power = car_characteristics['power']
            vehicle_age = 2024 - year

            # Crear dataframe con los valores introducidos
            input_data = pd.DataFrame({
                'year': [year],
                'kms': [kms],
                'power': [power],
                'vehicle_age': [vehicle_age],
                'fuel': [car_characteristics['fuel']],
                'shift': [car_characteristics['shift']],
                'make': [car_characteristics['make'] if car_characteristics['make'] else None],
                'model': [car_characteristics['model'] if car_characteristics['model'] else None],
                'version': [car_characteristics['version'] if car_characteristics['version'] else None]
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

            # Cargar pipeline
            pipeline = load_pipeline()

            # Predicción de precio y recomendaciones
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
                recommended_cars = recommended_cars[recommended_cars['fuel'].fillna('').str.lower() == car_characteristics['fuel'].lower()]
        
            if 'shift' in recommended_cars.columns:
                recommended_cars = recommended_cars[recommended_cars['shift'].fillna('').str.lower() == car_characteristics['shift'].lower()]
        
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

# Función para Valoración de Coches
def valoracion_coches():
    # Título principal
    st.title("Valoración de Coches de Segunda Mano")

    # Recoger características desde el sidebar
    car_characteristics = sidebar_car_characteristics(mode='valuation')

    # Contenedor para resultados
    result_container = st.container()

    with result_container:
        st.header("Resultado de la Valoración")
        
        # Botón de valoración
        if st.button("Valorar Coche"):
            # Validar que se hayan ingresado los campos obligatorios
            required_fields = ['make', 'model', 'fuel', 'year', 'kms', 'power', 'shift']
            missing_fields = [field for field in required_fields if not car_characteristics.get(field)]
            
            if missing_fields:
                st.error(f"Por favor, complete los siguientes campos: {', '.join(missing_fields)}")
            else:
                try:
                    # Preparar datos con columnas separadas
                    price_input_data = pd.DataFrame({
                        'make': [str(car_characteristics['make']).strip()],
                        'model': [str(car_characteristics['model']).strip()],
                        'fuel': [str(car_characteristics['fuel']).strip()],
                        'year': [int(car_characteristics['year'])],
                        'kms': [float(car_characteristics['kms'])],
                        'power': [float(car_characteristics['power'])],
                        'shift_automatic': [1 if car_characteristics['shift'] == 'Automático' else 0],
                        'shift_manual': [1 if car_characteristics['shift'] == 'Manual' else 0]
                    })

                    # Depuración: imprimir las columnas del DataFrame de entrada
                    st.write("Columnas de entrada:")
                    st.write(price_input_data.columns)
                    st.write("Datos de entrada:")
                    st.write(price_input_data)

                    # Cargar modelo de precio
                    price_model = load_price_model()
                    
                    # Predecir precio
                    predicted_price = price_model.predict(price_input_data)[0]
                    
                    # Mostrar precio predicho
                    st.metric("Precio Estimado", f"{predicted_price:,.2f} €")
                    
                    # Información adicional
                    st.write("### Detalles de la Valoración")
                    st.write(f"- Año: {car_characteristics['year']}")
                    st.write(f"- Kilómetros: {car_characteristics['kms']:,}")
                    st.write(f"- Potencia: {car_characteristics['power']} CV")
                    st.write(f"- Combustible: {car_characteristics['fuel']}")
                    st.write(f"- Tipo de Cambio: {car_characteristics['shift']}")
                    st.write(f"- Marca: {car_characteristics['make']}")
                    st.write(f"- Modelo: {car_characteristics['model']}")
                
                except Exception as e:
                    st.error(f"Error al valorar el coche: {str(e)}")
                    st.error("Por favor, verifique que todos los campos estén completados correctamente.")
                    # Imprimir el error completo para depuración
                    import traceback
                    st.error(traceback.format_exc())
            
# Función principal para configurar la navegación
def main():
    # Sidebar para navegación
    menu = st.sidebar.radio(
        "Navega en AutoMatch", 
        ["Inicio", "Buscador de Coches", "Valoración de Coches"]
    )

    # Renderizar la página seleccionada
    if menu == "Inicio":
        pagina_inicio()
    elif menu == "Buscador de Coches":
        buscador_coches()
    elif menu == "Valoración de Coches":
        valoracion_coches()

# Ejecutar la aplicación principal
main()

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