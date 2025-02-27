import streamlit as st
import pandas as pd
#import joblib  # Para cargar el modelo
#import pickle  # Alternativa si el modelo está en formato pkl

# Cargar el modelo
#@st.cache_resource
#def load_model():
#    model_path = "car_price_model.pkl"  # Ajustar según el formato real
#    try:
#        with open(model_path, 'rb') as file:
#            model = pickle.load(file)
#    except:
#        model = joblib.load(model_path)
#    return model

#model = load_model()

# Sidebar para entrada de características
st.sidebar.header("Características del coche")

year = st.sidebar.slider("Año del coche", 2000, 2024, 2015)
kms = st.sidebar.number_input("Kilómetros recorridos", min_value=0, max_value=500000, value=50000, step=1000)
power = st.sidebar.number_input("Potencia (CV)", min_value=50, max_value=600, value=150, step=10)
vehicle_age = 2024 - year

fuel = st.sidebar.selectbox("Tipo de combustible", ["Gasolina", "Diésel", "Eléctrico", "Híbrido"], index=0)
shift = st.sidebar.selectbox("Tipo de cambio", ["Manual", "Automático"], index=0)

make = st.sidebar.text_input("Marca (opcional)")
model_input = st.sidebar.text_input("Modelo (opcional)")

# Crear dataframe con los valores introducidos
input_data = pd.DataFrame({
    'year': [year],
    'kms': [kms],
    'power': [power],
    'vehicle_age': [vehicle_age],
    'fuel': [fuel],
    'shift': [shift],
    'make': [make if make else None],
    'model': [model_input if model_input else None]
})

st.write("### Coche seleccionado")
st.write(input_data)

# Predicción de precio
if st.button("Predecir Precio"):
    predicted_price = model.predict(input_data)[0]
    st.write(f"### Precio estimado: {predicted_price:,.2f} €")
    
    # Buscar coches similares (Simulación)
    st.write("### Recomendaciones basadas en el precio")
    recommended_cars = pd.DataFrame({
        'Marca': ["Toyota", "BMW", "Ford"],
        'Modelo': ["Corolla", "Serie 3", "Focus"],
        'Año': [2018, 2020, 2017],
        'Kilómetros': [45000, 60000, 70000],
        'Precio': [predicted_price * 0.95, predicted_price, predicted_price * 1.05]
    })
    
    st.table(recommended_cars)
