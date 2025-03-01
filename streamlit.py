import streamlit as st
import pandas as pd
import joblib  
import numpy as np 

# Load the model
@st.cache_resource
def load_model():
    model_path = "modelo_coches_rf.joblib"  
    return joblib.load(model_path)

model = load_model()

# Load data
df_cars = pd.read_csv("modelos_coches.csv")

# Explicitly remove ID column if it exists (using various possible names)
for col in df_cars.columns:
    if 'id' in col.lower():
        df_cars = df_cars.drop(columns=[col])

# Sidebar for car features
st.sidebar.header("Características del coche")

year = st.sidebar.slider("Año del coche", 2000, 2024, 2015)
kms = st.sidebar.number_input("Kilómetros recorridos", min_value=0, max_value=500000, value=50000, step=1000)
power = st.sidebar.number_input("Potencia (CV)", min_value=50, max_value=600, value=150, step=10)
vehicle_age = 2024 - year

fuel = st.sidebar.selectbox("Tipo de combustible", ["Gasolina", "Diésel", "Eléctrico", "Híbrido"], index=0)
shift = st.sidebar.selectbox("Tipo de cambio", ["Manual", "Automático"], index=0)

# Removed "(opcional)" label as requested
make = st.sidebar.text_input("Marca")
model_input = st.sidebar.text_input("Modelo")
version = st.sidebar.text_input("Versión")

# Create dataframe with input values
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

# Apply necessary transformations
input_data['log_kms'] = np.log(input_data['kms'] + 1)
input_data['km_per_year'] = input_data['kms'] / input_data['vehicle_age'] if input_data['vehicle_age'][0] > 0 else input_data['kms']
input_data['power_per_age'] = input_data['power'] / input_data['vehicle_age'] if input_data['vehicle_age'][0] > 0 else input_data['power']

# Price prediction
if st.button("Predecir Precio"):
    predicted_price = model.predict(input_data)[0]
    st.write(f"### Precio estimado: {predicted_price:,.2f} €")
    
    # Filter cars within ±10% of predicted price
    margin = 0.1
    min_price = predicted_price * (1 - margin)
    max_price = predicted_price * (1 + margin)

    
    recommended_cars = df_cars[(df_cars['price'] >= min_price) & (df_cars['price'] <= max_price)]
    
    # Filter cars within ±10% of input kilometers
    km_margin = 0.1
    min_kms = kms * (1 - km_margin)
    max_kms = kms * (1 + km_margin)
    

    recommended_cars = recommended_cars[(recommended_cars['kms'] >= min_kms) & (recommended_cars['kms'] <= max_kms)]
    
    # Case-insensitive filtering that handles NaN values
    if 'fuel' in recommended_cars.columns:
        recommended_cars = recommended_cars[recommended_cars['fuel'].fillna('').str.lower() == fuel.lower()]
    
    if 'shift' in recommended_cars.columns:
        recommended_cars = recommended_cars[recommended_cars['shift'].fillna('').str.lower() == shift.lower()]
    
    # Additional text filters
    if make.strip():
        recommended_cars = recommended_cars[recommended_cars['make'].fillna('').str.contains(make, case=False)]
    
    if model_input.strip():
        recommended_cars = recommended_cars[recommended_cars['model'].fillna('').str.contains(model_input, case=False)]
        
    if version.strip():
        recommended_cars = recommended_cars[recommended_cars['version'].fillna('').str.contains(version, case=False)]
    
    # Sample up to 5 recommendations
    if len(recommended_cars) > 0:
        recommended_cars = recommended_cars.sample(min(5, len(recommended_cars)))
        
        # Format kilometers with dot as thousands separator
        recommended_cars['kms'] = recommended_cars['kms'].apply(lambda x: f"{x:,.0f}".replace(",", "."))
        
        # Format price with comma as decimal separator
        recommended_cars['price'] = recommended_cars['price'].apply(lambda x: f"{x:,.2f}".replace(",", ".").replace(".", ",", 1))
        
        # Format power without decimals
        recommended_cars['power'] = recommended_cars['power'].apply(lambda x: f"{x:.0f}")
        
        st.write("### Recomendaciones basadas en el precio")
        
        # Display recommendations with explicit column selection to ensure no ID is shown
        display_columns = ['make', 'model', 'version', 'power', 'shift', 'fuel', 'kms', 'price']
        # Only include columns that exist in the dataframe
        display_columns = [col for col in display_columns if col in recommended_cars.columns]
        
        st.table(recommended_cars[display_columns])
    else:
        st.write("No se encontraron recomendaciones que coincidan con los criterios.")