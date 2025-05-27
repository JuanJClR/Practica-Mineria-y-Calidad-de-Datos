import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Cargar modelo
filename = 'modelo_rf.pkl'
with open(filename, 'rb') as f:
    best_rf, labelencoder, variables, min_max_scaler = pickle.load(f)

st.title('Predicción de Calidad del Aire')

# Captura de entradas numéricas
temperature = st.slider('Temperatura (°C)', min_value=-20.0, max_value=50.0, value=25.0, step=0.1)
humidity = st.slider('Humedad (%)', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
pm10 = st.slider('PM10 (µg/m³)', min_value=0.0, max_value=600.0, value=50.0, step=0.1)
no2 = st.slider('NO2 (ppb)', min_value=0.0, max_value=500.0, value=30.0, step=0.1)
so2 = st.slider('SO2 (ppb)', min_value=0.0, max_value=200.0, value=10.0, step=0.1)
co = st.slider('CO (ppm)', min_value=0.0, max_value=50.0, value=1.0, step=0.01)
density = st.slider('Densidad poblacional (hab/km²)', min_value=0, max_value=50000, value=550, step=10)


# Crear el DataFrame
data = pd.DataFrame([[
    temperature, humidity, pm10, no2, so2, co, density
]], columns=[
    "Temperature", "Humidity", "PM10", "NO2", "SO2", "CO", "Population_Density"
])

# Manejo de columna "Unnamed: 0"
if 'Unnamed: 0' in variables and 'Unnamed: 0' not in data.columns:
    data['Unnamed: 0'] = 0

# Preparación
data_preparada = data.copy()
for col in variables:
    if col not in data_preparada.columns:
        data_preparada[col] = 0
data_preparada = data_preparada[variables]

# Predicción
Y_fut = model_rf.predict(data_preparada)
data['Air Quality'] = labelencoder.inverse_transform(Y_fut)

# Eliminar columna innecesaria si existe
if 'Unnamed: 0' in data.columns:
    data.drop(columns=['Unnamed: 0'], inplace=True)

# Mostrar resultados
st.subheader("Resultado de la predicción")
st.write(data)
