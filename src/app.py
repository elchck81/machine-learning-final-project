import streamlit as st
import pandas as pd
import pickle

def load_model():
    with open ("../models/bike_sharing.pkl", "rb") as file:
        artifact = pickle.load(file)

    return artifact

st.title("Predicción de uso de bicicletas compartidas")
st.write(
    """
    Esta aplicación emplea un **Random Forest** para predecir la cantidad de bicletas compartidas
    en la ciudad de Washington DC en función del clima, hora del día, temporada y otras variables temporales.
    """
    )

season_map = {"Invierno":0, "Otoño":1, "Primavera":2, "Verano":3}
weathersit_map = {
    "Despejado/pocas nubes": 1,
    "Niebla/nubes":2,
    "Lluvia ligera / nieve ligera":3,
    "Lluvia fuerte / nieve / tormenta":4,

}

col1, col2 = st.columns(2)
with col1:
    season_label = st.selectbox("Estacióndel año", list(season_map.keys()))
    hr = st.slider("Hora del día",0,23,8)
    temp_c = st.number_input("Temperatura (°C)", -10.0, 40.0, 20.0)

with col2:
    hum_pct = st.slider("Humedad relativa(%)", 0, 100, 60)
    windspeed_kmh = st.slider("Velocidad del viento (km/h)", 0.0, 70.0, 10.0)
    weathersit_label = st.selectbox("Condición climática", list(weathersit_map.keys()))

holiday = st.checkbox("¿Es feriado?", value=False)
workingday = st.checkbox("¿Es día laboral? (ni fin de semana ni feriado)", value=True)

weekday = st.selectbox(
    "Día de la semana",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"][x],
    )

input_dict= {
    "season": season_map[season_label],
    "hr": hr,
    "temp": temp_c,
    "hum": hum_pct,
    "windspeed": windspeed_kmh,
    "weathersit": weathersit_map[weathersit_label],
    "holiday": int(holiday),
    "workingday": int(workingday),
    "weekday": weekday,



}

input_df = pd.DataFrame([input_dict])

st.write("### Datos de entrada al modelo")
st.dataframe(input_df)

artifact = load_model()
if st.button("Predecir cantidad de bicicletas"):
    pred = artifact['model'].predict(input_df)[0]
    st.metric("Bicicletas estimadas", value=int(round(pred)))


