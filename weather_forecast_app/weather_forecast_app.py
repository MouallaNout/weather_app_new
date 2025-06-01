
# weather_forecast_app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim

st.title("🌦️ نظام التنبؤ بالطقس لعدة أيام باستخدام LSTM")

city = st.text_input("🧭 أدخل اسم المدينة (بالإنجليزية):", "Damascus")
days = st.slider("عدد الأيام التاريخية للتدريب:", 15, 90, 30)
forecast_days = st.slider("عدد الأيام المستقبلية للتنبؤ:", 1, 5, 3)

if st.button("ابدأ التنبؤ"):
    with st.spinner("🔍 جارٍ تحديد موقع المدينة..."):
        try:
            geolocator = Nominatim(user_agent="weather_forecast_app")
            location = geolocator.geocode(city)
            if location is None:
                st.error("❌ لم يتم العثور على موقع المدينة.")
                st.stop()
            lat, lon = location.latitude, location.longitude
            st.success(f"📍 إحداثيات {city}: {lat:.2f}, {lon:.2f}")
        except Exception as e:
            st.error("❌ خطأ أثناء تحديد الإحداثيات.")
            st.stop()

    with st.spinner("🌐 جارٍ جلب بيانات الطقس التاريخية..."):
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days)
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
            f"&daily=temperature_2m_max,relative_humidity_2m_max,windspeed_10m_max&timezone=auto"
        )

        response = requests.get(url)
        if response.status_code != 200:
            st.error("❌ فشل في جلب البيانات المناخية.")
        else:
            data_json = response.json()
            df = pd.DataFrame({
                "Date": pd.to_datetime(data_json["daily"]["time"]),
                "Temperature": data_json["daily"]["temperature_2m_max"],
                "Humidity": data_json["daily"]["relative_humidity_2m_max"],
                "WindSpeed": data_json["daily"]["windspeed_10m_max"]
            })
            st.write("✅ البيانات التاريخية:")
            st.dataframe(df)

            forecasts = {}
            features = ["Temperature", "Humidity", "WindSpeed"]
            for feature in features:
                data = df[[feature]].values
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(data)

                X, y = [], []
                for i in range(len(scaled) - forecast_days):
                    X.append(scaled[i:i+forecast_days])
                    y.append(scaled[i+forecast_days])
                X, y = np.array(X), np.array(y)

                model = Sequential()
                model.add(LSTM(64, input_shape=(forecast_days, 1)))
                model.add(Dense(1))
                model.compile(optimizer="adam", loss="mse")
                model.fit(X, y, epochs=100, verbose=0)

                forecast_input = scaled[-forecast_days:].reshape(1, forecast_days, 1)
                pred_scaled = model.predict(forecast_input)
                pred = scaler.inverse_transform(pred_scaled)[0][0]
                forecasts[feature] = pred

            st.subheader("🔮 التوقعات ليوم الغد:")
            st.write(f"🌡️ درجة الحرارة المتوقعة: {forecasts['Temperature']:.2f} °C")
            st.write(f"💧 الرطوبة النسبية المتوقعة: {forecasts['Humidity']:.2f} %")
            st.write(f"💨 سرعة الرياح المتوقعة: {forecasts['WindSpeed']:.2f} كم/ساعة")

            st.subheader("📈 الرسوم البيانية:")
            for feature in features:
                st.line_chart(df.set_index("Date")[feature])
