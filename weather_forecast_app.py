
# weather_forecast_app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim

st.title("🌦️ نظام التنبؤ بالطقس لليوم التالي باستخدام التعلم الآلي")

city = st.text_input("🧭 أدخل اسم المدينة (بالإنجليزية):", "Damascus")
days = st.slider("عدد الأيام التاريخية للتدريب:", 15, 90, 30)

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

            st.subheader("🔮 التوقعات ليوم الغد:")

            features = ["Temperature", "Humidity", "WindSpeed"]
            predictions = {}

            for feature in features:
                data = df[[feature]].values
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data)

                X, y = [], []
                for i in range(len(scaled_data) - 1):
                    X.append(scaled_data[i])
                    y.append(scaled_data[i + 1])
                X, y = np.array(X), np.array(y).ravel()

                model = RandomForestRegressor(n_estimators=100)
                model.fit(X.reshape(-1, 1), y)

                pred_scaled = model.predict([scaled_data[-1]])
                pred = scaler.inverse_transform([pred_scaled])[0][0]
                predictions[feature] = pred

            st.write(f"🌡️ درجة الحرارة المتوقعة: {predictions['Temperature']:.2f} °C")
            st.write(f"💧 الرطوبة النسبية المتوقعة: {predictions['Humidity']:.2f} %")
            st.write(f"💨 سرعة الرياح المتوقعة: {predictions['WindSpeed']:.2f} كم/ساعة")

            st.subheader("📈 الرسوم البيانية:")
            for feature in features:
                st.line_chart(df.set_index("Date")[feature])
