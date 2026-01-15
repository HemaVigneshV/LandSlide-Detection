import pandas as pd
import numpy as np
import streamlit as st
import joblib
from datetime import datetime, timedelta
import altair as alt

# Load trained models
model = joblib.load("landslide_risk_model.pkl")
encoder = joblib.load("landslide_risk_encoder.pkl")

# Generate sample weather data for demonstration purposes (replace with actual data source in production)
def generate_weather_data():
    today = datetime.now()
    data = []
    for i in range(-10, 11):  # Generate data for 10 days before and after today
        date = today + timedelta(days=i)
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'hourly_rainfall': np.random.uniform(0, 100),
            '3_day_cumulative_rainfall': np.random.uniform(0, 600),
            '7_day_cumulative_rainfall': np.random.uniform(0, 1000),
            'Soil Moisture (0-7 cm)': np.random.uniform(10, 50),
            'Soil Moisture (7-28 cm)': np.random.uniform(10, 50)
        })
    return pd.DataFrame(data)

# Predict landslide risk using the trained model
def predict_landslide_risk(weather_data):
    features = [
        'hourly_rainfall',
        '3_day_cumulative_rainfall',
        '7_day_cumulative_rainfall',
        'Soil Moisture (0-7 cm)',
        'Soil Moisture (7-28 cm)'
    ]
    predictions = model.predict(weather_data[features])
    weather_data['landslide_risk'] = encoder.inverse_transform(predictions)
    return weather_data

# Find the next predicted landslide date
def find_next_landslide(weather_data):
    landslide_data = weather_data[(weather_data['landslide_risk'] != 'No Risk') & (weather_data['date'] >= datetime.now().strftime('%Y-%m-%d'))]
    if not landslide_data.empty:
        return landslide_data.iloc[0]['date']
    return "No landslide predicted in the next 10 days."

# Streamlit interface
st.image("landslide.png", use_container_width=True)  # Display an image above the title
st.title("Weather and Landslide Risk Prediction")
st.markdown("Select a day to view detailed weather predictions and landslide risk.")

# Generate and predict weather data
weather_data = generate_weather_data()
weather_data = predict_landslide_risk(weather_data)

# Sidebar for 7-day range date selection
seven_day_data = weather_data[(weather_data['date'] >= datetime.now().strftime('%Y-%m-%d')) &
                               (weather_data['date'] <= (datetime.now() + timedelta(days=6)).strftime('%Y-%m-%d'))]
selected_day = st.sidebar.selectbox("Select a day:", seven_day_data['date'].tolist())

# Display details for the selected day
if selected_day:
    day_data = weather_data[weather_data['date'] == selected_day].iloc[0]
    st.subheader(f"Weather Details for {selected_day}")
    st.write(f"**Hourly Rainfall:** {day_data['hourly_rainfall']:.2f} mm")
    st.write(f"**3-Day Cumulative Rainfall:** {day_data['3_day_cumulative_rainfall']:.2f} mm")
    st.write(f"**7-Day Cumulative Rainfall:** {day_data['7_day_cumulative_rainfall']:.2f} mm")
    st.write(f"**Soil Moisture (0-7 cm):** {day_data['Soil Moisture (0-7 cm)']:.2f}%")
    st.write(f"**Soil Moisture (7-28 cm):** {day_data['Soil Moisture (7-28 cm)']:.2f}%")
    st.write(f"**Landslide Risk Prediction:** {day_data['landslide_risk']}")

# Find and display the next predicted landslide date
next_landslide_date = find_next_landslide(weather_data)
st.subheader(f"Next Predicted Landslide Date: {next_landslide_date}")

# Plot interactive graph for past and future data
st.subheader("Interactive Graph: Weather Data for Past and Future 10 Days")
graph_data = pd.melt(weather_data, id_vars=['date'], value_vars=['hourly_rainfall', '3_day_cumulative_rainfall', '7_day_cumulative_rainfall'], 
                     var_name='Measurement', value_name='Value')
chart = alt.Chart(graph_data).mark_line(point=True).encode(
    x=alt.X('date:T', title='Date'),
    y=alt.Y('Value:Q', title='Values'),
    color='Measurement:N',
    tooltip=['date:T', 'Measurement:N', 'Value:Q']
).properties(
    width=800,
    height=400,
    title="Weather Data for Past and Future 10 Days"
).interactive()
st.altair_chart(chart, use_container_width=True)

# Add styling
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        margin: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
