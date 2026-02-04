import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Energy Optimisation Dashboard")

st.title("🌱 AI Energy Optimisation Dashboard")
st.write("Upload building energy data to analyse usage and get AI recommendations.")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

    X = df[['hour', 'day', 'occupancy']]
    y = df['energy_kwh']

    model = LinearRegression()
    model.fit(X, y)

    st.subheader("Predict Energy Usage")

    hour = st.slider("Hour", 0, 23, 12)
    day = st.slider("Day (1=Mon, 7=Sun)", 1, 7, 1)
    occupancy = st.selectbox("Occupancy", [1, 2, 3])

    prediction = model.predict([[hour, day, occupancy]])
    st.write(f"Predicted Energy: {prediction[0]:.2f} kWh")

    fig, ax = plt.subplots()
    ax.plot(df['hour'], df['energy_kwh'], marker='o')
    ax.set_xlabel("Hour")
    ax.set_ylabel("Energy (kWh)")
    st.pyplot(fig)

    if prediction[0] > 150:
        st.warning("High energy usage detected. Reduce AC and lighting.")
    else:
        st.success("Energy usage is efficient.")
else:
    st.info("Please upload a CSV file to start.")
