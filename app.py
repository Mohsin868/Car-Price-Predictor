# app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 🎯 Set page configuration
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Predictor (Multi-Feature)")

# 📥 Load data
@st.cache_data
def load_data():
    df = pd.read_csv("CarPrice_Assignment.csv")
    return df

df = load_data()

# 👀 Preview data
if st.checkbox("Show Raw Data"):
    st.dataframe(df[['price', 'horsepower', 'enginesize', 'carwidth', 'curbweight', 'highwaympg']].head())

# 🧪 Select Features and Target
features = ['horsepower', 'enginesize', 'carwidth', 'curbweight', 'highwaympg']
X = df[features]
y = df['price']

# 📚 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🏗️ Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 🧠 Sidebar: User Inputs
st.sidebar.header("🔧 Input Features")

def get_user_input():
    horsepower = st.sidebar.slider("Horsepower", 40, 300, 100)
    enginesize = st.sidebar.slider("Engine Size", 60, 325, 130)
    carwidth = st.sidebar.slider("Car Width", 60, 75, 65)
    curbweight = st.sidebar.slider("Curb Weight", 1500, 4000, 2500)
    highwaympg = st.sidebar.slider("Highway MPG", 15, 55, 30)

    data = {
        'horsepower': horsepower,
        'enginesize': enginesize,
        'carwidth': carwidth,
        'curbweight': curbweight,
        'highwaympg': highwaympg
    }

    return pd.DataFrame(data, index=[0])

user_input = get_user_input()

# 📈 Make Prediction
prediction = model.predict(user_input)[0]
st.metric("💰 Predicted Price", f"${prediction:,.2f}")

# 📊 Evaluation Metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Evaluation")
st.write(f"**Mean Squared Error:** {mse:,.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# 📉 Optional Plot: Actual vs Predicted
if st.checkbox("Show Actual vs Predicted Plot"):
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Actual vs Predicted Car Prices")
    st.pyplot(fig)
