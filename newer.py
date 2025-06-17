# app.py - Advanced Car Price Predictor (Without separate pickle file)

import streamlit as st
import pandas as pd
import numpy as np
import os # For checking if the data file exists

# Import necessary scikit-learn components for training
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor # Using RandomForestRegressor as it was the best in your notebook
from sklearn.metrics import mean_absolute_error, r2_score

# Suppress warnings if any from libraries
import warnings
warnings.filterwarnings('ignore')

# 🎯 Set page configuration
st.set_page_config(page_title="Advanced Car Price Predictor (In-App Training)", layout="centered")
st.title("🚗 Advanced Car Price Predictor")
st.markdown("**(Model trained directly in-app and cached)**")
st.markdown("Predict the price of a car based on its specifications. This model uses advanced preprocessing and a machine learning algorithm.")

# --- File Paths ---
DATA_FILE = "CarPrice_Assignment.csv"

# --- 1. Data Loading (Raw Data) ---
@st.cache_data
def load_raw_data(file_path):
    """Loads the raw CSV data."""
    if not os.path.exists(file_path):
        st.error(f"Error: Data file '{file_path}' not found. Please ensure it's in the same directory.")
        return None
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

raw_df = load_raw_data(DATA_FILE)

if raw_df is None:
    st.stop() # Stop the app if data can't be loaded

# --- 2. Model Training and Preprocessing Pipeline (CACHED) ---
@st.cache_resource # This is CRUCIAL: caches the trained model pipeline
def train_model_pipeline(data_df):
    """
    Performs data preprocessing and trains the machine learning model.
    This function will only run once per app session (or when data/code changes) due to caching.
    """
    df = data_df.copy() # Work on a copy to avoid modifying original DataFrame

    st.write("🔄 **Training Model (this might take a moment if it's the first run)...**")

    # Drop 'car_ID' if it exists (assuming it's just an index)
    if 'car_ID' in df.columns:
        df = df.drop('car_ID', axis=1)

    # Feature Engineering: Extract car make from 'CarName'
    if 'CarName' in df.columns:
        df['CarMake'] = df['CarName'].apply(lambda x: x.split(' ')[0].replace('-', ' ').strip().lower())
        df['CarMake'] = df['CarMake'].replace({
            'vw': 'volkswagen', 'vokswagen': 'volkswagen',
            'porsche': 'porcshe',
            'maxda': 'mazda',
            'toyouta': 'toyota',
            'Nissan': 'nissan'
        })
        df = df.drop('CarName', axis=1)

    # Feature Engineering: Add Power-to-weight ratio
    if 'horsepower' in df.columns and 'curbweight' in df.columns:
        df['PowerToWeightRatio'] = df['horsepower'] / df['curbweight']
    
    # Feature Engineering: Add Average MPG
    if 'citympg' in df.columns and 'highwaympg' in df.columns:
        df['AvgMPG'] = (df['citympg'] + df['highwaympg']) / 2

    # Separate target variable (price) from features
    if 'price' not in df.columns:
        raise KeyError("Missing 'price' column in the dataset.")

    X = df.drop('price', axis=1)
    y = df['price']

    # Identify numerical and categorical features AFTER feature engineering
    # These lists will be used to correctly set up sliders/selectboxes in the UI
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    # 'symboling' is now kept in numerical_features as requested.
    # No explicit removal here.

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor_pipeline = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Create the full model pipeline (preprocessor + regressor)
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_pipeline),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    # Split data for evaluation (optional, but good for reporting)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Evaluate the model for reporting metrics later
    y_pred = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.success("✅ Model training complete!")
    # Return all necessary components: model, metrics, and feature lists
    return model_pipeline, mae, r2, numerical_features, categorical_features

# Call the cached training function
# This line now directly receives the features lists from the training function
model_pipeline, mae_val, r2_val, numerical_features_used, categorical_features_used = train_model_pipeline(raw_df)

# --- 3. Sidebar: User Inputs for Prediction ---
st.sidebar.header("🔧 Configure Car Specifications")

# Function to get user inputs based on identified features
def get_user_input():
    input_data = {}

    st.sidebar.subheader("Numerical Features:")

    # Define a preferred order for some common features
    preferred_order = [
        'horsepower', 'enginesize', 'carwidth', 'curbweight',
        'citympg', 'highwaympg', 'symboling' # Place symboling after MPG
    ]
    
    # Collect inputs for preferred order features first
    for col in preferred_order:
        if col in numerical_features_used:
            # Skip engineered features from direct user input; they'll be calculated from base features
            if col in ['PowerToWeightRatio', 'AvgMPG']:
                continue
            
            if col in raw_df.columns: # Ensure the column exists in raw_df
                col_data = raw_df[col].dropna()
                if pd.api.types.is_numeric_dtype(col_data) and not col_data.empty:
                    min_val = float(col_data.min())
                    max_val = float(col_data.max())
                    mean_val = float(col_data.mean())
                    
                    if col_data.dtype in [np.int64, np.int32]:
                        step_val = 1.0
                    else:
                        step_val = (max_val - min_val) / 100.0
                        if step_val < 0.01: step_val = 0.01
                        elif step_val > 1.0: step_val = 1.0
                        step_val = round(step_val, 2)
                        if step_val == 0.0: step_val = 0.1

                    # Format min/max for display, handling decimals appropriately
                    min_display = f"{min_val:.0f}" if step_val >= 1.0 else f"{min_val:.2f}"
                    max_display = f"{max_val:.0f}" if step_val >= 1.0 else f"{max_val:.2f}"
                    
                    input_data[col] = st.sidebar.number_input(
                        f"{col.replace('_', ' ').title()} (Min: {min_display}, Max: {max_display})", # Display min/max here
                        min_value=min_val, 
                        max_value=max_val, 
                        value=mean_val, 
                        step=step_val, 
                        format="%.2f" if step_val < 1.0 else "%.0f"
                    )
                else:
                    st.sidebar.warning(f"Could not create number input for '{col}' (not purely numerical or empty). Defaulting to 0.")
                    input_data[col] = 0.0
            else:
                st.sidebar.warning(f"Numerical feature '{col}' not found in raw data. Defaulting to 0.")
                input_data[col] = 0.0
    
    # Collect inputs for any other numerical features not in preferred_order
    for col in numerical_features_used:
        if col not in preferred_order: # Only process if not already handled
            if col in ['PowerToWeightRatio', 'AvgMPG']:
                continue # Skip engineered features

            if col in raw_df.columns:
                col_data = raw_df[col].dropna()
                if pd.api.types.is_numeric_dtype(col_data) and not col_data.empty:
                    min_val = float(col_data.min())
                    max_val = float(col_data.max())
                    mean_val = float(col_data.mean())
                    
                    if col_data.dtype in [np.int64, np.int32]:
                        step_val = 1.0
                    else:
                        step_val = (max_val - min_val) / 100.0
                        if step_val < 0.01: step_val = 0.01
                        elif step_val > 1.0: step_val = 1.0
                        step_val = round(step_val, 2)
                        if step_val == 0.0: step_val = 0.1

                    min_display = f"{min_val:.0f}" if step_val >= 1.0 else f"{min_val:.2f}"
                    max_display = f"{max_val:.0f}" if step_val >= 1.0 else f"{max_val:.2f}"

                    input_data[col] = st.sidebar.number_input(
                        f"{col.replace('_', ' ').title()} (Min: {min_display}, Max: {max_display})", # Display min/max here
                        min_value=min_val, 
                        max_value=max_val, 
                        value=mean_val, 
                        step=step_val, 
                        format="%.2f" if step_val < 1.0 else "%.0f"
                    )
                else:
                    st.sidebar.warning(f"Could not create number input for '{col}' (not purely numerical or empty). Defaulting to 0.")
                    input_data[col] = 0.0
            else:
                st.sidebar.warning(f"Numerical feature '{col}' not found in raw data. Defaulting to 0.")
                input_data[col] = 0.0


    st.sidebar.subheader("Categorical Features:")
    for col in categorical_features_used:
        if col == 'CarMake':
            # Dynamically get unique makes from raw data and apply initial transformation
            unique_makes = raw_df['CarName'].apply(lambda x: x.split(' ')[0].replace('-', ' ').strip().lower()).unique().tolist()
            unique_makes.sort()
            input_data[col] = st.sidebar.selectbox(f"Car Make", unique_makes)
        elif col in raw_df.columns:
            unique_values = raw_df[col].unique().tolist()
            unique_values.sort()
            input_data[col] = st.sidebar.selectbox(f"{col.replace('_', ' ').title()}", unique_values)
        else:
            st.sidebar.warning(f"Categorical feature '{col}' not found in raw data. Defaulting to 'unknown'.")
            input_data[col] = "unknown"

    # Create a DataFrame from collected inputs
    user_df = pd.DataFrame([input_data])
    
    # Re-create engineered features here, mirroring the training pipeline's DataPreprocessing::_feature_engineering
    if 'horsepower' in user_df.columns and 'curbweight' in user_df.columns:
        user_df['PowerToWeightRatio'] = user_df['horsepower'] / user_df['curbweight']
    if 'citympg' in user_df.columns and 'highwaympg' in user_df.columns:
        user_df['AvgMPG'] = (user_df['citympg'] + user_df['highwaympg']) / 2
        
    return user_df

user_input_df = get_user_input()

st.subheader("Your Input Specifications:")
st.dataframe(user_input_df)

# --- 4. Make Prediction ---
if st.button("Predict Car Price", type="primary"):
    try:
        prediction = model_pipeline.predict(user_input_df)[0]
        st.success(f"💰 Predicted Car Price: **${prediction:,.2f}**")
        st.balloons()
    except Exception as e:
        st.error(f"Error making prediction. Please check inputs or model.")
        st.info(f"Detailed Error: {e}")
        st.info("Common issues: missing input features, incorrect feature names/types, or model expecting engineered features that were not created in the input DataFrame.")


# --- 5. Model Evaluation Display ---
st.subheader("📊 Model Performance Overview")
st.write(f"The trained model is a **{model_pipeline.named_steps['regressor'].__class__.__name__.replace('Regressor', '')}**.")
st.write(f"- **Mean Absolute Error (MAE):** ${mae_val:,.2f}")
st.write(f"- **R-squared (R² Score):** {r2_val:.4f}")
st.info("These metrics represent the model's performance on unseen data during its in-app training and evaluation phase.")

st.markdown("---")

# --- 6. Optional: Show Raw Data ---
if st.checkbox("Show Raw Data"):
    st.subheader("📋 Raw Data Preview")
    st.dataframe(raw_df.head(10), use_container_width=True)
    