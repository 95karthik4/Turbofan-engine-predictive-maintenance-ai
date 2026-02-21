import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# 1. Page Config
st.set_page_config(page_title="Jet Engine AI", layout="wide")

# 2. COLOR FIX (I added this to your code so text is never invisible)
st.markdown("""
<style>
    .stApp { background-color: #ffffff !important; }
    p, h1, h2, h3, div, span, label { color: #000000 !important; }
    div[data-testid="stMetricValue"] { color: #0066cc !important; }
    section[data-testid="stSidebar"] { background-color: #f0f2f6 !important; }
    div[data-baseweb="input"] > div {
        background-color: white !important;
        color: black !important;
        -webkit-text-fill-color: black !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Title
st.title("✈️ Predictive Maintenance Dashboard")

# --- Load Assets ---
@st.cache_resource
def load_assets():
    try:
        # Load Model
        model = tf.keras.models.load_model('nasa_rul_model.h5')

        # Load Data
        col_names = ['unit_number', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + ['s_{}'.format(i+1) for i in range(21)]
        data = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=col_names)
        return model, data
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

model, test_df = load_assets()

if model is not None:

    # --- Sidebar ---
    st.sidebar.header("Controls")
    # Using number_input is safer than selectbox for debugging
    selected_unit = int(st.sidebar.number_input("Enter Unit ID (Try 81 or 100)", min_value=1, max_value=100, value=81))

    # --- DEBUGGING SECTION (This will tell us the problem) ---
    st.write("---")
    st.subheader("🛠 Debugging Info")

    # Filter Data (Force Integer Match)
    unit_data = test_df[test_df['unit_number'] == selected_unit]

    st.write(f"**Selected Unit:** {selected_unit}")
    st.write(f"**Total Data Rows Found:** {len(unit_data)}")

    if len(unit_data) == 0:
        st.error("CRITICAL ERROR: No data found for this Unit ID. Check your 'test_FD001.txt' file.")
    elif len(unit_data) < 50:
        st.warning(f"Insufficient Data: This unit only has {len(unit_data)} cycles (Needs 50).")
    else:
        st.success(f"Success! Found {len(unit_data)} cycles. Running Prediction...")

        # --- PREDICTION LOGIC ---
        # Drop Dead Sensors
        drop_cols = ['setting_1', 'setting_2', 'setting_3', 's_1', 's_5', 's_6', 's_10', 's_14', 's_16', 's_18', 's_19']
        features = unit_data.drop(columns=drop_cols)

        # Normalize (Quick Approx)
        cols_norm = features.columns.difference(['unit_number', 'time_cycles'])
        features_norm = features.copy()
        for col in cols_norm:
            features_norm[col] = (features[col] - features[col].min()) / (features[col].max() - features[col].min() + 1e-6)

        # Reshape
        seq = features_norm[cols_norm].values[-50:].reshape(1, 50, len(cols_norm))

        # Predict
        pred_rul = float(model.predict(seq, verbose=0)[0][0])

        # Display Result
        st.metric("Predicted RUL (Cycles Left)", f"{pred_rul:.1f}")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 3))
        # Force white chart background
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#ffffff')

        ax.plot(unit_data['time_cycles'], unit_data['s_11'], label='Pressure Sensor (s_11)', color='blue')
        ax.set_title("Sensor 11 Trends", color='black')
        ax.tick_params(colors='black')
        ax.legend()
        st.pyplot(fig)
