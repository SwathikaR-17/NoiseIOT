import streamlit as st
import paho.mqtt.client as mqtt
import json
import threading
import pandas as pd
from datetime import datetime
import os
import numpy as np

# --- MQTT SETTINGS ---
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "esp32/noise"

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Noise Monitor", layout="wide")
st.title("🔊 Live Noise Monitor Dashboard")

# --- CSS for Glass Effect ---
st.markdown(
    """
    <style>
    .glass-card {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        color: #ffffff;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True
)

# --- Shared State ---
latest_sound = {"value": None}
if "chart_data" not in st.session_state:
    st.session_state.chart_data = []

# --- CSV Logging ---
csv_file = f"noise_data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to MQTT broker")
        client.subscribe(MQTT_TOPIC)
    else:
        print("❌ Failed to connect")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        sound_val = payload.get("sound", None)
        if sound_val is not None:
            latest_sound["value"] = int(sound_val)
    except Exception as e:
        print("Error decoding:", e)

# --- MQTT Client Thread ---
def mqtt_thread():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()

threading.Thread(target=mqtt_thread, daemon=True).start()

# --- Rule-based Noise Class ---
def get_noise_class(val):
    if val <= 40:
        return "Quiet", "#00FF00"       # Green
    elif val <= 70:
        return "Moderate", "#FFFF00"    # Yellow
    elif val <= 90:
        return "Loud", "#FFA500"        # Orange
    else:
        return "Very Loud", "#FF0000"   # Red

# --- Layout: Left (Graph) / Right (Metrics) ---
col1, col2 = st.columns([2, 1])

# --- Main Loop (using Streamlit auto-update via button press or rerun) ---
val = latest_sound["value"]

if val is not None:
    # Update chart data
    st.session_state.chart_data.append(val)
    if len(st.session_state.chart_data) > 30:
        st.session_state.chart_data = st.session_state.chart_data[-30:]

    # CSV Logging
    df_new = pd.DataFrame([{
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Noise_dB": val
    }])
    df_new.to_csv(csv_file, mode='a', index=False, header=not os.path.exists(csv_file))

    # --- Graph ---
    with col1:
        st.markdown('<div class="glass-card"><h3>Noise Level (Last 30 Readings)</h3></div>', unsafe_allow_html=True)
        st.line_chart(st.session_state.chart_data)

    # --- Metrics + Noise Class ---
    avg_val = np.mean(st.session_state.chart_data)
    max_val = np.max(st.session_state.chart_data)
    min_val = np.min(st.session_state.chart_data)
    noise_class, color = get_noise_class(val)

    with col2:
        st.markdown(f'<div class="glass-card"><h3>Metrics</h3></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="glass-card"><p>Average: <span class="metric-value">{avg_val:.1f} dB</span></p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="glass-card"><p>Max: <span class="metric-value">{max_val:.1f} dB</span></p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="glass-card"><p>Min: <span class="metric-value">{min_val:.1f} dB</span></p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="glass-card"><p>Noise Class: <span class="metric-value" style="color:{color}">{noise_class}</span></p></div>', unsafe_allow_html=True)
else:
    st.warning("⚠️ Waiting for live data from ESP32...")
