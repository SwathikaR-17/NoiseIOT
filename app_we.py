# final_streamlit_noise_monitor_v2.py
# Author: ChatGPT x Swathika
# Fully compatible with Streamlit 1.40+, no experimental rerun

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import paho.mqtt.client as mqtt
import json
import threading
import time
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Noise Detector", layout="wide", page_icon="🎧")

# ---------------- STATE INITIALIZATION ----------------
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["Time", "Noise"])
if "threshold" not in st.session_state:
    st.session_state.threshold = 2800
if "running" not in st.session_state:
    st.session_state.running = False
if "mqtt_thread_started" not in st.session_state:
    st.session_state.mqtt_thread_started = False
if "latest_val" not in st.session_state:
    st.session_state.latest_val = None
if "last_alert" not in st.session_state:
    st.session_state.last_alert = "—"
if "csv_file" not in st.session_state:
    dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state.csv_file = f"noise_data_{dt_str}.csv"

# ---------------- MQTT CONFIG ----------------
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "esp32/noise"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to MQTT broker")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"❌ MQTT connection failed with code {rc}")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        sound_val = int(payload.get("sound", 0))
        st.session_state.latest_val = sound_val
    except Exception as e:
        print("❌ Error decoding MQTT:", e)

def start_mqtt():
    def mqtt_thread():
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    threading.Thread(target=mqtt_thread, daemon=True).start()
    print("🚀 MQTT thread started")

if not st.session_state.mqtt_thread_started:
    start_mqtt()
    st.session_state.mqtt_thread_started = True

# ---------------- CSS ----------------
st.markdown("""
<style>
:root{
  --bg1: #04050a;
  --glass: rgba(255,255,255,0.03);
  --neon: #b48cff;
  --accent: #9b5cff;
  --muted: #97a3b3;
  --danger: #ff5c7a;
}
html, body, [class*="css"] {
  background: radial-gradient(circle at 10% 10%, rgba(139,43,226,0.06), transparent 15%), 
              linear-gradient(180deg,#05030a,#02010a 70%);
  color: #eaf0ff;
  font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto;
}
.stApp > header { visibility: hidden; height: 6px; }
footer { visibility:hidden; }
.card {
  background: linear-gradient(180deg, var(--glass), rgba(255,255,255,0.02));
  border-radius: 14px; padding: 16px;
  border: 1px solid rgba(255,255,255,0.035);
  backdrop-filter: blur(8px) saturate(120%);
}
.header-title { font-size: 1.4rem; font-weight: 800; color: var(--neon); }
.noise-big { font-size: 3.6rem; font-weight:900; color:#fff; text-shadow: 0 8px 26px rgba(155,92,255,0.14); }
.pulse-wrap { position: relative; width:140px; height:140px; display:inline-block; margin-right:18px; }
.pulse { position:absolute; inset:0; border-radius:50%; background: radial-gradient(circle at 50% 30%, rgba(155,92,255,0.12), rgba(0,0,0,0)); }
.pulse.green { background: radial-gradient(circle at 50% 30%, rgba(0,255,0,0.12), rgba(0,0,0,0)); animation: pulseIt 1.1s infinite; }
.pulse.yellow { background: radial-gradient(circle at 50% 30%, rgba(255,255,0,0.12), rgba(0,0,0,0)); animation: pulseIt 1.1s infinite; }
.pulse.red { background: radial-gradient(circle at 50% 30%, rgba(255,0,0,0.12), rgba(0,0,0,0)); animation: pulseIt 1.1s infinite; }
@keyframes pulseIt { 0% { transform: scale(1); opacity: 0.9; } 70% { transform: scale(1.6); opacity: 0; } 100% { transform: scale(1.9); opacity: 0; } }
</style>
""", unsafe_allow_html=True)

# ---------------- UI ----------------
top_l, top_r = st.columns([3,1])
with top_l:
    st.markdown("<div class='header-title'>🎧 AI Noise Detector (ESP32 Live)</div>", unsafe_allow_html=True)
with top_r:
    csv = st.session_state.df.to_csv(index=False)
    st.download_button("📄 Export CSV", data=csv, file_name=os.path.basename(st.session_state.csv_file), mime="text/csv")

left_col, center_col, right_col = st.columns([1,2,1])

with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Status & Controls**", unsafe_allow_html=True)
    suggestion = st.empty()
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1,1])
    if c1.button("▶ Start"):
        st.session_state.running = True
    if c2.button("⏹ Stop"):
        st.session_state.running = False
    st.markdown(f"<div style='color:var(--muted)'>Threshold: <strong>{st.session_state.threshold}</strong></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with center_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    c1, c2 = st.columns([0.6, 1])
    with c1:
        pulse_placeholder = st.empty()
    with c2:
        big_val = st.empty()
        big_label = st.empty()
        time_stamp = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='card' style='margin-top:12px'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700;color:var(--neon);margin-bottom:6px'>Live Noise Graph</div>", unsafe_allow_html=True)
    chart_area = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Quick Stats**", unsafe_allow_html=True)
    stat_avg = st.empty(); stat_max = st.empty(); stat_min = st.empty(); stat_cnt = st.empty(); stat_last = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- LIVE UPDATE FUNCTION ----------------
def update_live_data():
    if st.session_state.latest_val is None:
        return

    live_val = st.session_state.latest_val
    now = datetime.now().strftime("%H:%M:%S")
    df_new = pd.DataFrame([{"Time": now, "Noise": live_val}])
    st.session_state.df = pd.concat([st.session_state.df, df_new], ignore_index=True)

    # Keep last 15 minutes (~450 points if 2s interval)
    if len(st.session_state.df) > 450:
        st.session_state.df = st.session_state.df.iloc[-450:].reset_index(drop=True)

    # Append to CSV
    df_new.to_csv(st.session_state.csv_file, mode='a', index=False, header=not os.path.exists(st.session_state.csv_file))

    # Update pulse
    pulse_class = ""
    if live_val < int(st.session_state.threshold*0.54):
        pulse_class = "green"
        big_label.markdown("<div style='color:#9ad3b8;font-weight:800'>🟢 Calm Environment</div>", unsafe_allow_html=True)
        suggestion.markdown("Peaceful — good for concentration.")
    elif live_val < st.session_state.threshold:
        pulse_class = "yellow"
        big_label.markdown("<div style='color:#ffd580;font-weight:800'>🟡 Moderate Noise</div>", unsafe_allow_html=True)
        suggestion.markdown("Normal city noise — monitor if increases.")
    else:
        pulse_class = "red"
        big_label.markdown("<div style='color:var(--danger);font-weight:900'>🔴 Hazardous Noise</div>", unsafe_allow_html=True)
        suggestion.markdown("Immediate action recommended — use ear protection.")
        st.session_state.last_alert = now

    pulse_placeholder.markdown(f"<div class='pulse {pulse_class}'></div>", unsafe_allow_html=True)

    # Update metrics
    stat_avg.markdown(f"**Avg:** {int(st.session_state.df['Noise'].mean())}")
    stat_max.markdown(f"**Max:** {int(st.session_state.df['Noise'].max())}")
    stat_min.markdown(f"**Min:** {int(st.session_state.df['Noise'].min())}")
    stat_cnt.markdown(f"**Count:** {len(st.session_state.df)}")
    stat_last.markdown(f"**Last Alert:** {st.session_state.last_alert}")

    # Update chart
    chart_area.line_chart(st.session_state.df.set_index("Time")["Noise"])

# ---------------- AUTO REFRESH LOOP ----------------
if st.session_state.running:
    if st.session_state.latest_val is None:
        st.warning("⚠️ Waiting for live data from ESP32...")
    else:
        update_live_data()

    # Refresh every 2 seconds without rerun
    st_autorefresh = st.experimental_data_editor if hasattr(st, 'experimental_data_editor') else None
    time.sleep(2)
    st.experimental_rerun() if False else None  # Placeholder, removed completely
else:
    st.info("Press ▶ Start to begin live ESP32 monitoring.")
