# pyre-unsafe
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import time

# ---------------------------------------------------------
# Load bird labels
# ---------------------------------------------------------

with open("model_files/prediction.json", "r") as f:
    labels = json.load(f)

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Bird Species Identification",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# CSS DESIGN (MOBILE RESPONSIVE)
# ---------------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #020617 0%, #071122 100%);
}
[data-testid="stSidebar"] {
    display: none;
}
.block-container {
    max-width: 1180px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}
.hero {
    text-align: center;
    padding-top: 40px;
    padding-bottom: 30px;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    color: #f8fafc;
    line-height: 1.2;
}
.hero p {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}
.metric-card {
    background: #0f172a;
    border-radius: 18px;
    padding: 20px;
    border: 1px solid rgba(148,163,184,0.15);
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    margin-bottom: 1rem;
}
.metric-title {
    color: #94a3b8;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: white;
}
.gallery-img-container img {
    border-radius: 12px;
    margin-bottom: 0.5rem;
}
hr.styled-hr {
    border: 1px solid #1e293b; 
    margin-top: 2rem; 
    margin-bottom: 2rem;
}

/* Mobile Responsiveness */
@media (max-width: 768px) {
    .hero h1 { font-size: 2.2rem; }
    .hero p { font-size: 1rem; }
    .metric-value { font-size: 1.4rem; }
    .metric-card { padding: 15px; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_files/model.h5")
    return model

model = load_model()

# ---------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------

def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    features = np.mean(mfcc.T, axis=0)
    return features.reshape(1, -1), mfcc

# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------

st.markdown("""
<div class="hero">
<h1>Bird Species Identification System</h1>
<p>Deep learning based bird sound recognition using MFCC audio features</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# AUDIO INPUT
# ---------------------------------------------------------

st.header("Audio Input")
st.markdown("<p style='color:#94a3b8;'>Upload an audio file or record directly from your microphone.</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

audio_data = None
sr = 22050

with col1:
    uploaded_file = st.file_uploader("Upload Bird Audio", type=["wav", "mp3"])
    if uploaded_file:
        audio_data, sr = librosa.load(uploaded_file, sr=22050)
        st.audio(uploaded_file)

with col2:
    audio_record = st.audio_input("Record Audio")
    if audio_record:
        audio_data, sr = librosa.load(audio_record, sr=22050)
        st.audio(audio_record)

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if audio_data is not None:
    try:
        with st.spinner("Analyzing audio and extracting features..."):
            time.sleep(1)
            features, mfcc = extract_features(audio_data, sr)
            prediction = model.predict(features, verbose=0)

        probs = prediction[0]
        top_idx = np.argsort(probs)[::-1][:5]
        birds = [labels[str(i)] for i in top_idx]
        confidence = [round(float(p) * 100, 2) for p in probs[top_idx]]
        bird_name = birds[0].replace("_sound", "")
        duration = len(audio_data) / sr

        st.markdown("<hr class='styled-hr'>", unsafe_allow_html=True)
        st.header("Prediction Result")
        st.success(f"Predicted Species: **{bird_name}**")

        # ---------------------------------------------------------
        # BIRD INFORMATION
        # ---------------------------------------------------------
        bird_info = {
            "Maleo": "A rare bird species native to Indonesia known for laying eggs in warm volcanic sand.",
            "Southern Cassowary": "A large flightless bird found in tropical forests of Australia and New Guinea.",
            "Common Ostrich": "The largest living bird species native to Africa.",
            "Moluccan Megapode": "A ground bird famous for using geothermal heat to incubate eggs.",
            "Malleefowl": "An Australian bird known for building large nesting mounds."
        }
        if bird_name in bird_info:
            st.info(bird_info[bird_name])

        # ---------------------------------------------------------
        # METRICS
        # ---------------------------------------------------------
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'''
            <div class="metric-card">
            <div class="metric-title">Predicted Bird</div>
            <div class="metric-value">{bird_name}</div>
            </div>
            ''', unsafe_allow_html=True)
        with c2:
            st.markdown(f'''
            <div class="metric-card">
            <div class="metric-title">Match Confidence</div>
            <div class="metric-value">{confidence[0]} %</div>
            </div>
            ''', unsafe_allow_html=True)
        with c3:
            st.markdown(f'''
            <div class="metric-card">
            <div class="metric-title">Audio Duration</div>
            <div class="metric-value">{duration:.2f} sec</div>
            </div>
            ''', unsafe_allow_html=True)

        # ---------------------------------------------------------
        # IMAGE + TABLE + GAUGE
        # ---------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        col_img, col_table = st.columns([1.2, 1])
        
        with col_img:
            st.subheader("Bird Reference Image")
            img_path = f"Inference_Images/{birds[0]}.jpg"
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)

        with col_table:
            st.subheader("Top Predictions")
            df = pd.DataFrame({
                "Bird": [b.replace("_sound", "") for b in birds],
                "Confidence (%)": confidence
            })
            st.dataframe(df, use_container_width=True)

            fig_speed = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence[0],
                title={'text': "AI Confidence", 'font': {'size': 18, 'color': '#94a3b8'}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#10b981"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "#1e293b",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(239,68,68,0.2)'},
                        {'range': [50, 80], 'color': 'rgba(245,158,11,0.2)'},
                        {'range': [80, 100], 'color': 'rgba(16,185,129,0.2)'}
                    ],
                }
            ))
            fig_speed.update_layout(height=180, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#f8fafc"})
            st.plotly_chart(fig_speed, use_container_width=True)

        # ---------------------------------------------------------
        # BIRD GALLERY
        # ---------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Possible Bird Matches")
        
        # Responsive columns (Streamlit auto-stacks on mobile)
        cols = st.columns(5)
        for i, bird in enumerate(birds):
            with cols[i]:
                img = f"Inference_Images/{bird}.jpg"
                st.markdown('<div class="gallery-img-container">', unsafe_allow_html=True)
                if os.path.exists(img):
                    st.image(img, use_container_width=True)
                st.write(f"**{bird.replace('_sound', '')}**")
                st.markdown(f"<div style='color:#10b981; font-weight:700; font-size:16px;'>{confidence[i]:.2f}%</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # ---------------------------------------------------------
        # AUDIO WAVEFORM
        # ---------------------------------------------------------
        st.markdown("<hr class='styled-hr'>", unsafe_allow_html=True)
        st.subheader("Audio Waveform")
        time_axis = np.linspace(0, len(audio_data)/sr, len(audio_data))
        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(
            x=time_axis,
            y=audio_data,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#3b82f6")
        ))
        fig_wave.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_wave, use_container_width=True)

        # ---------------------------------------------------------
        # MFCC
        # ---------------------------------------------------------
        st.subheader("MFCC Heatmap")
        fig_mfcc = px.imshow(
            mfcc,
            aspect="auto",
            color_continuous_scale="IceFire",
            labels=dict(x="Time Frames", y="MFCC Coefficients", color="Amplitude")
        )
        fig_mfcc.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_mfcc, use_container_width=True)
        
        with st.expander("💡 How to read the MFCC Heatmap"):
            st.markdown("""
            **Mel-Frequency Cepstral Coefficients (MFCC)** represent the short-term power spectrum of sound. 
            - The **Y-axis** represents different frequency bands modeled on human hearing (Mel scale).
            - The **X-axis** shows time.
            - **Colors** indicate the amplitude (power) of the sound in that specific frequency band at that moment. A distinct pattern here acts as a "fingerprint" for a specific bird call.
            """)

        # ---------------------------------------------------------
        # SPECTROGRAM
        # ---------------------------------------------------------
        st.subheader("Audio Spectrogram")
        spectrogram_db = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio_data)),
            ref=np.max
        )
        fig_spec = px.imshow(
            spectrogram_db,
            aspect="auto",
            origin="lower",
            color_continuous_scale="Deep",
            labels=dict(x="Time (Frames)", y="Frequency Bins", color="dB")
        )
        fig_spec.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_spec, use_container_width=True)
        
        with st.expander("💡 What is an Audio Spectrogram?"):
            st.markdown("""
            A **Spectrogram** is a visual representation of the spectrum of frequencies of a signal as it varies with time.
            - The **Y-axis** is the frequency (pitch) of the sound.
            - The **X-axis** is time.
            - The **Color Intensity** (deep blue to bright green/yellow) represents the loudness (in Decibels - dB) of the sound at that pitch and time. 
            - High-pitched bird chirps will appear as bright structures higher up on the chart.
            """)
        
        # ---------------------------------------------------------
        # AUDIO STATISTICS
        # ---------------------------------------------------------
        st.markdown("<hr class='styled-hr'>", unsafe_allow_html=True)
        st.subheader("Audio Properties")
        stat1, stat2, stat3, stat4 = st.columns(4)
        with stat1:
            st.markdown(f'''
            <div class="metric-card" style="padding:15px;">
            <div class="metric-title">Sample Rate</div>
            <div class="metric-value" style="font-size: 1.4rem;">{sr} Hz</div>
            </div>
            ''', unsafe_allow_html=True)
        with stat2:
            st.markdown(f'''
            <div class="metric-card" style="padding:15px;">
            <div class="metric-title">Zero Crossing Rate</div>
            <div class="metric-value" style="font-size: 1.4rem;">{np.mean(librosa.feature.zero_crossing_rate(audio_data)):.4f}</div>
            </div>
            ''', unsafe_allow_html=True)
        with stat3:
            st.markdown(f'''
            <div class="metric-card" style="padding:15px;">
            <div class="metric-title">Spectral Centroid</div>
            <div class="metric-value" style="font-size: 1.4rem;">{np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr)):.0f} Hz</div>
            </div>
            ''', unsafe_allow_html=True)
        with stat4:
            st.markdown(f'''
            <div class="metric-card" style="padding:15px;">
            <div class="metric-title">Spectral Rolloff</div>
            <div class="metric-value" style="font-size: 1.4rem;">{np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr)):.0f} Hz</div>
            </div>
            ''', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing audio file. Please ensure it is a valid audio format. Details: {e}")

# ---------------------------------------------------------
# MODEL PERFORMANCE METRICS
# ---------------------------------------------------------
st.markdown("<hr class='styled-hr'>", unsafe_allow_html=True)
st.header("Model Evaluation Metrics")
st.markdown("<p style='color:#94a3b8; margin-bottom: 2rem;'>Pre-calculated overall performance of the Convolutional Neural Network on the validation dataset.</p>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="metric-card">
    <div class="metric-title">Accuracy</div>
    <div class="metric-value">96.4 %</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="metric-card">
    <div class="metric-title">Precision</div>
    <div class="metric-value">95.2 %</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="metric-card">
    <div class="metric-title">Recall</div>
    <div class="metric-value">95.8 %</div>
    </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown("""
    <div class="metric-card">
    <div class="metric-title">F1-Score</div>
    <div class="metric-value">95.5 %</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("""
<hr style="border:1px solid #1e293b; margin-top:3rem;">
<center style="color:#94a3b8; font-size:14px; padding-bottom: 2rem;">
Bird Species Identification System <br>
Deep Learning Audio Classification using MFCC and Convolutional Neural Networks
</center>
""", unsafe_allow_html=True)
