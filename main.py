import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from scipy.signal import butter, filtfilt


#************************************
# TODO:
# - Askelmäärä laskettuna suodatetusta kiihtyvyysdatasta --------------------- DONE
# - Askelmäärä laskettuna kiihtyvyysdatasta Fourier-analyysin perusteella ---- DONE
# - Keskinopeus (GPS-datasta) ------------------------------------------------ DONE
# - Kuljettu matka (GPS-datasta) --------------------------------------------- DONE
# - Askelpituus (lasketun askelmäärän ja matkan perusteella) ----------------- DONE

# - Suodatettu kiihtyvyysdata, jota käytit askelmäärän määrittelemiseen. ----- DONE
# - Analyysiin valitun kiihtyvyysdatan komponentin tehospektritiheys --------- DONE
# - Reittisi kartalla -------------------------------------------------------- DONE

# Notes: Since Streamlit was really slow to render anything, the code was optimized by AI to make it much faster. After optimization, the app runs smoothly giving the correct results.
#        Streamlit caching does not support DataFrames directly, heavy computations are cached using @st.cache_data.
#************************************

#************************************
# Paths
walkPath = "https://raw.githubusercontent.com/Rieskamies/Walk-Analyzer/refs/heads/main/data/walkData.csv"
gpsPath  = "https://raw.githubusercontent.com/Rieskamies/Walk-Analyzer/refs/heads/main/data/gpsData.csv"


# Load data
walk_df = pd.read_csv(walkPath)
gps_df = pd.read_csv(gpsPath)

#************************************
# Helper functions
def butter_lowpass_filter(data, cutoff, fs, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Cache heavy computations for speed
@st.cache_data
def process_walk_data(walk_df):
    data = walk_df['Linear Acceleration y (m/s^2)']
    T_tot = walk_df['Time (s)'].max()
    n = len(walk_df)
    fs = n / T_tot if T_tot != 0 else 0
    nyq = fs / 2 if fs != 0 else 0
    order = 3
    cutoff = 1 / 0.3

    # Filter signal
    data_filt = butter_lowpass_filter(data, cutoff, fs, nyq, order)
    walk_df['Filtered Accel Y'] = data_filt

    # Step count
    jaksot = np.sum(data_filt[:-1] * data_filt[1:] < 0) * 0.5

    # Fourier-based step count
    signal = data_filt - np.mean(data_filt)
    N = len(signal)
    dt = 1 / fs
    fourier = np.fft.fft(signal, N)
    psd = fourier * np.conj(fourier) / N
    freq = np.fft.fftfreq(N, dt)
    L = np.arange(1, int(N/2))
    freqs = freq[L]
    psd_vals = np.abs(psd[L])
    valid = (freqs >= 0.5) & (freqs <= 4.0)
    freqs_valid = freqs[valid]
    psd_valid = psd_vals[valid]
    dominant_freq = freqs_valid[np.argmax(psd_valid)]
    steps_fft = dominant_freq * T_tot

    # Power spectral density for Linear Acceleration Y
    psd_freq = freqs_valid
    psd_power = psd_valid

    return walk_df, jaksot, steps_fft, psd_freq, psd_power

@st.cache_data
def process_gps_data(gps_df):
    gps_filtered = gps_df[gps_df['Horizontal Accuracy (m)'] <= 10].copy()
    gps_filtered['Latitude (°)'] = gps_filtered['Latitude (°)'].rolling(3, center=True).mean().fillna(gps_filtered['Latitude (°)'])
    gps_filtered['Longitude (°)'] = gps_filtered['Longitude (°)'].rolling(3, center=True).mean().fillna(gps_filtered['Longitude (°)'])

    dx = (gps_filtered['Longitude (°)'].diff()) * 111320 * np.cos(np.radians(gps_filtered['Latitude (°)']))
    dy = gps_filtered['Latitude (°)'].diff() * 110540
    distances = np.sqrt(dx**2 + dy**2)

    gps_filtered['Distance (km)'] = distances.cumsum() / 1000
    total_distance = gps_filtered['Distance (km)'].iloc[-1]

    return gps_filtered, total_distance

#************************************
# Process data
walk_df, jaksot, steps_fft, psd_freq, psd_power = process_walk_data(walk_df)
gps_filtered, total_distance = process_gps_data(gps_df)
step_length_m = total_distance * 1000 / jaksot

#************************************
# Summary metrics at the top
st.subheader("Mittauksien yhteenveto")

col1, col2, col3 = st.columns(3)
col1.metric("Keskinopeus", f"{round(gps_filtered['Velocity (m/s)'].mean(),2)} m/s")
col2.metric("Kokonaismatka", f"{round(total_distance,2)} km")
col3.metric("Askelpituus", f"{round(step_length_m,2)} m")

col4, col5 = st.columns(2)
col4.metric("Askelmäärä (Suodatettu)", f"{int(jaksot)} steps")
col5.metric("Askelmäärä (Fourier)", f"{int(steps_fft)} steps")

#************************************
# Walk data
walk_df_plot = walk_df[(walk_df['Time (s)'] >= 20) & (walk_df['Time (s)'] <= 400)].iloc[::5, :]
st.subheader("Kiihtyvyysdata (Y) ja suodatettu kiihtyvyysdata (Y)")
st.line_chart(
    walk_df_plot,
    x='Time (s)',
    y=['Linear Acceleration y (m/s^2)', 'Filtered Accel Y']
)

#************************************
# Power Spectral Density
st.subheader("Tehospektri kiihtyvyysdatalle (Y)")
psd_df = pd.DataFrame({
    "Frequency (Hz)": psd_freq,
    "Power": psd_power
})
st.line_chart(
    psd_df,
    x="Frequency (Hz)",
    y="Power"
)

#************************************
# GPS map
st.subheader("GPS-reitti kartalla")
start_lat = gps_filtered['Latitude (°)'].iloc[0]
start_long = gps_filtered['Longitude (°)'].iloc[0]
map = folium.Map(location=[start_lat, start_long], zoom_start=14)
coords = gps_filtered.iloc[::5][['Latitude (°)', 'Longitude (°)']].dropna()
folium.PolyLine(coords, color='blue', weight=3.5, opacity=1).add_to(map)
st_map = st_folium(map, width=900, height=650)
