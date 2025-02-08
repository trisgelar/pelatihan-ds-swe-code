import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --- Membaca Data ---
df = pd.read_csv('data/error_log_updated.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# --- Persiapan Data ---

# Urutkan data berdasarkan User ID dan Timestamp
df = df.sort_values(['User ID', 'Timestamp'])

# Hitung selisih waktu (dalam jam) ke error berikutnya untuk setiap user
df['Next Error (hours)'] = df.groupby('User ID')['Timestamp'].diff().shift(-1).dt.total_seconds() / 3600

# Hapus baris terakhir untuk setiap user (karena tidak ada error berikutnya)
df = df.dropna(subset=['Next Error (hours)'])

# Tambahkan kolom 'Hour of Day'
df['Hour of Day'] = df['Timestamp'].dt.hour

# Tambahkan kolom 'Day of Week' (0 = Senin, 6 = Minggu)
df['Day of Week'] = df['Timestamp'].dt.dayofweek

# Ubah 'Platform' menjadi numerik
platform_mapping = {'Web': 1, 'Android': 2, 'iOS': 3}
df['Platform Numeric'] = df['Platform'].map(platform_mapping)

# Ubah 'Severity Level' menjadi numerik
severity_level_mapping = {'Debug': 1, 'Info': 2, 'Warning': 3, 'Error': 4, 'Critical': 5, 'Fatal': 6}
df['Severity Level Numeric'] = df['Severity Level'].map(severity_level_mapping)

# Pilih fitur dan target variabel
features = ['Hour of Day', 'Day of Week', 'Platform Numeric', 'Severity Level Numeric']
target = 'Next Error (hours)'

# Pisahkan fitur (X) dan target (y)
X = df[features]
y = df[target]

# --- Membangun Model ---

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model
model = LinearRegression()

# Latih model dengan data latih
model.fit(X_train, y_train)

# --- Implementasi di Streamlit ---

st.header("Prediksi Waktu Error")

# Input fitur
hour_of_day = st.number_input("Jam (0-23):", min_value=0, max_value=23)
day_of_week = st.number_input("Hari dalam Seminggu (0-6):", min_value=0, max_value=6)
platform_numeric = st.selectbox("Platform:", options=list(platform_mapping.keys()))
severity_level_numeric = st.selectbox("Severity Level:", options=list(severity_level_mapping.keys()))

# Prediksi
if st.button("Prediksi")
    # Buat input data untuk prediksi
    input_data = pd.DataFrame({
        'Hour of Day': [hour_of_day],
        'Day of Week': [day_of_week],
        'Platform Numeric': [platform_mapping[platform_numeric]],
        'Severity Level Numeric': [severity_level_mapping[severity_level_numeric]]
    })

    prediksi = model.predict(input_data)[0]
    st.write(f"Prediksi waktu error berikutnya: {prediksi:.2f} jam")