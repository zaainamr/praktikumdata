import pickle
from pathlib import Path

import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_FILES = {
    "lr_model": MODEL_DIR / "linear_regression_model.pkl",
    "scaler_lr": MODEL_DIR / "scaler_lr.pkl",
    "nb_model": MODEL_DIR / "naive_bayes_model.pkl",
    "scaler_nb": MODEL_DIR / "scaler_nb.pkl",
}

st.set_page_config(
    page_title="Dashboard RS Sehat Sentosa",
    page_icon=":material/local_hospital:",
    layout="wide",
)

@st.cache_resource
def load_models():
    try:
        lr_model = pickle.load(open(MODEL_FILES["lr_model"], "rb"))
        scaler_lr = pickle.load(open(MODEL_FILES["scaler_lr"], "rb"))
        nb_model = pickle.load(open(MODEL_FILES["nb_model"], "rb"))
        scaler_nb = pickle.load(open(MODEL_FILES["scaler_nb"], "rb"))
        return lr_model, scaler_lr, nb_model, scaler_nb
    except FileNotFoundError as exc:
        st.error(
            ":material/warning: File model tidak ditemukan. Pastikan folder 'models/' berisi file .pkl yang diperlukan."
        )
        st.stop()

lr_model, scaler_lr, nb_model, scaler_nb = load_models()

st.sidebar.title(":material/dashboard: Navigasi Utama")
st.sidebar.markdown("Silakan pilih menu layanan:")
page = st.sidebar.radio(
    "Menu",
    [
        ":material/home: Beranda",
        ":material/water_drop: Prediksi Glukosa (Regresi)",
        ":material/stethoscope: Klasifikasi Diabetes",
    ],
)
st.sidebar.markdown("---")
st.sidebar.info(
    ":material/info: Dashboard analitik ini dirancang khusus untuk tim medis RS Sehat Sentosa guna deteksi dini risiko diabetes."
)

if page == ":material/home: Beranda":
    st.title(":material/local_hospital: Sistem Cerdas Deteksi Dini Diabetes")
    st.markdown("### Selamat Datang di Dashboard Rumah Sakit Sehat Sentosa")
    st.write(
        """
    Aplikasi ini ditenagai oleh model **Machine Learning** terintegrasi untuk membantu tenaga medis:

    1. **Prediksi Glukosa:** Mengestimasi kadar glukosa darah pasien secara kuantitatif berdasarkan profil klinis (Algoritma *Linear Regression*).
    2. **Klasifikasi Risiko:** Memprediksi apakah pasien berisiko tinggi terkena diabetes atau tidak (Algoritma *Naive Bayes + SMOTE*).

    :material/arrow_back: *Silakan pilih menu pada panel navigasi di sebelah kiri untuk mulai menggunakan fitur.*
    """
    )

elif page == ":material/water_drop: Prediksi Glukosa (Regresi)":
    st.title(":material/water_drop: Prediksi Kadar Glukosa Darah")
    st.markdown("Geser slider di bawah ini untuk memasukkan data klinis pasien dan melihat estimasi kadar glukosa darah.")

    with st.form("form_regresi"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader(":material/person: Data Diri")
            age = st.slider("Umur (Tahun)", min_value=1, max_value=100, value=30)
            pregnancies = st.slider("Jumlah Kehamilan", min_value=0, max_value=20, value=0)
            bmi = st.slider("BMI (Indeks Massa Tubuh)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
            bp = st.slider("Tekanan Darah", min_value=50.0, max_value=200.0, value=120.0, step=1.0)
            hba1c = st.slider("HbA1c (%)", min_value=3.0, max_value=15.0, value=5.5, step=0.1)

        with col2:
            st.subheader(":material/biotech: Profil Lipid & Fisik")
            ldl = st.slider("Kadar LDL", min_value=0.0, max_value=300.0, value=100.0, step=1.0)
            hdl = st.slider("Kadar HDL", min_value=0.0, max_value=150.0, value=50.0, step=1.0)
            trig = st.slider("Triglycerides", min_value=0.0, max_value=500.0, value=150.0, step=1.0)
            wc = st.slider("Lingkar Pinggang (cm)", min_value=30.0, max_value=150.0, value=90.0, step=1.0)
            hc = st.slider("Lingkar Pinggul (cm)", min_value=30.0, max_value=150.0, value=100.0, step=1.0)
            whr = st.slider("Waist-to-Hip Ratio (WHR)", min_value=0.5, max_value=1.5, value=0.9, step=0.01)

        with col3:
            st.subheader(":material/medical_information: Riwayat Klinis")
            fh_val = st.select_slider("Riwayat Keluarga Diabetes", options=["Tidak", "Ya"])
            dt_val = st.select_slider("Tipe Diet Terjaga", options=["Tidak", "Ya"])
            ht_val = st.select_slider("Hipertensi", options=["Tidak", "Ya"])
            meds_val = st.select_slider("Penggunaan Obat", options=["Tidak", "Ya"])
            out_val = st.select_slider("Status Diabetes Saat Ini", options=["Negatif", "Positif"])

            fh = 1 if fh_val == "Ya" else 0
            dt = 1 if dt_val == "Ya" else 0
            ht = 1 if ht_val == "Ya" else 0
            meds = 1 if meds_val == "Ya" else 0
            outcome = 1 if out_val == "Positif" else 0

        st.markdown("---")
        submit_button = st.form_submit_button(label=":material/analytics: Hitung Prediksi Glukosa")

        if submit_button:
            input_data = np.array([[age, pregnancies, bmi, bp, hba1c, ldl, hdl, trig, wc, hc, whr, fh, dt, ht, meds, outcome]])
            input_scaled = scaler_lr.transform(input_data)
            prediksi = lr_model.predict(input_scaled)

            st.success(f"### :material/science: Estimasi Kadar Glukosa Darah: **{prediksi[0]:.2f} mg/dL**")
            if prediksi[0] > 140:
                st.warning(":material/warning: Perhatian: Prediksi glukosa berada di atas ambang batas normal, disarankan pemeriksaan lebih lanjut.")

elif page == ":material/stethoscope: Klasifikasi Diabetes":
    st.title(":material/stethoscope: Deteksi Dini Risiko Diabetes")
    st.markdown("Geser parameter pasien di bawah ini untuk melihat probabilitas risiko tinggi terkena diabetes.")

    with st.form("form_klasifikasi"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader(":material/vital_signs: Data Klinis Utama")
            age = st.slider("Umur (Tahun)", min_value=1, max_value=100, value=35)
            pregnancies = st.slider("Jumlah Kehamilan", min_value=0, max_value=20, value=0)
            bmi = st.slider("BMI (Indeks Massa Tubuh)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
            glucose = st.slider("Kadar Glukosa", min_value=50.0, max_value=300.0, value=110.0, step=1.0)
            bp = st.slider("Tekanan Darah", min_value=50.0, max_value=200.0, value=120.0, step=1.0)
            hba1c = st.slider("HbA1c (%)", min_value=3.0, max_value=15.0, value=5.5, step=0.1)

        with col2:
            st.subheader(":material/biotech: Data Penunjang")
            ldl = st.slider("Kadar LDL", min_value=0.0, max_value=300.0, value=100.0, step=1.0)
            hdl = st.slider("Kadar HDL", min_value=0.0, max_value=150.0, value=50.0, step=1.0)
            trig = st.slider("Triglycerides", min_value=0.0, max_value=500.0, value=150.0, step=1.0)
            wc = st.slider("Lingkar Pinggang (cm)", min_value=30.0, max_value=150.0, value=90.0, step=1.0)
            hc = st.slider("Lingkar Pinggul (cm)", min_value=30.0, max_value=150.0, value=100.0, step=1.0)
            whr = st.slider("Waist-to-Hip Ratio (WHR)", min_value=0.5, max_value=1.5, value=0.9, step=0.01)

        with col3:
            st.subheader(":material/medical_information: Riwayat Klinis")
            fh_val = st.select_slider("Riwayat Keluarga Diabetes", options=["Tidak", "Ya"])
            dt_val = st.select_slider("Tipe Diet Terjaga", options=["Tidak", "Ya"])
            ht_val = st.select_slider("Hipertensi", options=["Tidak", "Ya"])
            meds_val = st.select_slider("Penggunaan Obat", options=["Tidak", "Ya"])

            fh = 1 if fh_val == "Ya" else 0
            dt = 1 if dt_val == "Ya" else 0
            ht = 1 if ht_val == "Ya" else 0
            meds = 1 if meds_val == "Ya" else 0

        st.markdown("---")
        submit_button = st.form_submit_button(label=":material/search: Jalankan Analisis Risiko")

        if submit_button:
            input_data = np.array([[age, pregnancies, bmi, glucose, bp, hba1c, ldl, hdl, trig, wc, hc, whr, fh, dt, ht, meds]])
            input_scaled = scaler_nb.transform(input_data)
            hasil = nb_model.predict(input_scaled)

            if hasil[0] == 1:
                st.error("### :material/emergency: HASIL: PASIEN BERISIKO TINGGI (DIABETES)")
                st.write("Pasien menunjukkan indikasi kuat penyakit diabetes. Segera jadwalkan konsultasi lanjutan.")
            else:
                st.success("### :material/check_circle: HASIL: PASIEN BERISIKO RENDAH (NORMAL)")
                st.write("Pasien dalam kondisi aman dan tidak menunjukkan pola risiko diabetes.")
