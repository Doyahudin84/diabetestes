import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from model import load_model, predict_diabetes
from preprocessing import preprocess_data, validate_data
from utils import get_feature_importance, display_diabetes_info
from risk_explanation import generate_personalized_explanation
from ai_integration import get_ai_health_advice

# Set page configuration
st.set_page_config(
    page_title="Aplikasi Prediksi Diabetes",
    page_icon="🏥",
    layout="wide"
)

# Initialize model
model, scaler = load_model()

# App title and description
st.title("Aplikasi Prediksi Diabetes")
st.markdown("*by doyahudin*")
st.markdown("""
Aplikasi ini menggunakan machine learning untuk memprediksi kemungkinan diabetes berdasarkan data pasien.
Unggah data Anda atau masukkan informasi pasien secara individual untuk mendapatkan prediksi.
""")

# Sidebar with options
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Menu", ["Beranda", "Unggah Data", "Prediksi Individual", "Tentang Diabetes"])

# Home page
if page == "Beranda":
    st.header("Selamat Datang di Alat Prediksi Diabetes")
    st.write("""
    ### Cara menggunakan aplikasi ini:
    
    1. **Unggah Data**: Unggah file CSV dengan data pasien untuk prediksi batch
    2. **Prediksi Individual**: Masukkan informasi pasien individual untuk prediksi tunggal
    3. **Tentang Diabetes**: Pelajari lebih lanjut tentang diabetes dan faktor risiko
    
    ### Model menggunakan fitur-fitur berikut untuk prediksi:
    
    - Kehamilan: Jumlah kehamilan
    - Glukosa: Konsentrasi glukosa plasma (2 jam dalam tes toleransi glukosa oral)
    - Tekanan Darah: Tekanan darah diastolik (mm Hg)
    - Ketebalan Kulit: Ketebalan lipatan kulit trisep (mm)
    - Insulin: Insulin serum 2 jam (mu U/ml)
    - BMI: Indeks massa tubuh (berat dalam kg/(tinggi dalam m)^2)
    - Fungsi Silsilah Diabetes: Fungsi silsilah diabetes (ukuran riwayat keluarga diabetes)
    - Usia: Usia dalam tahun
    """)
    
    # Sample dataset information
    st.info("""
    Model ini dilatih pada Dataset Diabetes Pima Indians, yang berisi informasi dari
    pasien wanita keturunan Pima Indian. Model mencapai akurasi sekitar 77% pada data uji.
    """)

# Upload Data page
elif page == "Unggah Data":
    st.header("Unggah Data Pasien")
    st.write("Unggah file CSV dengan data pasien untuk mendapatkan prediksi diabetes.")
    
    # Template for download
    st.markdown("""
    ### Format Template CSV
    CSV Anda harus memiliki kolom-kolom berikut:
    - Pregnancies (Kehamilan) - Catatan: Untuk pasien pria, nilai ini harus 0
    - Glucose (Glukosa)
    - BloodPressure (Tekanan Darah)
    - SkinThickness (Ketebalan Kulit)
    - Insulin
    - BMI
    - DiabetesPedigreeFunction (Fungsi Silsilah Diabetes)
    - Age (Usia)
    - Gender (Jenis Kelamin) - Opsional, gunakan 'Pria' atau 'Wanita'
    
    Nilai harus berupa angka kecuali kolom Gender. Nilai yang hilang sebaiknya dibiarkan kosong.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load and display data
            data = pd.read_csv(uploaded_file)
            st.subheader("Pratinjau data yang diunggah:")
            st.write(data.head())
            
            # Check if Gender column exists, if not add a selection
            if 'Gender' not in data.columns:
                st.info("Kolom Gender tidak ditemukan dalam file CSV. Silakan pilih jenis kelamin default.")
                default_gender = st.radio("Jenis Kelamin Default", ["Pria", "Wanita"])
                # Add Gender column to the data
                data['Gender'] = 0 if default_gender == "Pria" else 1
            
            # Data validation
            is_valid, message = validate_data(data)
            
            if not is_valid:
                st.error(message)
            else:
                # Data preprocessing and prediction
                X = preprocess_data(data, scaler)
                
                # Make predictions
                predictions, probabilities = predict_diabetes(model, X)
                
                # Add predictions to the data
                results = data.copy()
                results["Prediksi"] = ["Diabetes" if p == 1 else "Tidak Diabetes" for p in predictions]
                results["Probabilitas"] = ["{:.2f}%".format(p * 100) for p in probabilities]
                
                # Display results
                st.subheader("Hasil Prediksi")
                st.write(results)
                
                # Summary statistics
                st.subheader("Ringkasan Prediksi")
                positive_count = sum(predictions)
                total_count = len(predictions)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Diabetes Terdeteksi", f"{positive_count} dari {total_count} pasien")
                    st.metric("Persentase", f"{(positive_count/total_count)*100:.2f}%")
                
                with col2:
                    # Pie chart of predictions
                    fig = px.pie(
                        names=["Tidak Diabetes", "Diabetes"],
                        values=[(total_count-positive_count), positive_count],
                        title="Distribusi Prediksi",
                        color_discrete_sequence=["#3498db", "#e74c3c"]
                    )
                    st.plotly_chart(fig)
                
                # Feature distributions
                st.subheader("Distribusi Fitur")
                feature_to_plot = st.selectbox(
                    "Pilih fitur untuk visualisasi",
                    data.columns
                )
                
                fig = px.histogram(
                    data, 
                    x=feature_to_plot, 
                    color=results["Prediksi"],
                    barmode="overlay",
                    title=f"Distribusi {feature_to_plot} berdasarkan Prediksi",
                    color_discrete_map={"Diabetes": "#e74c3c", "Tidak Diabetes": "#3498db"}
                )
                st.plotly_chart(fig)
                
                # Feature importance
                st.subheader("Pentingnya Fitur")
                importance_df = get_feature_importance(model, data.columns)
                
                fig = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Pentingnya Fitur untuk Prediksi Diabetes",
                    color="Importance",
                    color_continuous_scale=["#3498db", "#e74c3c"]
                )
                st.plotly_chart(fig)
                
        except Exception as e:
            st.error(f"Error memproses file: {str(e)}")

# Single Prediction page
elif page == "Prediksi Individual":
    st.header("Prediksi Pasien Individual")
    st.write("Masukkan informasi pasien untuk mendapatkan prediksi diabetes.")
    
    # Input form
    with st.form("patient_data_form"):
        # Tambahkan pemilihan jenis kelamin
        gender = st.radio("Jenis Kelamin", ["Pria", "Wanita"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Kehamilan hanya ditampilkan untuk wanita
            if gender == "Wanita":
                pregnancies = st.number_input("Kehamilan", min_value=0, max_value=20, value=0)
            else:
                pregnancies = 0  # Untuk pria, nilai kehamilan adalah 0
                st.info("Kehamilan tidak berlaku untuk pasien pria")
                
            glucose = st.number_input("Glukosa (mg/dL)", min_value=0, max_value=300, value=120)
            blood_pressure = st.number_input("Tekanan Darah (mm Hg)", min_value=0, max_value=200, value=70)
            skin_thickness = st.number_input("Ketebalan Kulit (mm)", min_value=0, max_value=100, value=20)
        
        with col2:
            insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=1000, value=80)
            bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
            diabetes_pedigree = st.number_input("Fungsi Silsilah Diabetes", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
            age = st.number_input("Usia (tahun)", min_value=0, max_value=120, value=30)
        
        submit_button = st.form_submit_button("Prediksi")
    
    if submit_button:
        # Create a dataframe from input
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree],
            'Age': [age],
            'Gender': [0 if gender == "Pria" else 1]  # 0 untuk pria, 1 untuk wanita
        })
        
        # Menyimpan jenis kelamin untuk digunakan dalam penjelasan
        gender_info = gender
        
        # Preprocess input data
        X = preprocess_data(input_data, scaler)
        
        # Make prediction
        prediction, probability = predict_diabetes(model, X)
        
        # Display result
        st.subheader("Hasil Prediksi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction[0] == 1:
                st.error("**Diabetes Terdeteksi**")
            else:
                st.success("**Tidak Terdeteksi Diabetes**")
            
            st.write(f"Probabilitas: {probability[0] * 100:.2f}%")
        
        with col2:
            # Gauge chart for probability
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability[0] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probabilitas Diabetes"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#e74c3c" if prediction[0] == 1 else "#3498db"},
                    'steps': [
                        {'range': [0, 50], 'color': "#3498db"},
                        {'range': [50, 100], 'color': "#e74c3c"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            st.plotly_chart(fig)
        
        # Feature explanation
        st.subheader("Analisis Fitur")
        importance_df = get_feature_importance(model, input_data.columns)
        
        # Horizontal bar chart for feature importance
        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Pentingnya Fitur untuk Prediksi Diabetes",
            color="Importance",
            color_continuous_scale=["#3498db", "#e74c3c"]
        )
        st.plotly_chart(fig)
        
        # Value analysis - compare to normal ranges
        st.subheader("Analisis Nilai")
        
        # Define normal ranges
        normal_ranges = {
            "Glucose": [70, 99],
            "BloodPressure": [60, 80],
            "BMI": [18.5, 24.9],
            "Insulin": [16, 166]
        }
        
        # Create a comparison table
        analysis_data = []
        for feature, value in input_data.iloc[0].items():
            if feature in normal_ranges:
                min_val, max_val = normal_ranges[feature]
                status = "Normal" if min_val <= value <= max_val else "Abnormal"
                normal_range = f"{min_val} - {max_val}"
            else:
                status = "N/A"
                normal_range = "N/A"
            
            analysis_data.append({
                "Fitur": feature,
                "Nilai": value,
                "Rentang Normal": normal_range,
                "Status": status
            })
        
        analysis_df = pd.DataFrame(analysis_data)
        st.table(analysis_df)
        
        # Generate personalized risk explanation
        st.subheader("Penjelasan Risiko Personal")
        
        explanation = generate_personalized_explanation(
            input_data, 
            prediction[0], 
            probability[0], 
            importance_df,
            gender_info
        )
        
        # Display overall explanation
        st.markdown(explanation["overall_message"])
        
        # Create tabs for different sections of the explanation
        tabs = st.tabs(["Faktor Risiko Detail", "Rekomendasi Gaya Hidup", "Analisis AI", "Ringkasan"])
        
        # Detailed risk factors tab
        with tabs[0]:
            st.subheader("Penjelasan Faktor Risiko Utama")
            for i, factor in enumerate(explanation["detailed_factors"]):
                with st.expander(f"{i+1}. {factor['feature']} - Dampak: {factor['importance']:.3f}"):
                    st.markdown(factor["explanation"])
                    
                    # Add color indicators for risk impact
                    if factor["risk_impact"] == "positive":
                        st.markdown("🔴 **Meningkatkan risiko**")
                    elif factor["risk_impact"] == "negative":
                        st.markdown("🟢 **Menurunkan risiko**")
                    else:
                        st.markdown("🟡 **Dampak netral**")
        
        # Lifestyle recommendations tab
        with tabs[1]:
            st.subheader("Rekomendasi Personal")
            for i, recommendation in enumerate(explanation["lifestyle_recommendations"]):
                st.markdown(f"**{i+1}.** {recommendation}")
            
            st.info("Rekomendasi ini hanya saran. Silakan konsultasikan dengan profesional kesehatan untuk saran medis.")
        
        # AI Analysis tab
        with tabs[2]:
            st.subheader("Analisis AI")
            with st.spinner("Menganalisis data dengan AI..."):
                ai_advice = get_ai_health_advice(input_data, explanation["risk_level"], prediction[0])
            
            st.write("Berdasarkan analisis data Anda, berikut rekomendasi kesehatan:")
            st.markdown(ai_advice)
            
            st.info("Catatan: Analisis AI ini menggunakan model bahasa yang didukung oleh Hugging Face. Meskipun berguna, ini bukan pengganti konsultasi dengan profesional kesehatan.")
        
        # Summary tab
        with tabs[3]:
            st.subheader("Ringkasan Penilaian Risiko")
            
            # Display risk level with color coding
            risk_level = explanation["risk_level"]
            risk_level_indo = {
                "Very Low": "Sangat Rendah",
                "Low": "Rendah",
                "Moderate": "Sedang",
                "High": "Tinggi",
                "Very High": "Sangat Tinggi"
            }
            
            risk_colors = {
                "Very Low": "#3498db",  # Blue
                "Low": "#2ecc71",       # Green
                "Moderate": "#f39c12",  # Orange
                "High": "#e74c3c",      # Red
                "Very High": "#c0392b"  # Dark Red
            }
            
            risk_indo = risk_level_indo.get(risk_level, "Sedang")
            
            st.markdown(
                f"<div style='background-color:{risk_colors.get(risk_level, '#f39c12')}; padding:10px; border-radius:5px;'>"
                f"<h3 style='color:white; text-align:center; margin:0;'>Tingkat Risiko: {risk_indo}</h3>"
                f"<p style='color:white; text-align:center; margin:0;'>Probabilitas: {explanation['probability']*100:.1f}%</p>"
                f"</div>",
                unsafe_allow_html=True
            )
            
            # Summary of key points
            st.markdown("### Poin Utama")
            
            # Count positive and negative factors
            positive_factors = sum(1 for f in explanation["detailed_factors"] if f["risk_impact"] == "positive")
            negative_factors = sum(1 for f in explanation["detailed_factors"] if f["risk_impact"] == "negative")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Faktor Risiko", positive_factors)
            with col2:
                st.metric("Faktor Protektif", negative_factors)
        
        st.markdown("### Langkah Selanjutnya")
        if explanation["risk_level"] in ["High", "Very High"]:
            st.warning("Berdasarkan tingkat risiko Anda, kami merekomendasikan untuk berkonsultasi dengan profesional kesehatan segera untuk evaluasi yang tepat.")
        else:
            st.success("Tingkat risiko Anda saat ini menunjukkan bahwa mempertahankan gaya hidup sehat dan pemeriksaan rutin adalah tepat.")

# About Diabetes page
elif page == "Tentang Diabetes":
    display_diabetes_info()

# Footer
st.markdown("---")
st.markdown("Aplikasi Prediksi Diabetes | Didukung oleh Machine Learning")
