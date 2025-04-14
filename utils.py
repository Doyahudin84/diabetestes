import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def get_feature_importance(model, feature_names):
    """
    Get feature importance from the trained model.
    
    Args:
        model: Trained machine learning model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance
    """
    # Get feature importance from the model
    try:
        importances = model.feature_importances_
        
        # Create a DataFrame for better visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=True)
        
        return importance_df
    except:
        # Return empty DataFrame if model doesn't support feature importance
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.zeros(len(feature_names))
        })

def display_diabetes_info():
    """
    Display educational information about diabetes.
    """
    st.header("Tentang Diabetes")
    
    st.subheader("Apa itu Diabetes?")
    st.write("""
    Diabetes adalah kondisi kesehatan kronis yang mempengaruhi bagaimana tubuh Anda mengubah makanan menjadi energi. 
    Ketika Anda makan, sebagian besar makanan dipecah menjadi glukosa (gula) dan dilepaskan ke aliran darah Anda.
    Pankreas Anda melepaskan insulin, yang bertindak seperti kunci untuk membiarkan gula darah masuk ke sel-sel tubuh Anda untuk digunakan sebagai energi.
    
    Jika Anda memiliki diabetes, tubuh Anda tidak memproduksi cukup insulin atau tidak dapat menggunakan insulin yang dihasilkannya dengan baik.
    Hal ini menyebabkan terlalu banyak gula darah tetap berada di aliran darah, yang dari waktu ke waktu dapat menyebabkan masalah kesehatan serius.
    """)
    
    st.subheader("Jenis-jenis Diabetes")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Diabetes Tipe 1")
        st.write("""
        - Tubuh tidak memproduksi insulin
        - Sistem kekebalan menyerang sel-sel penghasil insulin
        - Biasanya didiagnosis pada anak-anak dan dewasa muda
        - Memerlukan suntikan insulin setiap hari
        """)
    
    with col2:
        st.markdown("### Diabetes Tipe 2")
        st.write("""
        - Tubuh tidak menggunakan insulin dengan baik (resistensi insulin)
        - Jenis diabetes yang paling umum
        - Sering terkait dengan faktor gaya hidup
        - Terkadang dapat dikelola dengan diet dan olahraga
        """)
    
    with col3:
        st.markdown("### Diabetes Gestasional")
        st.write("""
        - Berkembang selama kehamilan
        - Biasanya hilang setelah melahirkan
        - Meningkatkan risiko diabetes tipe 2 di kemudian hari
        - Dapat mempengaruhi kesehatan bayi
        """)
    
    st.subheader("Faktor Risiko untuk Diabetes Tipe 2")
    st.write("""
    - **Riwayat keluarga** dengan diabetes
    - **Usia** - risiko meningkat dengan usia, terutama setelah 45 tahun
    - **Kelebihan berat badan atau obesitas** - terutama dengan kelebihan lemak di sekitar perut
    - **Ketidakaktifan fisik** - kurang dari 3 kali seminggu aktivitas fisik
    - **Tekanan darah tinggi** (140/90 mm Hg atau lebih tinggi)
    - **Kadar kolesterol abnormal** - trigliserida tinggi dan HDL rendah
    - **Riwayat diabetes gestasional** atau melahirkan bayi dengan berat lebih dari 4 kg
    - **Sindrom ovarium polikistik** (PCOS)
    - **Riwayat prediabetes**
    - **Ras/etnis** - risiko lebih tinggi pada orang Afrika, Hispanik/Latino, Penduduk Asli Amerika, Kepulauan Pasifik, dan Asia
    """)
    
    st.subheader("Gejala Diabetes")
    
    symptoms = [
        "Sering buang air kecil",
        "Haus berlebihan",
        "Penurunan berat badan tanpa sebab",
        "Sangat lapar",
        "Perubahan penglihatan tiba-tiba",
        "Kesemutan atau mati rasa di tangan atau kaki",
        "Merasa sangat lelah",
        "Kulit sangat kering",
        "Luka sembuh lambat",
        "Infeksi lebih sering dari biasanya"
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        for symptom in symptoms[:5]:
            st.write(f"• {symptom}")
    
    with col2:
        for symptom in symptoms[5:]:
            st.write(f"• {symptom}")
    
    st.subheader("Pencegahan dan Pengelolaan")
    st.write("""
    ### Mencegah Diabetes Tipe 2
    
    - **Pertahankan berat badan sehat** - Kehilangan bahkan 5-7% berat badan dapat membantu
    - **Aktivitas fisik teratur** - Setidaknya 150 menit per minggu olahraga sedang
    - **Makan sehat** - Fokus pada buah-buahan, sayuran, biji-bijian utuh, protein tanpa lemak
    - **Hindari merokok** - Merokok meningkatkan risiko diabetes dan komplikasinya
    
    ### Mengelola Diabetes
    
    - **Pemantauan gula darah** - Tes rutin sesuai petunjuk dokter Anda
    - **Kepatuhan pengobatan** - Minum obat yang diresepkan secara konsisten
    - **Pemeriksaan kesehatan rutin** - Termasuk pemeriksaan mata, kaki, dan ginjal
    - **Manajemen stres** - Stres dapat mempengaruhi kadar gula darah
    - **Tidur dengan baik** - Usahakan tidur 7-8 jam dengan kualitas baik setiap malam
    """)
    
    st.subheader("Kapan Harus Ke Dokter")
    st.warning("""
    Jika Anda mengalami beberapa gejala diabetes, terutama haus meningkat, sering buang air kecil,
    penglihatan kabur, atau penurunan berat badan tak terduga, segera konsultasikan dengan profesional kesehatan. 
    Diagnosis dan pengobatan dini dapat mencegah komplikasi serius.
    """)
    
    st.info("""
    **Disclaimer**: Aplikasi ini memberikan estimasi risiko diabetes berdasarkan machine learning.
    Ini bukan pengganti saran medis profesional, diagnosis, atau pengobatan.
    Selalu minta saran dari dokter Anda atau penyedia layanan kesehatan berkualifikasi lainnya dengan pertanyaan apa pun
    yang mungkin Anda miliki mengenai kondisi medis.
    """)
