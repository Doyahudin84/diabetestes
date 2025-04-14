import requests
import json
import streamlit as st

# Hugging Face Inference API - ini API gratis dengan batasan penggunaan
# Tidak memerlukan API key untuk beberapa model dasar
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/"

def get_ai_health_advice(input_data, risk_level, prediction):
    """
    Mendapatkan saran kesehatan dari model AI menggunakan Hugging Face API.
    
    Args:
        input_data: Data pasien
        risk_level: Tingkat risiko diabetes
        prediction: Prediksi diabetes (0/1)
        
    Returns:
        advice: Saran kesehatan dari AI
    """
    # Model yang digunakan (bisa diganti dengan model lain)
    model = "google/flan-t5-small"  # Model AI yang lebih kecil dan gratis untuk digunakan
    
    # Memformat prompt untuk model
    if prediction == 1:
        prompt = f"""
        Sebagai ahli kesehatan, berikan 5 saran medis untuk seseorang dengan risiko diabetes {risk_level}.
        
        Data pasien:
        - Glukosa: {input_data['Glucose'].values[0]} mg/dL
        - Tekanan Darah: {input_data['BloodPressure'].values[0]} mm Hg
        - BMI: {input_data['BMI'].values[0]}
        - Usia: {input_data['Age'].values[0]} tahun
        - Jenis Kelamin: {"Pria" if input_data['Gender'].values[0] == 0 else "Wanita"}
        
        Berikan saran yang spesifik dan praktis dalam bahasa Indonesia.
        """
    else:
        prompt = f"""
        Sebagai ahli kesehatan, berikan 5 saran pencegahan diabetes untuk seseorang dengan risiko diabetes {risk_level}.
        
        Data pasien:
        - Glukosa: {input_data['Glucose'].values[0]} mg/dL
        - Tekanan Darah: {input_data['BloodPressure'].values[0]} mm Hg
        - BMI: {input_data['BMI'].values[0]}
        - Usia: {input_data['Age'].values[0]} tahun
        - Jenis Kelamin: {"Pria" if input_data['Gender'].values[0] == 0 else "Wanita"}
        
        Berikan saran yang spesifik dan praktis dalam bahasa Indonesia.
        """
    
    try:
        # API call ke Hugging Face Inference API
        api_url = f"{HUGGINGFACE_API_URL}{model}"
        response = requests.post(
            api_url,
            json={"inputs": prompt},
            headers={"Content-Type": "application/json"}
        )
        
        # Periksa respon
        if response.status_code == 200:
            advice = response.json()[0]["generated_text"]
            return advice
        else:
            # Fallback jika API tidak tersedia
            return get_fallback_advice(input_data, risk_level, prediction)
    except Exception as e:
        print(f"Error saat mengakses AI: {str(e)}")
        return get_fallback_advice(input_data, risk_level, prediction)

def get_fallback_advice(input_data, risk_level, prediction):
    """
    Memberikan saran kesehatan yang telah diprogram sebagai fallback.
    """
    gender = "Pria" if input_data['Gender'].values[0] == 0 else "Wanita"
    bmi = input_data['BMI'].values[0]
    glucose = input_data['Glucose'].values[0]
    age = input_data['Age'].values[0]
    
    # Saran berdasarkan risiko
    if prediction == 1:
        # Saran untuk pasien berisiko diabetes
        advice = "Saran Kesehatan Untuk Risiko Diabetes:\n\n"
        
        # BMI
        if bmi > 25:
            advice += "1. Menurunkan berat badan sangat penting untuk Anda. Targetkan penurunan 5-10% dari berat badan saat ini.\n"
        else:
            advice += "1. Pertahankan berat badan ideal Anda dengan pola makan seimbang dan aktivitas fisik rutin.\n"
        
        # Glukosa
        if glucose > 120:
            advice += "2. Kadar glukosa Anda cukup tinggi. Batasi konsumsi karbohidrat sederhana dan gula olahan.\n"
        else:
            advice += "2. Pantau kadar glukosa darah Anda secara rutin untuk memastikannya tetap dalam rentang normal.\n"
        
        # Umur
        if age > 45:
            advice += "3. Pada usia Anda, pemeriksaan kesehatan lengkap setiap 6 bulan sangat direkomendasikan.\n"
        else:
            advice += "3. Lakukan pemeriksaan kesehatan tahunan dan tes glukosa darah puasa secara rutin.\n"
        
        # Gender
        if gender == "Wanita":
            advice += "4. Wanita dengan riwayat diabetes gestasional perlu pemantauan lebih ketat. Konsultasikan dengan dokter Anda tentang kebutuhan spesifik Anda.\n"
        else:
            advice += "4. Pria cenderung kurang rutin memeriksakan kesehatan. Jadwalkan pemeriksaan rutin dengan dokter Anda.\n"
        
        # Umum
        advice += "5. Konsumsi makanan tinggi serat, rendah lemak, dan batasi alkohol. Lakukan aktivitas fisik setidaknya 150 menit per minggu."
    else:
        # Saran untuk pencegahan diabetes
        advice = "Saran Pencegahan Diabetes:\n\n"
        
        # BMI
        if bmi > 25:
            advice += "1. Berat badan Anda di atas ideal. Program penurunan berat badan moderat dapat mengurangi risiko diabetes hingga 58%.\n"
        else:
            advice += "1. Pertahankan berat badan ideal Anda untuk mencegah risiko diabetes di masa depan.\n"
        
        # Glukosa
        if glucose > 100:
            advice += "2. Kadar glukosa Anda mendekati batas prediabetes. Kurangi konsumsi gula dan karbohidrat olahan.\n"
        else:
            advice += "2. Kadar glukosa Anda normal. Pertahankan dengan diet seimbang dan aktivitas fisik.\n"
        
        # Umur
        if age > 40:
            advice += "3. Risiko diabetes meningkat seiring usia. Mulai skrining diabetes secara teratur.\n"
        else:
            advice += "3. Bangun kebiasaan sehat sejak dini untuk mengurangi risiko diabetes di masa depan.\n"
        
        # Gender
        if gender == "Wanita":
            advice += "4. Hormon pada wanita dapat mempengaruhi sensitivitas insulin. Pertahankan aktivitas fisik rutin untuk kesehatan metabolisme.\n"
        else:
            advice += "4. Pria memiliki risiko diabetes pada usia lebih muda. Batasi konsumsi alkohol dan makanan tinggi lemak jenuh.\n"
        
        # Umum
        advice += "5. Konsumsi diet kaya serat, sayuran, dan protein tanpa lemak. Hindari merokok dan batasi alkohol."
    
    return advice

def add_ai_section_to_app(st, input_data, prediction, probability, risk_level):
    """
    Tambahkan bagian AI ke aplikasi Streamlit
    """
    st.subheader("💡 Analisis AI")
    
    with st.spinner("Menganalisis data dengan AI..."):
        ai_advice = get_ai_health_advice(input_data, risk_level, prediction)
    
    st.write("Berdasarkan analisis data Anda, berikut rekomendasi kesehatan:")
    st.markdown(ai_advice)
    
    st.info("Catatan: Analisis AI ini menggunakan model bahasa yang didukung oleh Hugging Face. Meskipun berguna, ini bukan pengganti konsultasi dengan profesional kesehatan.")