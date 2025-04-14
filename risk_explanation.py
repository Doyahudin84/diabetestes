import pandas as pd
import numpy as np

def get_risk_level(probability):
    """
    Convert probability to risk level.
    
    Args:
        probability: Probability of diabetes (0-1)
        
    Returns:
        risk_level: String describing risk level
    """
    if probability < 0.2:
        return "Very Low"
    elif probability < 0.4:
        return "Low"
    elif probability < 0.6:
        return "Moderate"
    elif probability < 0.8:
        return "High"
    else:
        return "Very High"

def analyze_feature_contribution(input_data, importance_df):
    """
    Analyze how each feature contributes to the risk.
    
    Args:
        input_data: DataFrame containing patient data (single row)
        importance_df: DataFrame with feature importance
        
    Returns:
        risk_factors: List of dictionaries with feature analysis
    """
    # Define normal ranges for features
    normal_ranges = {
        "Glucose": {"min": 70, "max": 99, "unit": "mg/dL"},
        "BloodPressure": {"min": 60, "max": 80, "unit": "mm Hg"},
        "BMI": {"min": 18.5, "max": 24.9, "unit": "kg/m²"},
        "Insulin": {"min": 16, "max": 166, "unit": "mu U/ml"},
        "SkinThickness": {"min": 10, "max": 25, "unit": "mm"},
        "Pregnancies": {"min": 0, "max": 10, "unit": ""},
        "DiabetesPedigreeFunction": {"min": 0.0, "max": 0.5, "unit": ""},
        "Age": {"min": 0, "max": 120, "unit": "years"}
    }
    
    risk_factors = []
    
    # Get top features by importance
    sorted_importance = importance_df.sort_values('Importance', ascending=False)
    
    for _, row in sorted_importance.iterrows():
        feature = row['Feature']
        importance = row['Importance']
        value = input_data[feature].values[0]
        
        if feature in normal_ranges:
            normal_range = normal_ranges[feature]
            unit = normal_range["unit"]
            
            # Check if value is outside normal range
            if value < normal_range["min"]:
                status = "Below normal range"
                risk_impact = "negative" if importance < 0.05 else "positive"
            elif value > normal_range["max"]:
                status = "Above normal range"
                risk_impact = "positive" if importance >= 0.05 else "neutral"
            else:
                status = "Within normal range"
                risk_impact = "negative"
                
            risk_factors.append({
                "feature": feature,
                "value": value,
                "unit": unit,
                "status": status,
                "importance": importance,
                "risk_impact": risk_impact,
                "normal_min": normal_range["min"],
                "normal_max": normal_range["max"]
            })
        else:
            risk_factors.append({
                "feature": feature,
                "value": value,
                "unit": "",
                "status": "No normal range defined",
                "importance": importance,
                "risk_impact": "neutral",
                "normal_min": None,
                "normal_max": None
            })
    
    return risk_factors

def get_feature_explanation(feature, status, value, normal_min, normal_max, unit):
    """
    Generate explanation text for a specific feature.
    
    Args:
        feature: Feature name
        status: Status (above/below/within normal range)
        value: Feature value
        normal_min: Minimum normal value
        normal_max: Maximum normal value
        unit: Unit of measurement
        
    Returns:
        explanation: Personalized explanation for this feature
    """
    explanations = {
        "Glucose": {
            "Above normal range": f"Kadar glukosa Anda adalah {value} {unit}, yang di atas rentang normal {normal_min}-{normal_max} {unit}. Glukosa darah tinggi adalah indikator kunci risiko diabetes.",
            "Below normal range": f"Kadar glukosa Anda adalah {value} {unit}, yang di bawah rentang normal {normal_min}-{normal_max} {unit}. Meskipun glukosa rendah biasanya tidak terkait dengan risiko diabetes, ini mungkin mengindikasikan masalah kesehatan lainnya.",
            "Within normal range": f"Kadar glukosa Anda adalah {value} {unit}, yang berada dalam rentang normal {normal_min}-{normal_max} {unit}. Ini adalah faktor positif dalam penilaian risiko diabetes Anda."
        },
        "BloodPressure": {
            "Above normal range": f"Tekanan darah Anda adalah {value} {unit}, yang di atas rentang normal {normal_min}-{normal_max} {unit}. Tekanan darah tinggi sering menyertai dan meningkatkan risiko diabetes.",
            "Below normal range": f"Tekanan darah Anda adalah {value} {unit}, yang di bawah rentang normal {normal_min}-{normal_max} {unit}. Meskipun tekanan darah rendah tidak secara langsung terkait dengan diabetes, ini harus dipantau.",
            "Within normal range": f"Tekanan darah Anda adalah {value} {unit}, yang berada dalam rentang normal {normal_min}-{normal_max} {unit}. Ini adalah faktor positif dalam penilaian risiko diabetes Anda."
        },
        "BMI": {
            "Above normal range": f"BMI Anda adalah {value} {unit}, yang di atas rentang normal {normal_min}-{normal_max} {unit}. BMI yang lebih tinggi dikaitkan dengan peningkatan risiko diabetes.",
            "Below normal range": f"BMI Anda adalah {value} {unit}, yang di bawah rentang normal {normal_min}-{normal_max} {unit}. Meskipun BMI rendah biasanya tidak meningkatkan risiko diabetes, ini mungkin mengindikasikan masalah kesehatan lainnya.",
            "Within normal range": f"BMI Anda adalah {value} {unit}, yang berada dalam rentang normal {normal_min}-{normal_max} {unit}. BMI yang sehat dikaitkan dengan risiko diabetes yang lebih rendah."
        },
        "Insulin": {
            "Above normal range": f"Kadar insulin Anda adalah {value} {unit}, yang di atas rentang normal {normal_min}-{normal_max} {unit}. Kadar insulin tinggi dapat mengindikasikan resistensi insulin, faktor kunci dalam diabetes tipe 2.",
            "Below normal range": f"Kadar insulin Anda adalah {value} {unit}, yang di bawah rentang normal {normal_min}-{normal_max} {unit}. Kadar insulin rendah dapat dikaitkan dengan diabetes tipe 1 atau kondisi metabolik lainnya.",
            "Within normal range": f"Kadar insulin Anda adalah {value} {unit}, yang berada dalam rentang normal {normal_min}-{normal_max} {unit}. Ini adalah faktor positif dalam penilaian risiko diabetes Anda."
        },
        "SkinThickness": {
            "Above normal range": f"Pengukuran ketebalan kulit Anda adalah {value} {unit}, yang di atas rentang normal {normal_min}-{normal_max} {unit}. Ketebalan kulit yang meningkat dapat dikaitkan dengan lemak tubuh yang lebih tinggi dan potensi risiko diabetes.",
            "Below normal range": f"Pengukuran ketebalan kulit Anda adalah {value} {unit}, yang di bawah rentang normal {normal_min}-{normal_max} {unit}.",
            "Within normal range": f"Pengukuran ketebalan kulit Anda adalah {value} {unit}, yang berada dalam rentang normal {normal_min}-{normal_max} {unit}."
        },
        "Pregnancies": {
            "Above normal range": f"Jumlah kehamilan Anda adalah {value}, yang relatif tinggi. Beberapa kehamilan dapat mempengaruhi risiko diabetes, khususnya diabetes gestasional.",
            "Below normal range": f"Jumlah kehamilan Anda adalah {value}.",
            "Within normal range": f"Jumlah kehamilan Anda adalah {value}."
        },
        "DiabetesPedigreeFunction": {
            "Above normal range": f"Fungsi silsilah diabetes Anda adalah {value}, yang di atas rentang tipikal {normal_min}-{normal_max}. Ini menunjukkan riwayat keluarga diabetes yang lebih kuat, meningkatkan risiko Anda.",
            "Below normal range": f"Fungsi silsilah diabetes Anda adalah {value}, yang di bawah rentang tipikal {normal_min}-{normal_max}. Ini menunjukkan riwayat keluarga diabetes yang terbatas.",
            "Within normal range": f"Fungsi silsilah diabetes Anda adalah {value}, yang berada dalam rentang tipikal {normal_min}-{normal_max}."
        },
        "Age": {
            "Above normal range": f"Usia Anda adalah {value} tahun. Risiko diabetes cenderung meningkat dengan usia, terutama setelah 45 tahun.",
            "Below normal range": f"Usia Anda adalah {value} tahun.",
            "Within normal range": f"Usia Anda adalah {value} tahun. Usia adalah faktor dalam risiko diabetes, dengan risiko biasanya meningkat seiring bertambahnya usia."
        }
    }
    
    if feature in explanations and status in explanations[feature]:
        return explanations[feature][status]
    else:
        return f"{feature} Anda adalah {value} {unit}."

def generate_personalized_explanation(input_data, prediction, probability, importance_df, gender):
    """
    Generate personalized explanation of diabetes risk.
    
    Args:
        input_data: DataFrame containing patient data (single row)
        prediction: Binary prediction (0/1)
        probability: Probability of diabetes
        importance_df: DataFrame with feature importance
        gender: String indicating gender ("Pria" or "Wanita")
        
    Returns:
        explanation: Dictionary with personalized explanation
    """
    # Get risk level
    risk_level = get_risk_level(probability)
    
    # Analyze feature contributions
    risk_factors = analyze_feature_contribution(input_data, importance_df)
    
    # Generate overall explanation with gender-specific note
    if gender == "Pria":
        model_note = "Perlu diingat bahwa model ini dilatih terutama dengan data pasien wanita. Hasilnya untuk pasien pria harus diinterpretasikan dengan hati-hati."
    else:
        model_note = ""
        
    if prediction == 1:
        overall_message = f"Berdasarkan input Anda, model kami memprediksi risiko diabetes **{risk_level}** dengan probabilitas {probability*100:.1f}%. Ini berarti Anda memiliki beberapa faktor risiko yang dapat berkontribusi pada perkembangan diabetes. {model_note}"
    else:
        overall_message = f"Berdasarkan input Anda, model kami memprediksi risiko diabetes **{risk_level}** dengan probabilitas {probability*100:.1f}%. Meskipun risiko keseluruhan Anda tampaknya rendah, tetap penting untuk mempertahankan gaya hidup sehat. {model_note}"
    
    # Generate detailed explanations for each factor
    detailed_factors = []
    for factor in risk_factors[:5]:  # Focus on top 5 factors
        # Skip pregnancy factor for men or handle differently
        if gender == "Pria" and factor["feature"] == "Pregnancies":
            continue
            
        explanation = get_feature_explanation(
            factor["feature"], 
            factor["status"], 
            factor["value"],
            factor["normal_min"],
            factor["normal_max"],
            factor["unit"]
        )
        
        # Translate feature names for display
        feature_names_indo = {
            "Pregnancies": "Kehamilan",
            "Glucose": "Glukosa",
            "BloodPressure": "Tekanan Darah",
            "SkinThickness": "Ketebalan Kulit",
            "Insulin": "Insulin",
            "BMI": "BMI",
            "DiabetesPedigreeFunction": "Fungsi Silsilah Diabetes",
            "Age": "Usia"
        }
        
        display_name = feature_names_indo.get(factor["feature"], factor["feature"])
        
        detailed_factors.append({
            "feature": display_name,
            "explanation": explanation,
            "importance": factor["importance"],
            "risk_impact": factor["risk_impact"]
        })
    
    # Generate lifestyle recommendations based on gender
    lifestyle_recommendations = []
    
    # Check glucose levels
    glucose = input_data["Glucose"].values[0]
    if glucose > 99:
        lifestyle_recommendations.append("Pantau kadar glukosa darah Anda secara teratur dan pertimbangkan untuk berkonsultasi dengan penyedia layanan kesehatan.")
        lifestyle_recommendations.append("Kurangi asupan karbohidrat sederhana dan gula dalam makanan Anda.")
    
    # Check BMI
    bmi = input_data["BMI"].values[0]
    if bmi > 25:
        lifestyle_recommendations.append(f"BMI Anda {bmi:.1f} menunjukkan Anda mungkin mendapat manfaat dari strategi manajemen berat badan.")
        lifestyle_recommendations.append("Targetkan setidaknya 150 menit aktivitas fisik sedang per minggu.")
    
    # Add gender-specific recommendations
    if gender == "Wanita":
        # Add female-specific recommendations
        pregnancy_count = input_data["Pregnancies"].values[0]
        if pregnancy_count > 0:
            lifestyle_recommendations.append("Wanita dengan riwayat kehamilan memiliki risiko diabetes yang berbeda. Diskusikan riwayat kehamilan Anda dengan dokter Anda.")
        
        lifestyle_recommendations.append("Wanita dengan riwayat diabetes gestasional harus melakukan pemeriksaan diabetes secara teratur.")
    
    # Add general recommendations
    lifestyle_recommendations.append("Tetap terhidrasi dengan minum banyak air sepanjang hari.")
    lifestyle_recommendations.append("Sertakan lebih banyak makanan kaya serat seperti sayuran, buah-buahan, dan biji-bijian utuh dalam makanan Anda.")
    lifestyle_recommendations.append("Batasi konsumsi alkohol dan hindari merokok.")
    lifestyle_recommendations.append("Kelola stres melalui aktivitas seperti meditasi, yoga, atau teknik relaksasi lainnya.")
    
    # Return complete explanation
    return {
        "prediction": "Diabetes" if prediction == 1 else "Tidak Diabetes",
        "probability": probability,
        "risk_level": risk_level,
        "overall_message": overall_message,
        "detailed_factors": detailed_factors,
        "lifestyle_recommendations": lifestyle_recommendations
    }