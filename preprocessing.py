import pandas as pd
import numpy as np

def validate_data(data):
    """
    Validate input data format and values.
    
    Args:
        data: DataFrame containing patient data
        
    Returns:
        is_valid: Boolean indicating if data is valid
        message: Error message if data is invalid
    """
    # Check required columns
    required_columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        return False, f"Missing columns: {', '.join(missing_columns)}"
    
    # Check data types - all required columns should be numeric
    non_numeric_cols = [col for col in required_columns if not pd.api.types.is_numeric_dtype(data[col])]
    if non_numeric_cols:
        return False, f"Non-numeric columns: {', '.join(non_numeric_cols)}"
        
    # Check Gender column if present (can be non-numeric)
    if 'Gender' in data.columns and not pd.api.types.is_numeric_dtype(data['Gender']):
        # Convert Gender strings to numeric values if needed
        if data['Gender'].dtype == 'object':
            try:
                # Try to convert strings like "Pria"/"Wanita" to 0/1
                gender_map = {'Pria': 0, 'Wanita': 1, 'Male': 0, 'Female': 1, '0': 0, '1': 1}
                data['Gender'] = data['Gender'].map(gender_map).astype(float)
            except:
                return False, "Gender column must contain values: 'Pria', 'Wanita', 'Male', 'Female', 0, or 1"
    
    # Check for non-negative values
    negative_value_cols = []
    for col in required_columns:
        if (data[col] < 0).any():
            negative_value_cols.append(col)
    
    if negative_value_cols:
        return False, f"Negative values found in: {', '.join(negative_value_cols)}"
    
    # Check value ranges - basic sanity checks
    if (data['Pregnancies'] > 25).any():
        return False, "Invalid pregnancy count (>25)"
    
    if (data['Glucose'] > 500).any():
        return False, "Invalid glucose values (>500 mg/dL)"
    
    if (data['BloodPressure'] > 300).any():
        return False, "Invalid blood pressure values (>300 mm Hg)"
    
    if (data['BMI'] > 100).any():
        return False, "Invalid BMI values (>100)"
    
    if (data['Age'] > 120).any():
        return False, "Invalid age values (>120 years)"
    
    return True, "Data validation successful"

def preprocess_data(data, scaler):
    """
    Preprocess data for the model.
    
    Args:
        data: DataFrame containing patient data
        scaler: StandardScaler for feature scaling
        
    Returns:
        X_scaled: Preprocessed features ready for prediction
    """
    # Create a copy to avoid modifying the original data
    df = data.copy()
    
    # Handle Gender column if present
    has_gender = 'Gender' in df.columns
    
    # If Gender is not present, add it (default to female)
    if not has_gender:
        df['Gender'] = 1  # Default to female if gender is not specified
    
    # Convert Gender to numeric if it's not already
    if df['Gender'].dtype == 'object':
        gender_map = {'Pria': 0, 'Wanita': 1, 'Male': 0, 'Female': 1, '0': 0, '1': 1}
        df['Gender'] = df['Gender'].map(gender_map).fillna(1).astype(float)
    
    # Required features
    required_columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Gender'
    ]
    
    # Ensure all required columns are present
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0  # Default placeholder for missing columns
    
    # Handle missing values (replace 0 with median)
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        # Replace 0 with NaN
        df[col] = df[col].replace(0, np.nan)
        # Replace NaN with median
        median_val = df[col].median()
        if np.isnan(median_val):  # If all values are NaN, use reasonable defaults
            if col == 'Glucose':
                median_val = 120
            elif col == 'BloodPressure':
                median_val = 70
            elif col == 'SkinThickness':
                median_val = 20
            elif col == 'Insulin':
                median_val = 80
            elif col == 'BMI':
                median_val = 25
        df[col] = df[col].fillna(median_val)  # Fixed deprecated warning
    
    # Extract features in the correct order
    X = df[required_columns]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled
