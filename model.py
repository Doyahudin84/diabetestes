import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def create_and_train_model():
    """
    Create and train a diabetes prediction model using both male and female data.
    Returns the trained model and scaler.
    """
    try:
        # Pima Indians Diabetes Dataset (female data)
        url_female = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        column_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ]
        
        # Load female dataset
        df_female = pd.read_csv(url_female, names=column_names)
        
        # Add gender column (1 for female)
        df_female['Gender'] = 1
        
        # Generate data for males based on general population statistics
        # Note: This is a simplified approach and ideally should be replaced with real male dataset
        # We'll create male data by:
        # 1. Setting all Pregnancies to 0
        # 2. Adjusting other features based on general population differences
        
        # Create a copy of female data and adjust for males
        df_male = df_female.copy()
        df_male['Pregnancies'] = 0  # Males cannot be pregnant
        df_male['Gender'] = 0  # 0 for male
        
        # Adjust other features based on general population differences
        # Note: These adjustments are approximate and should be replaced with real data
        df_male['Glucose'] = df_male['Glucose'] * np.random.uniform(0.95, 1.05, len(df_male))
        df_male['BloodPressure'] = df_male['BloodPressure'] * np.random.uniform(1.0, 1.1, len(df_male))  # Males tend to have slightly higher BP
        df_male['BMI'] = df_male['BMI'] * np.random.uniform(0.95, 1.05, len(df_male))
        df_male['Insulin'] = df_male['Insulin'] * np.random.uniform(0.9, 1.1, len(df_male))
        
        # Combine datasets
        df = pd.concat([df_female, df_male], ignore_index=True)
        
        # Replace zero values with NaN for certain columns
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_cols:
            df[col] = df[col].replace(0, np.nan)
        
        # Simple imputation - replace NaN with median
        for col in zero_cols:
            # Fix deprecated warning by using new approach
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        
        # Split data
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df[['Gender', 'Outcome']])
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        
        # Evaluate model performance by gender
        male_mask = X_test['Gender'] == 0
        female_mask = X_test['Gender'] == 1
        
        male_accuracy = accuracy_score(y_test[male_mask], y_pred[male_mask])
        female_accuracy = accuracy_score(y_test[female_mask], y_pred[female_mask])
        
        print(f"Male Accuracy: {male_accuracy:.4f}")
        print(f"Female Accuracy: {female_accuracy:.4f}")
        
        return model, scaler
        
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        # Return a simple fallback model if there's an error
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        scaler = StandardScaler()
        return model, scaler

def load_model():
    """
    Load or create the diabetes prediction model.
    Checks if a pre-trained model exists and loads it, otherwise creates a new one.
    """
    model_path = "diabetes_model.pkl"
    scaler_path = "diabetes_scaler.pkl"
    
    # Check if model files exist
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            # Load the pre-trained model and scaler
            print("Loading pre-trained model and scaler...")
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
            with open(scaler_path, 'rb') as scaler_file:
                scaler = pickle.load(scaler_file)
            return model, scaler
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # If loading fails, create a new model
            return create_and_train_and_save_model()
    else:
        # Create and save a new model
        return create_and_train_and_save_model()

def create_and_train_and_save_model():
    """
    Create, train and save the model and scaler.
    """
    model, scaler = create_and_train_model()
    
    # Save the model and scaler
    try:
        print("Saving trained model and scaler...")
        with open("diabetes_model.pkl", 'wb') as model_file:
            pickle.dump(model, model_file)
        with open("diabetes_scaler.pkl", 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)
    except Exception as e:
        print(f"Error saving model: {str(e)}")
    
    return model, scaler

def predict_diabetes(model, X):
    """
    Make diabetes predictions using the trained model.
    
    Args:
        model: Trained machine learning model
        X: Preprocessed features
        
    Returns:
        predictions: Binary predictions (0/1)
        probabilities: Probability of positive class
    """
    # Get predictions
    predictions = model.predict(X)
    
    # Get probabilities for the positive class (diabetes)
    probabilities = model.predict_proba(X)[:, 1]
    
    return predictions, probabilities
