# run_all.py
import numpy as np
import joblib
from data_preprocessing import preprocess_data
from feature_engineering import enhance_features
from train_model import train_models  # Ensure you have this script and function
from model_evaluation import evaluate_models  # Ensure you have this script and function

def main():
    dataset_path = 'data.csv'  # Update with your dataset path

    try:
        X_train, X_test, y_train, y_test = preprocess_data(dataset_path)
        print("Data preprocessing completed successfully.")
    except Exception as e:
        print(f"Failed during data preprocessing: {e}")
        return

    try:
        X_train, X_test = enhance_features(X_train, X_test)
        print("Feature engineering completed successfully.")
    except Exception as e:
        print(f"Failed during feature engineering: {e}")
        return

    try:
        models = train_models(X_train, y_train, X_test, y_test)  # Assumes returning a dict of model paths
        print("Models training completed successfully.")
    except Exception as e:
        print(f"Failed during model training: {e}")
        return

    try:
        for model_name, model_path in models.items():
            print(f"Evaluating model: {model_name}")
            model = joblib.load(model_path)
            evaluate_models(model, X_test, y_test)
        print("Model evaluation completed successfully.")
    except Exception as e:
        print(f"Failed during model evaluation: {e}")
        return

if __name__ == "__main__":
    main()
